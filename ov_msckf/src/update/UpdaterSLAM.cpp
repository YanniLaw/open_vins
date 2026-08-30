/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "UpdaterSLAM.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterSLAM::UpdaterSLAM(UpdaterOptions &options_slam, UpdaterOptions &options_aruco, ov_core::FeatureInitializerOptions &feat_init_options)
    : _options_slam(options_slam), _options_aruco(options_aruco) {

  // Save our raw pixel noise squared
  _options_slam.sigma_pix_sq = std::pow(_options_slam.sigma_pix, 2);
  _options_aruco.sigma_pix_sq = std::pow(_options_aruco.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

/**
 * @brief Perform delayed initialization of features
 * delayed_init 是 SLAM 特征"转正"的仪式——把一批三角化好的候选特征，正式加入滤波器的状态向量（成为"路标"）。
 * 一个特征被跟踪足够久、三角化足够好之后，调用 delayed_init 把它从"临时测量轨迹"升级成状态向量里的一个路标（带位置均值 + 协方差），
 * 从此每帧都能用它约束未来位姿——这就是 MSCKF→SLAM 的"延迟初始化"。
 * 为什么需要"延迟初始化"?
 * 往状态向量里加一个新变量，不仅要它的初值，还必须有它跟其他状态之间的协方差（交叉项）。
 * 一帧两帧的测量给不出可靠的协方差 → 所以必须"延迟"：等特征积累足够多观测、三角化稳定后再一次性初始化。这就是 delayed_init 存在的意义
 *******************************
 MSCKF特征与SLAM特征的区别：
	                    MSCKF 特征	                      SLAM 特征
1.进状态向量吗？	 不进，用完即弃（零空间投影消掉）	      进（加入 _features_SLAM）
2.生命周期	            一帧用完就删	                  长期存活，直到被边缘化
3.作用	              只约束当前这批克隆	           持续约束未来位姿（长期回环/漂移抑制）
4.代价	                状态向量小	                     状态向量变大
********************************
 * @note 这个函数只做"延迟初始化"的工作，不做 EKF 更新。EKF 更新在 update() 里做。
 * @param state The current state of the filter
 * @param feature_vec The vector of features to be initialized
 */
void UpdaterSLAM::delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature 清理掉不在克隆时刻上的观测
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements 统计该特征的总测量数
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  // 创建相机位姿克隆图：遍历每个相机 × 每个克隆时刻，计算相机在世界系下的位姿
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  // 三角化+高斯牛顿精化：遍历每个特征，尝试三角化出 3D 位置，再用高斯牛顿精化。失败（退化、射线平行）→ 剔除，避免带病特征进更新。
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // 4. Compute linear system for each feature, nullspace project, and reject
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    // 利用ID大小区分 ArUco 特征和 SLAM 特征，分别使用不同的表示方式
    auto feat_rep =
        ((int)feat.featid < state->_options.max_aruco_features) ? state->_options.feat_rep_aruco : state->_options.feat_rep_slam;
    feat.feat_representation = feat_rep;
    // 单深度表示（SINGLE）在构造雅可比时被临时当成 MSCKF 逆深度——因为 SINGLE 只差"把 bearing 边缘化掉"这一步，后面单独处理。
    if (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;  // 观测对特征的雅可比矩阵（残差对特征的偏导数） 2m x 3 或 2m x 1
    Eigen::MatrixXd H_x;  // 观测对状态的雅可比矩阵（残差对状态的偏导数） 2m x 已有状态维数
    Eigen::VectorXd res;  // 观测残差向量（实际观测 - 预测观测）
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature 构造雅可比矩阵和残差向量
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // If we are doing the single feature representation, then we need to remove the bearing portion
    // To do so, we project the bearing portion onto the state and depth Jacobians and the residual.
    // This allows us to directly initialize the feature as a depth-old feature
    // 为什么要这样做？
    // 单深度表示的路标在状态里只有 1 个自由度（深度ρ），bearing（2 维）在初始化时就被边缘化固定了（不再估计）。所以
    // 1. 把深度的雅可比抽出来，和状态雅可比拼在一起（H_xf）；
    // 2. 剩下的 bearing 雅可比（H_f）用零空间投影消掉——这一步至关重要，它承认"bearing 不是精确已知的，
    // 我们也一并把它边缘化掉了"，保证估计器一致性（否则会过于乐观）
    // 3. 投影完再拆开：H_x = 对已有状态，H_f = 对新路标的 1 维深度
    if (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

      // Append the Jacobian in respect to the depth of the feature
      // 把深度(1维)的雅可比并到 H_xf 末尾，H_f 只留 bearing(2维)
      Eigen::MatrixXd H_xf = H_x;
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
      H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1);
      H_f.conservativeResize(H_f.rows(), H_f.cols() - 1);

      // Nullspace project the bearing portion
      // This takes into account that we have marginalized the bearing already
      // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
      // 把 bearing 部分零空间投影掉
      UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

      // Split out the state portion and feature portion
      // 再拆回: H_x = 状态部分, H_f = 深度部分
      H_x = H_xf.block(0, 0, H_xf.rows(), H_xf.cols() - 1);
      H_f = H_xf.block(0, H_xf.cols() - 1, H_xf.rows(), 1);
    }

    // Create feature pointer (we will always create it of size three since we initialize the single invese depth as a msckf anchored
    // representation)
    // 创建路标对象：如果是单深度表示，路标在状态里只有 1 个自由度（深度ρ），否则是 3 个自由度（xyz）。
    // 注意这里的路标对象只是一个临时对象，后续要把它加入滤波器状态。
    int landmark_size = (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 1 : 3;
    auto landmark = std::make_shared<Landmark>(landmark_size);
    landmark->_featid = feat.featid;
    landmark->_feat_representation = feat_rep;
    landmark->_unique_camera_id = (*it2)->anchor_cam_id;
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      landmark->_anchor_cam_id = feat.anchor_cam_id;
      landmark->_anchor_clone_timestamp = feat.anchor_clone_timestamp;
      landmark->set_from_xyz(feat.p_FinA, false);
      landmark->set_from_xyz(feat.p_FinA_fej, true);
    } else {
      landmark->set_from_xyz(feat.p_FinG, false);
      landmark->set_from_xyz(feat.p_FinG_fej, true);
    }
    // 路标的初值来自三角化结果（p_FinA/p_FinG），当前值和 FEJ 值都存好。之后 initialize 会用观测把它精化

    // Measurement noise matrix
    double sigma_pix_sq =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    Eigen::MatrixXd R = sigma_pix_sq * Eigen::MatrixXd::Identity(res.rows(), res.rows());

    // Try to initialize, delete new pointer if we failed
    double chi2_multipler =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    // MSCKF后面会消掉H_f，而SLAM不会，而且会把H_f用来初始化路标的协方差和交叉项
    if (StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, R, res, chi2_multipler)) {
      state->_features_SLAM.insert({(*it2)->featid, landmark});
      (*it2)->to_delete = true;
      it2++;
    } else {
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
    }
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  if (!feature_vec.empty()) {
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds total\n", (rT3 - rT1).total_microseconds() * 1e-6);
  }
}

/**
 * @brief Update the SLAM state with the given feature measurements.
 * 用已经在状态向量里的 SLAM 特征（landmark）的最新观测，对滤波器状态做 EKF 更新
 * @param state The current SLAM state.
 * @param feature_vec The vector of feature measurements to update the state with.
 */
void UpdaterSLAM::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Get the landmark and its representation
    // For single depth representation we need at least two measurement
    // This is because we do nullspace projection
    // 按照特征表示区分最小观测数:
    // 1. 单深度表示的路标在状态里只有 1 个自由度（深度ρ），bearing（2 维）在初始化时就被边缘化了。
    //    更新时需要对观测做零空间投影消掉 bearing 自由度，至少要有 2 个观测才够
    // 2. 其他表示的路标在状态里有 3 个自由度（xyz），只要有 1 个观测就能约束它。
    std::shared_ptr<Landmark> landmark = state->_features_SLAM.at((*it0)->featid);
    int required_meas = (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 2 : 1;

    // Remove if we don't have enough
    if (ct_meas < 1) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0); // 该特征彻底没用了，这一轮也不用
    } else if (ct_meas < required_meas) {
      it0 = feature_vec.erase(it0); // 仅从本轮移除（不标删除），等下次观测凑够了再更新
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      // 每个观测提供 2 个残差（像素坐标 u,v），所以总残差维数 = 2 * 所有特征的观测数
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  size_t max_hx_size = state->max_covariance_size(); // 当前完整协方差的维数

  // Large Jacobian, residual, and measurement noise of *all* features for this update
  // 预分配大矩阵（批量更新的核心）
  // 为什么一次性分配满？ 
  // 因为 EKF 更新要把所有通过检验的特征拼接成一个大的线性系统，一次性 EKFUpdate，比逐特征更新快很多，
  // 且保证一致性（同一批观测共享同一个线性化点）
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size); // 所有特征的总残差
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); // 对状态的总雅可比
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size); // 总测量噪声
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping; // 是"状态对象 → 在大 Hx_big 中的列偏移"的记账表
  std::vector<std::shared_ptr<Type>> Hx_order_big; // 记录变量出现的顺序。同一个变量（比如某个克隆位姿 PoseJPL）被多个特征观测到，只登记一次列区间，后续追加到对应的列块。
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject
  // 逐特征构造雅可比 → 卡方检验 → 拼接
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Ensure we have the landmark and it is the same
    // 检查是否在状态向量里有这个路标，且 featid 一致
    assert(state->_features_SLAM.find((*it2)->featid) != state->_features_SLAM.end());
    assert(state->_features_SLAM.at((*it2)->featid)->_featid == (*it2)->featid);

    // Get our landmark from the state
    std::shared_ptr<Landmark> landmark = state->_features_SLAM.at((*it2)->featid);

    // Convert the state landmark into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = landmark->_feat_representation;
    if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = landmark->_anchor_cam_id;
      feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
      feat.p_FinA = landmark->get_xyz(false);
      feat.p_FinA_fej = landmark->get_xyz(true);
    } else {
      feat.p_FinG = landmark->get_xyz(false);
      feat.p_FinG_fej = landmark->get_xyz(true);
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    // 1. 统计涉及的状态变量（克隆位姿、路标）
    // 2. 将特征位置统一到全局系
    // 3. 计算残差向量以及对状态的雅可比矩阵
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
    // 拼接成 H_xf（路标已在状态里，直接并入状态雅可比）
    Eigen::MatrixXd H_xf = H_x;
    if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

      // Append the Jacobian in respect to the depth of the feature
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
      H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1); // 抽深度列
      H_f.conservativeResize(H_f.rows(), H_f.cols() - 1); // 剩 bearing 列

      // Nullspace project the bearing portion
      // This takes into account that we have marginalized the bearing already
      // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
      UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res); // 消 bearing

    } else {

      // Else we have the full feature in our state, so just append it
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
      H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;
    }

    // Append to our Jacobian order vector  // 把该特征对应的路标也加进状态列表
    std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
    Hxf_order.push_back(landmark);

    // Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
    Eigen::MatrixXd S = H_xf * P_marg * H_xf.transpose();
    double sigma_pix_sq =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    S.diagonal() += sigma_pix_sq * Eigen::VectorXd::Ones(S.rows()); // 新息协方差
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    // ArUco 特征: 打印警告剔除本次观测，不算失败计数
    // 普通 SLAM 特征: 累计失败次数，后续可能用于淘汰该路标，同样从本轮剔除
    double chi2_multipler =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    if (chi2 > chi2_multipler * chi2_check) {
      if ((int)feat.featid < state->_options.max_aruco_features) {
        PRINT_WARNING(YELLOW "[SLAM-UP]: rejecting aruco tag %d for chi2 thresh (%.3f > %.3f)\n" RESET, (int)feat.featid, chi2,
                      chi2_multipler * chi2_check);
      } else {
        landmark->update_fail_count++;
      }
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      continue;
    }

    // Debug print when we are going to update the aruco tags
    if ((int)feat.featid < state->_options.max_aruco_features) {
      PRINT_DEBUG("[SLAM-UP]: accepted aruco tag %d for chi2 thresh (%.3f < %.3f)\n", (int)feat.featid, chi2, chi2_multipler * chi2_check);
    }

    // We are good!!! Append to our large H vector
    // 通过检验的特征，拼入大矩阵
    size_t ct_hx = 0;
    for (const auto &var : Hxf_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});  // 新变量登记列偏移
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_xf.rows(), var->size()) = H_xf.block(0, ct_hx, H_xf.rows(), var->size());
      ct_hx += var->size();
    }

    // Our isotropic measurement noise
    R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) *= sigma_pix_sq;

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);
  R_big.conservativeResize(ct_meas, ct_meas);

  // 5. With all good SLAM features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[SLAM-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[SLAM-UP]: %.4f seconds creating linear system\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size(),
            (int)Hx_big.rows());
  PRINT_ALL("[SLAM-UP]: %.4f seconds total\n", (rT3 - rT1).total_microseconds() * 1e-6);
}

/**
 * @brief 
 * 背景：什么是"锚点"，为什么需要换锚点？
  在 OpenVINS 里，锚点式表示（如 ANCHORED_INVERSE_DEPTH、ANCHORED_INVERSE_DEPTH_SINGLE、
  ANCHORED_MSCKF_INVERSE_DEPTH）的路标不直接存全局坐标，而是存相对于某个"锚点相机位姿"
  （_anchor_clone_timestamp + _anchor_cam_id）的坐标（如逆深度 ρ 或归一化坐标）。
 * 问题：状态里维持一个滑窗克隆。最老的克隆会被边缘化（marginalize）掉。
  如果某个路标的锚点正好在那只要被边缘化的老克隆上，锚点坐标系就要消失了—— 
  必须在边缘化之前把锚点"搬到"一个还活着的克隆（默认是最新的 IMU 克隆）上，
  否则该路标就无法再被观测方程使用（update() 里取锚点位姿会访问一个已被移除的克隆）。

  这正是 change_anchors 存在的意义：在边缘化发生前，把锚点落在"将死"克隆上的路标，重新锚定到新克隆。
  不是重新估计 landmark，而是在旧 anchor clone 被滑窗删除前，对 anchored SLAM landmark 
  做一次“带协方差和 FEJ 的严格状态重参数化”，从而让这个 landmark 能继续留在 EKF 里长期使用
  调用时机：VioManager::do_feature_propagate_update() 在 StateHelper::marginalize_old_clone() 之前调用。
 * @param state 
 */
void UpdaterSLAM::change_anchors(std::shared_ptr<State> state) {

  // Return if we do not have enough clones
  // 1. 克隆数量没超过上限，说明还不会边缘化，直接返回
  if ((int)state->_clones_IMU.size() <= state->_options.max_clone_size) {
    return;
  }

  // Get the marginalization timestep, and change the anchor for any feature seen from it
  // NOTE: for now we have anchor the feature in the same camera as it is before
  // NOTE: this also does not change the representation of the feature at all right now
  // 2. 拿到即将被边缘化的时间戳
  double marg_timestep = state->margtimestep();
  // 3. 遍历所有 SLAM 路标
  for (auto &f : state->_features_SLAM) {
    // Skip any features that are in the global frame
    // 跳过全局系表示（GLOBAL_3D / GLOBAL_FULL_INVERSE_DEPTH）
    // 它们不依赖任何锚点，不需要换
    if (f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D ||
        f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH)
      continue;
    // Else lets see if it is anchored in the clone that will be marginalized
    // 若该路标的锚点正好在被边缘化的克隆时刻上
    assert(marg_timestep <= f.second->_anchor_clone_timestamp);
    if (f.second->_anchor_clone_timestamp == marg_timestep) {
      // 搬到最新时刻的克隆，保持相机 id 不变
      perform_anchor_change(state, f.second, state->_timestamp, f.second->_anchor_cam_id);
    }
  }
}

/**
 * @brief Change the anchor of a given SLAM landmark to a new clone and camera.
 * 它的任务：把路标的位置值（含 FEJ 值）和协方差从"旧锚点表示"变换到"新锚点表示"。
 * 核心思想是：同一个物理点新旧表示下是同一个量，只是参数化不同，所以可以先求新旧表示之间的误差传播雅可比Φ，
 * 再用 Φ 做一次"协方差传播"
  1. 计算旧锚点相机位姿（R_GtoOLD, p_OLDinG）和新锚点相机位姿（R_GtoNEW, p_NEWinG）
  2. 计算旧锚点到新锚点的变换（R_OLDtoNEW, p_OLDinNEW）
  3. 用变换把路标从旧锚点坐标系变换到新锚点坐标系，更新路标的 _anchor_clone_timestamp、_anchor_cam_id、p_FinA
 * 
 * @param state 主滤波器
 * @param landmark 待换锚点的路标
 * @param new_anchor_timestamp 新锚点的时间戳(状态最新的时间戳)
 * @param new_cam_id 新锚点的相机 id
 */
void UpdaterSLAM::perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<Landmark> landmark, double new_anchor_timestamp,
                                        size_t new_cam_id) {

  // Assert that this is an anchored representation
  assert(LandmarkRepresentation::is_relative_representation(landmark->_feat_representation));
  assert(landmark->_anchor_cam_id != -1);

  // Create current feature representation
  UpdaterHelper::UpdaterHelperFeature old_feat;
  old_feat.featid = landmark->_featid;
  old_feat.feat_representation = landmark->_feat_representation;
  old_feat.anchor_cam_id = landmark->_anchor_cam_id;
  old_feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
  old_feat.p_FinA = landmark->get_xyz(false);
  old_feat.p_FinA_fej = landmark->get_xyz(true);

  // Get Jacobians of p_FinG wrt old representation
  Eigen::MatrixXd H_f_old; // 全局特征点对旧特征参数的雅可比矩阵（残差对旧特征参数的偏导数） 3 x 3 或 3 x 1
  std::vector<Eigen::MatrixXd> H_x_old; // 全局特征点对旧锚点相关状态的雅可比矩阵（残差对状态的偏导数） 3 x 已有状态维数
  std::vector<std::shared_ptr<Type>> x_order_old;
  UpdaterHelper::get_feature_jacobian_representation(state, old_feat, H_f_old, H_x_old, x_order_old);

  // Create future feature representation
  UpdaterHelper::UpdaterHelperFeature new_feat;
  new_feat.featid = landmark->_featid;
  new_feat.feat_representation = landmark->_feat_representation;
  new_feat.anchor_cam_id = new_cam_id;
  new_feat.anchor_clone_timestamp = new_anchor_timestamp;

  //==========================================================================
  //==========================================================================
  // 把位置值从旧锚点变换到新锚点（均值 + FEJ）

  // OLD: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoIOLD = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot();
  Eigen::Matrix<double, 3, 3> R_GtoOLD = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot() * R_GtoIOLD;
  Eigen::Matrix<double, 3, 1> p_OLDinG = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos() -
                                         R_GtoOLD.transpose() * state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

  // NEW: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoINEW = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot();
  Eigen::Matrix<double, 3, 3> R_GtoNEW = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot() * R_GtoINEW;
  Eigen::Matrix<double, 3, 1> p_NEWinG = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos() -
                                         R_GtoNEW.transpose() * state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

  // Calculate transform between the old anchor and new one
  Eigen::Matrix<double, 3, 3> R_OLDtoNEW = R_GtoNEW * R_GtoOLD.transpose();
  Eigen::Matrix<double, 3, 1> p_OLDinNEW = R_GtoNEW * (p_OLDinG - p_NEWinG);
  new_feat.p_FinA = R_OLDtoNEW * landmark->get_xyz(false) + p_OLDinNEW;

  //==========================================================================
  //==========================================================================

  // OLD: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoIOLD_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot_fej();
  Eigen::Matrix<double, 3, 3> R_GtoOLD_fej = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot() * R_GtoIOLD_fej;
  Eigen::Matrix<double, 3, 1> p_OLDinG_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos_fej() -
                                             R_GtoOLD_fej.transpose() * state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

  // NEW: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoINEW_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot_fej();
  Eigen::Matrix<double, 3, 3> R_GtoNEW_fej = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot() * R_GtoINEW_fej;
  Eigen::Matrix<double, 3, 1> p_NEWinG_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos_fej() -
                                             R_GtoNEW_fej.transpose() * state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

  // Calculate transform between the old anchor and new one
  Eigen::Matrix<double, 3, 3> R_OLDtoNEW_fej = R_GtoNEW_fej * R_GtoOLD_fej.transpose();
  Eigen::Matrix<double, 3, 1> p_OLDinNEW_fej = R_GtoNEW_fej * (p_OLDinG_fej - p_NEWinG_fej);
  new_feat.p_FinA_fej = R_OLDtoNEW_fej * landmark->get_xyz(true) + p_OLDinNEW_fej;

  // Get Jacobians of p_FinG wrt new representation
  // 计算在新锚点表示下的雅可比矩阵等
  Eigen::MatrixXd H_f_new;
  std::vector<Eigen::MatrixXd> H_x_new;
  std::vector<std::shared_ptr<Type>> x_order_new;
  UpdaterHelper::get_feature_jacobian_representation(state, new_feat, H_f_new, H_x_new, x_order_new);

  //==========================================================================
  //==========================================================================

  // anchor change 本质上是一个 state reparameterization
  /* 因为新旧锚点表示下的路标是同一个物理点，只是参数化不同
  所以有 G_Pf_old = h_old(λ_old,x_old) = G_Pf_new = h_new(λ_new,x_new)
  对误差线性化
  δG_Pf_old ≈ H_f_old * δλ_old + H_x_old * δx_old
  δG_Pf_new ≈ H_f_new * δλ_new + H_x_new * δx_new
  两者相等，所以有 
  H_f_new * δλ_new + H_x_new * δx_new = H_f_old * δλ_old + H_x_old * δx_old
  于是 δλ_new = H_f_new^{-1} * (H_f_old * δλ_old + H_x_old * δx_old - H_x_new * δx_new)
  这就是 anchor change 的误差传播公式，H_f_new^{-1} 是新锚点表示下的路标误差对全局坐标误差的雅可比矩阵。
  然后用 δλ_new 做一次协方差传播，得到新的协方差矩阵。
  */
  // New phi order is just the landmark
  std::vector<std::shared_ptr<Type>> phi_order_NEW;
  phi_order_NEW.push_back(landmark);

  // Loop through all our orders and append them
  std::vector<std::shared_ptr<Type>> phi_order_OLD;
  int current_it = 0;
  std::map<std::shared_ptr<Type>, int> Phi_id_map;
  for (const auto &var : x_order_old) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  for (const auto &var : x_order_new) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  Phi_id_map.insert({landmark, current_it});
  phi_order_OLD.push_back(landmark);
  current_it += landmark->size();

  // Anchor change Jacobian
  int phisize = (new_feat.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(phisize, current_it);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(phisize, phisize);

  // Inverse of our new representation
  // pf_new_error = Hfnew^{-1}*(Hfold*pf_olderror+Hxold*x_olderror-Hxnew*x_newerror)
  Eigen::MatrixXd H_f_new_inv;
  if (phisize == 1) {
    H_f_new_inv = 1.0 / H_f_new.squaredNorm() * H_f_new.transpose();
  } else {
    H_f_new_inv = H_f_new.colPivHouseholderQr().solve(Eigen::Matrix<double, 3, 3>::Identity());
  }

  // Place Jacobians for old anchor
  //  δf_new = Phi * [δx_old; δx_new; δf_old]
  // Phi = H_f_new^{-1} * [H_f_old, H_x_old, -H_x_new]
  for (size_t i = 0; i < H_x_old.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_old[i]), phisize, x_order_old[i]->size()).noalias() += H_f_new_inv * H_x_old[i];
  }

  // Place Jacobians for old feat
  Phi.block(0, Phi_id_map.at(landmark), phisize, phisize) = H_f_new_inv * H_f_old;

  // Place Jacobians for new anchor
  for (size_t i = 0; i < H_x_new.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_new[i]), phisize, x_order_new[i]->size()).noalias() -= H_f_new_inv * H_x_new[i];
  }

  // Perform covariance propagation
  // 因为这里anchor change 不是一个随机动态过程，它只是deterministic coordinate reparameterization
  // 没有新增process noise，所以 Q = 0
  StateHelper::EKFPropagation(state, phi_order_NEW, phi_order_OLD, Phi, Q); // Q = 0

  // Set state from new feature
  landmark->_featid = new_feat.featid;
  landmark->_feat_representation = new_feat.feat_representation;
  landmark->_anchor_cam_id = new_feat.anchor_cam_id;
  landmark->_anchor_clone_timestamp = new_feat.anchor_clone_timestamp;
  landmark->set_from_xyz(new_feat.p_FinA, false);
  landmark->set_from_xyz(new_feat.p_FinA_fej, true);
  landmark->has_had_anchor_change = true;
}
