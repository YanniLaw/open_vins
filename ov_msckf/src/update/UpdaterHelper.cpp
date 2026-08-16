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

#include "UpdaterHelper.h"

#include "state/State.h"

#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

/**
 * @brief 计算全局特征位置对特征表示参数的导数(以及必要时对锚点状态的导数)
 * 因为这部分只与特征表示和锚点有关、与具体观测无关，所以在 get_feature_jacobian_full 中每个特征只调用一次，
 * 把结果（dpfg_dlambda、dpfg_dx、dpfg_dx_order）提到观测循环外面复用。
 * @param state 
 * @param feature 
 * @param H_f 
 * @param H_x 
 * @param x_order 
 */
void UpdaterHelper::get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<Type>> &x_order) {

  // Global XYZ representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
    H_f.resize(3, 3);
    H_f.setIdentity();
    return;
  }

  // Global inverse depth representation 全局球坐标逆深度
  // 参数 λ=[θ,ϕ,ρ]^T (方位角、极角、逆深度)，表示函数为(球坐标): 
  //                 | cosθ sinϕ |
  //  G_Pf = 1 / ρ * | sinθ sinϕ |
  //                 |    cosϕ   |
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH) {

    // Get the feature linearization point
    Eigen::Matrix<double, 3, 1> p_FinG = (state->_options.do_fej) ? feature.p_FinG_fej : feature.p_FinG;

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double g_rho = 1 / p_FinG.norm();               // 逆深度ρ
    double g_phi = std::acos(g_rho * p_FinG(2));    // 极角ϕ(从z轴)
    // double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
    double g_theta = std::atan2(p_FinG(1), p_FinG(0));  // 方位角θ
    Eigen::Matrix<double, 3, 1> p_invFinG;
    p_invFinG(0) = g_theta;
    p_invFinG(1) = g_phi;
    p_invFinG(2) = g_rho;

    // Get inverse depth bearings
    double sin_th = std::sin(p_invFinG(0, 0));
    double cos_th = std::cos(p_invFinG(0, 0));
    double sin_phi = std::sin(p_invFinG(1, 0));
    double cos_phi = std::cos(p_invFinG(1, 0));
    double rho = p_invFinG(2, 0);

    // Construct the Jacobian [∂y/∂x]ij = ∂yi / ∂xj 
    H_f.resize(3, 3);
    H_f << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    return;
  }

  //======================================================================
  //======================================================================
  //======================================================================

  // Assert that we have an anchor pose for this feature
  assert(feature.anchor_cam_id != -1); // 确保确实是有锚点相机的

  // Anchor pose orientation and position, and camera calibration for our anchor camera
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
  Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
  Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
  Eigen::Vector3d p_FinA = feature.p_FinA;

  // If I am doing FEJ, I should FEJ the anchor states (should we fej calibration???)
  // Also get the FEJ position of the feature if we are
  if (state->_options.do_fej) {
    // "Best" feature in the global frame
    Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
    // Transform the best into our anchor frame using FEJ
    R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
    p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
    p_FinA = (R_GtoI.transpose() * R_ItoC.transpose()).transpose() * (p_FinG_best - p_IinG) + p_IinC;
  }
  Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_ItoC.transpose();

  // Jacobian for our anchor pose
  Eigen::Matrix<double, 3, 6> H_anc;
  H_anc.block(0, 0, 3, 3).noalias() = -R_GtoI.transpose() * skew_x(R_ItoC.transpose() * (p_FinA - p_IinC));
  H_anc.block(0, 3, 3, 3).setIdentity();

  // Add anchor Jacobians to our return vector
  x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
  H_x.push_back(H_anc);

  // Get calibration Jacobians (for anchor clone)
  if (state->_options.do_calib_camera_pose) {
    Eigen::Matrix<double, 3, 6> H_calib;
    H_calib.block(0, 0, 3, 3).noalias() = -R_CtoG * skew_x(p_FinA - p_IinC);
    H_calib.block(0, 3, 3, 3) = -R_CtoG;
    x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
    H_x.push_back(H_calib);
  }

  // If we are doing anchored XYZ feature
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    H_f = R_CtoG;
    return;
  }

  // If we are doing full inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double a_rho = 1 / p_FinA.norm();
    double a_phi = std::acos(a_rho * p_FinA(2));
    double a_theta = std::atan2(p_FinA(1), p_FinA(0));
    Eigen::Matrix<double, 3, 1> p_invFinA;
    p_invFinA(0) = a_theta;
    p_invFinA(1) = a_phi;
    p_invFinA(2) = a_rho;

    // Using anchored inverse depth
    double sin_th = std::sin(p_invFinA(0, 0));
    double cos_th = std::cos(p_invFinA(0, 0));
    double sin_phi = std::sin(p_invFinA(1, 0));
    double cos_phi = std::cos(p_invFinA(1, 0));
    double rho = p_invFinA(2, 0);
    // assert(p_invFinA(2,0)>=0.0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  // If we are doing the MSCKF version of inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    Eigen::Matrix<double, 3, 1> p_invFinA_MSCKF;
    p_invFinA_MSCKF(0) = p_FinA(0) / p_FinA(2);
    p_invFinA_MSCKF(1) = p_FinA(1) / p_FinA(2);
    p_invFinA_MSCKF(2) = 1 / p_FinA(2);

    // Using the MSCKF version of inverse depth
    double alpha = p_invFinA_MSCKF(0, 0);
    double beta = p_invFinA_MSCKF(1, 0);
    double rho = p_invFinA_MSCKF(2, 0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << (1.0 / rho), 0.0, -(1.0 / (rho * rho)) * alpha, 0.0, (1.0 / rho), -(1.0 / (rho * rho)) * beta, 0.0, 0.0,
        -(1.0 / (rho * rho));
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  /// CASE: Estimate single depth of the feature using the initial bearing
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double rho = 1.0 / p_FinA(2);
    Eigen::Vector3d bearing = rho * p_FinA;

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Vector3d d_pfinA_drho;
    d_pfinA_drho << -(1.0 / (rho * rho)) * bearing;
    H_f = R_CtoG * d_pfinA_drho;
    return;
  }

  // Failure, invalid representation that is not programmed
  assert(false);
}

/**
 * @brief Compute the full Jacobian for a given feature with respect to the feature itself and the involved states.
 * 为单个特征构造完整线性化测量模型。把一个特征在所有相机、所有时刻的观测堆叠起来，
 * 输出残差 r、特征雅可比Hf、状态雅可比Hx以及涉及的状态列表。目标是把下面这个线性化方程建立出来：
 *  r = H_f * delta_f + H_x * delta_x
 * 对于某一次观测，测量值是去畸变后的像素坐标(u,v)，预测值来自投影链:
 * pIi = R_GtoIi * (p_FinG - p_IinG)  // 全局系下的特征坐标 → 当前相机系下的特征坐标
 * pCi = R_ItoCi * pIi + p_IinCi // 当前相机系下的特征坐标 → 当前相机像素坐标
 * z = distort([x/z]) , [x,y,z]^T = pCi // 当前相机像素坐标 → 去畸变后的归一化像素坐标(u,v)
 *             [y/z]
 * 
 * 残差 r = zm - zo,线性化后对状态和特征分别求导，就得到H_x和H_f。
 * 注意，H_x是一个大矩阵，列数是所有涉及状态的总维数，行数是该特征的总观测次数*2。
 * @param state 滤波器状态
 * @param feature 特征（含所有观测、三角化均值）
 * @param H_f 输出：观测对特征误差的雅可比 2n x 3(或2n x 1)  视表达方式而定
 * @param H_x 输出：观测对状态误差的雅可比 2n x 状态维数
 * @param res 输出：测量残差 2n x1 
 * @param x_order 输出：H_x 各列对应的状态列表
 */
void UpdaterHelper::get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {

  // Total number of measurements for this feature
  // 统计该特征的总观测次数（所有相机、所有时刻）
  int total_meas = 0;
  for (auto const &pair : feature.timestamps) {
    // 每个观测贡献 2 行（u、v），所以残差长度是 2 * total_meas
    total_meas += (int)pair.second.size();
  }

  // Compute the size of the states involved with this feature
  int total_hx = 0; // 所有涉及状态的总维数
  std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // 状态对象 → 列偏移(在H_x中的起始列索引)
  // 遍历观测到该特征的所有相机，统计涉及的状态维度并建立映射
  for (auto const &pair : feature.timestamps) {

    // Our extrinsics and intrinsics
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);

    // If doing calibration extrinsics 相机外参 6维
    if (state->_options.do_calib_camera_pose) {
      map_hx.insert({calibration, total_hx});
      x_order.push_back(calibration);
      total_hx += calibration->size();
    }

    // If doing calibration intrinsics 相机内参以及畸变
    if (state->_options.do_calib_camera_intrinsics) {
      map_hx.insert({distortion, total_hx});
      x_order.push_back(distortion);
      total_hx += distortion->size();
    }

    // Loop through all measurements for this specific camera
    // 遍历该相机的所有观测，添加该相机每次观测对应的 IMU 克隆位姿（6维）
    for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

      // Add this clone if it is not added already
      std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
      // 去重：因为有可能多个相机的观测对应同一个IMU克隆，确保每个克隆只被添加一次
      if (map_hx.find(clone_Ci) == map_hx.end()) {
        map_hx.insert({clone_Ci, total_hx});
        x_order.push_back(clone_Ci);
        total_hx += clone_Ci->size();
      }
    }
  }

  // If we are using an anchored representation, make sure that the anchor is also added
  // 补充锚点状态: 如果特征是锚点表示，还要额外登记锚点克隆位姿及其外参（保证锚点状态也在雅可比里被考虑到）
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    // 锚点克隆位姿 + 锚点相机外参（若标定)
    // Assert we have a clone
    assert(feature.anchor_cam_id != -1);

    // Add this anchor if it is not added already
    std::shared_ptr<PoseJPL> clone_Ai = state->_clones_IMU.at(feature.anchor_clone_timestamp);
    if (map_hx.find(clone_Ai) == map_hx.end()) {
      map_hx.insert({clone_Ai, total_hx});
      x_order.push_back(clone_Ai);
      total_hx += clone_Ai->size();
    }

    // Also add its calibration if we are doing calibration
    if (state->_options.do_calib_camera_pose) {
      // Add this anchor if it is not added already
      std::shared_ptr<PoseJPL> clone_calib = state->_calib_IMUtoCAM.at(feature.anchor_cam_id);
      if (map_hx.find(clone_calib) == map_hx.end()) {
        map_hx.insert({clone_calib, total_hx});
        x_order.push_back(clone_calib);
        total_hx += clone_calib->size();
      }
    }
  }

  //=========================================================================
  //=========================================================================

  // Calculate the position of this feature in the global frame
  // If anchored, then we need to calculate the position of the feature in the global
  // 把特征位置统一到全局系
  // 非锚点表示直接使用全局坐标；锚点表示则通过锚点位姿把 p_FinA 变换到全局系，方便后面所有观测统一投影。
  // FEJ 时使用第一估计值。
  Eigen::Vector3d p_FinG = feature.p_FinG;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    // Assert that we have an anchor pose for this feature
    assert(feature.anchor_cam_id != -1);
    // Get calibration for our anchor camera
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    // Anchor pose orientation and position
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    // Feature in the global frame
    p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
  }

  // Calculate the position of this feature in the global frame FEJ
  // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
  Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    p_FinG_fej = p_FinG;
  }

  //=========================================================================
  //=========================================================================

  // Allocate our residual and Jacobians
  // 分配残差向量和雅可比矩阵的内存
  int c = 0; // 其实就是处理的特征index
  // 根据特征的表示类型决定雅可比矩阵的列数，如果是锚点逆深度表示，则列数为1，否则为3
  int jacobsize = (feature.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  res = Eigen::VectorXd::Zero(2 * total_meas);  // 残差向量，每个观测有两个分量 (u, v)，所以长度是 2 * total_meas
  H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);   // 观测对特征的雅可比
  H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);    // 观测对状态的雅可比

  // Derivative of p_FinG in respect to feature representation.
  // This only needs to be computed once and thus we pull it out of the loop
  // 这部分只与特征表示有关，与具体观测无关，所以只算一次，在循环外提取
  Eigen::MatrixXd dpfg_dlambda; // 全局特征位置P_FinG对特征表示参数λ的导数
  std::vector<Eigen::MatrixXd> dpfg_dx; // 全局特征位置P_FinG对锚点状态（锚点位姿、锚点外参）的导数，及其对应的状态列表
  std::vector<std::shared_ptr<Type>> dpfg_dx_order;
  UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

  // Assert that all the ones in our order are already in our local jacobian mapping
#ifndef NDEBUG
  for (auto &type : dpfg_dx_order) {
    assert(map_hx.find(type) != map_hx.end());
  }
#endif

  // Loop through each camera for this feature
  // 遍历该特征在所有相机下的观测
  for (auto const &pair : feature.timestamps) {
    // pair.first 是相机ID，pair.second 是该
    // Our calibration between the IMU and CAMi frames
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);      // 相机内参及畸变参数
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first); // 相机外参
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    // Loop through all measurements for this specific camera
    // 遍历该特征在该相机下的所有时间观测
    for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

      //=========================================================================
      //=========================================================================

      // Get current IMU clone state 获取该次观测的IMU位姿
      std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
      Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
      Eigen::Vector3d p_IiinG = clone_Ii->pos();

      // Get current feature in the IMU 计算该特征点在该次观测下的imu坐标系坐标  全局->imu
      Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

      // Project the current feature into the current frame of reference
      Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC; // 该特征在相机坐标系下的坐标 imu->相机
      Eigen::Vector2d uv_norm;
      uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2); // 归一化坐标

      // Distort the normalized coordinates (radtan or fisheye)
      Eigen::Vector2d uv_dist;
      uv_dist = state->_cam_intrinsics_cameras.at(pair.first)->distort_d(uv_norm); // 还原畸变图像坐标

      // Our residual 残差
      Eigen::Vector2d uv_m;
      uv_m << (double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1);
      res.block(2 * c, 0, 2, 1) = uv_m - uv_dist; // 对应位置的残差赋值，每个测量特征贡献 2 x 1 残差向量

      //=========================================================================
      //=========================================================================

      // If we are doing first estimate Jacobians, then overwrite with the first estimates
      // 若开启 FEJ，所有量换用第一估计值（保证雅可比在固定点线性化，保证可观性）
      if (state->_options.do_fej) {
        R_GtoIi = clone_Ii->Rot_fej();
        p_IiinG = clone_Ii->pos_fej();
        // R_ItoC = calibration->Rot_fej();
        // p_IinC = calibration->pos_fej();
        p_FinIi = R_GtoIi * (p_FinG_fej - p_IiinG);
        p_FinCi = R_ItoC * p_FinIi + p_IinC;
        // uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);
        // cam_d = state->get_intrinsics_CAM(pair.first)->fej();
      }

      // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
      // 畸变对归一化坐标/内参的导数
      Eigen::MatrixXd dz_dzn, dz_dzeta;
      state->_cam_intrinsics_cameras.at(pair.first)->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

      // Normalized coordinates in respect to projection function
      // 归一化坐标对相机系3D点（投影函数）的导数
      Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
      dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

      // Derivative of p_FinCi in respect to p_FinIi
      // p_FC 对 p_FG 的导数（旋转链）
      Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;

      // Derivative of p_FinCi in respect to camera clone state
      // p_FC 对克隆6自由度位姿的导数（旋转3+平移3）
      Eigen::MatrixXd dpfc_dclone = Eigen::MatrixXd::Zero(3, 6);
      dpfc_dclone.block(0, 0, 3, 3).noalias() = R_ItoC * skew_x(p_FinIi);
      dpfc_dclone.block(0, 3, 3, 3) = -dpfc_dpfg;

      //=========================================================================
      //=========================================================================

      // Precompute some matrices
      // 链式法则合成最终雅可比
      Eigen::MatrixXd dz_dpfc = dz_dzn * dzn_dpfc;    // 观测对 p_FC
      Eigen::MatrixXd dz_dpfg = dz_dpfc * dpfc_dpfg;  // 观测对 p_FG

      // CHAINRULE: get the total feature Jacobian
      // ① 特征雅可比
      H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg * dpfg_dlambda;

      // CHAINRULE: get state clone Jacobian
      // ② 克隆位姿雅可比
      H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dclone;

      // CHAINRULE: loop through all extra states and add their
      // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
      // ③ 额外状态（锚点位姿/锚点外参）雅可比：累加
      for (size_t i = 0; i < dpfg_dx_order.size(); i++) {
        H_x.block(2 * c, map_hx[dpfg_dx_order.at(i)], 2, dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg * dpfg_dx.at(i);
      }

      //=========================================================================
      //=========================================================================

      // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
      // 相机外参与内参雅可比
      if (state->_options.do_calib_camera_pose) {

        // Calculate the Jacobian
        Eigen::MatrixXd dpfc_dcalib = Eigen::MatrixXd::Zero(3, 6);
        dpfc_dcalib.block(0, 0, 3, 3) = skew_x(p_FinCi - p_IinC); // 对外参6自由度
        dpfc_dcalib.block(0, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();

        // Chainrule it and add it to the big jacobian
        // 这里用的是 +=，因为同一相机的外参/内参可能与 的锚点外参是同一个对象，需要累加而不是覆盖
        H_x.block(2 * c, map_hx[calibration], 2, calibration->size()).noalias() += dz_dpfc * dpfc_dcalib;
      }

      // Derivative of measurement in respect to distortion parameters
      if (state->_options.do_calib_camera_intrinsics) {
        H_x.block(2 * c, map_hx[distortion], 2, distortion->size()) = dz_dzeta; // 畸变参数
      }

      // Move the Jacobian and residual index forward
      c++;
    }
  }
}

/**
 * @brief 零空间投影 把特征点误差 δf 从量测方程里消掉，只保留对状态误差 δx 的约束
 * 原始线性化量测模型可以写成 r = Hx δx + Hf δf + n
 * @param H_f 
 * @param H_x 
 * @param res 
 */
void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
}

void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) {
    for (int m = (int)H_x.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols());
  res.conservativeResize(r, res.cols());
}
