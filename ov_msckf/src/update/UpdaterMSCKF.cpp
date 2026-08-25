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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared 像素噪声方差
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer 三角化器
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // 预计算卡方 95% 分位点表（自由度 1~499）
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

/**
 * @brief Will compute the system for our sparse features and update the filter.
 * 这是 MSCKF 更新的主执行函数：输入一批特征（feature_vec），输出对状态的 EKF 校正。
 * 整体是一条流水线：清理 → 准备相机位姿 → 三角化 → 构造线性系统（含零空间投影+卡方剔除）→ 测量压缩 → EKF 更新
 * @param state State of the filter
 * @param feature_vec Features that can be used for update
 */
void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
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
    (*it0)->clean_old_measurements(clonetimes); // 只保留该特征点在克隆时刻上的测量

    // Count how many measurements 统计该特征的总测量数
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) { // 少于 2 个 → 无法三角化 → 剔除
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps  构建相机位姿克隆图
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) { // 遍历每个相机

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose 对每个相机 × 每个克隆时刻，计算相机在世界系下的位姿
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map key 为克隆时刻，value 为相机位姿
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map key 为相机索引，value 为该相机在各克隆时刻的位姿
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  // 用多视角观测三角化出特征 3D 位置，再用高斯牛顿精化。失败（退化、射线平行）→ 剔除，避免带病特征进更新。
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    // 两种三角化方式（1D 逆深度 or 3D）
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    // 高斯牛顿精化（可选，默认开）
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // Remove the feature if not a success 任一失败 → 剔除
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();
  // 构造大线性系统之前的"预分配 + 记账结构初始化"。
  // 目的是：为接下来要堆叠所有特征方程的大矩阵 Hx_big/res_big 预先申请好足够的存储空间，并初始化好"状态列块"的登记表。
  // Calculate the max possible measurement size
  // 计算所有特征观测的最大可能测量维度(行数)，用于预分配残差向量的大小。
  // 得到所有特征原始测量的总行数（在零空间投影、卡方剔除之前），作为 res_big 行数的上界
  // 注意为什么是"最大可能"：此时 feature_vec 只是通过了三角化，第 4 步里还会有特征被卡方检验剔除，且零空间投影会让每个特征的行数从 
  // 2m 减到 2m−3。所以这个值只是上限，真正用多少行要等第 4 步跑完才知道（后面用 conservativeResize 收缩）。
  size_t max_meas_size = 0; // 最大可能测量维数（行数）
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) { // 遍历每个特征 × 每个相机，累加该相机对该特征的观测次数
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size(); // 每个观测是归一化坐标(u,v)两个维度
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  // 如果特征用逆深度表示，max_covariance_size() 里的相关处理会把路标按 1 自由度计
  // 得到的 max_hx_size 是 Hx_big 列数的上界：实际参与本次更新的状态（只有被特征观测到的克隆/外参/内参）通常小于这个值
  size_t max_hx_size = state->max_covariance_size(); // 最大可能状态维数（列数）
  for (auto &landmark : state->_features_SLAM) {
    // SLAM 路标是单独用 UpdaterSLAM 更新的，不会出现在本函数 Hx_big 的列里——所以要把它们的维度从"最大状态大小"里扣掉，避免浪费列空间；
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  // 预分配大矩阵 + 记账结构
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); // 该状态在大矩阵里的起始列（跨特征共享）
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping; // 状态类型 → 列偏移
  std::vector<std::shared_ptr<Type>> Hx_order_big; // 状态类型的有序列表
  size_t ct_jacob = 0; // 已登记的状态总列数
  size_t ct_meas = 0;  // 已填入的测量行数(残差累计长度)

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
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
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
    Eigen::MatrixXd H_f; // 残差对特征点的雅可比矩阵 2m x 3(或2m x 1)
    Eigen::MatrixXd H_x; // 残差对状态的雅可比矩阵   2m x 状态维数
    Eigen::VectorXd res; // 残差向量(重投影)        2m x 1 
    std::vector<std::shared_ptr<Type>> Hx_order; // 记录当前Hx矩阵每一列块对应哪个状态对象(标定状态，IMU克隆状态)

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Nullspace project 零空间投影
    // 经过零空间投影后，H_f 被消掉，H_x 和 res 被变换为 Hx' = Q^T Hx, res' = Q^T res，其中 Q 是由所有 Givens 旋转组成的正交矩阵。
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check 卡方距离检验
    // 每个特征经过三角化、构造雅可比、零空间投影后，会得到一个残差向量r。这一步要问一个问题："这个残差，大到离谱了吗？" 
    // 如果大到用噪声模型都解释不了（马氏距离超限），就认为它是外点（动态物体、误匹配、三角化炸了），直接删掉，不参与状态更新。
    // 为什么需要它？
    // MSCKF 每次会喂进来几十上百个特征。其中必然混着:
    // 1. 动态物体上的特征点（比如车牌、行人、车灯）
    // 2. 误匹配的特征点（比如误匹配到了背景的纹理） 
    // 3. 在特征边缘、深度退化处三角化出的病态点（比如观测点共线、视差太小）
    // 这些都会导致三角化出来的 3D 点会在不同帧之间跳来跳去，导致残差加大；
    // 如果把这些垃圾特征也丢进 EKF 更新，会污染状态估计。所以需要在更新前，
    // 用统计学方法把"残差异常大"的特征筛掉——这就是 RANSAC 之外另一种经典的**门控（gating）**手段。
    /* 马氏距离 = "以噪声标准差为单位"的残差
    普通欧氏距离 ||r||_2 = sqrt(r^T r) 只考虑了残差的大小，没有考虑噪声协方差 S 的影响。
    由于残差不同分量的不确定度不一样（有的方向噪声大、有的小），所以要用马氏距离，它把残差"除以不确定度"再求模
    马氏距离 d = r^T*S^-1*r 考虑了噪声协方差 S 的影响，S 是残差 r 的协方差
    直观理解：它量的是"这个残差相当于多少个标准差的异常"。r的每个分量被S 归一化后，d2是个无量纲的"离谱程度"
    马氏距离的平方服从卡方分布
    如果这个特征是"好"的（残差确实服从 N(0,S)），那么 chi2 = r^T * S^-1 * r 服从自由度为残差维度 res.rows() 的卡方分布。
    所以对每个特征我们能算出"正常范围"：
    - 95% 分位点: 好的特征有 95% 的概率落在这个范围内，5% 的概率落在外面
    - 如果 chi2 > 95%分位点，那只有 5% 的可能它是好的 → 更合理的解释是它是外点 → 剔除
    note： 为什么自由度是残差维度res.rows()？因为残差是高斯噪声下的随机变量，残差维度就是它的自由度。
    自由度 = 残差里独立分量的个数
    零空间投影前:一个特征有m个观测点，每个观测点有(u,v)两个分量 → 残差维度 = 2m
    零空间投影后:每个特征的残差维度从 2m 减到 2m−3（因为投影掉了3个自由度） → 自由度 = 2m−3
    又由于res.rows()已经是投影后的行数，所以天然就是自由度，代码直接用res.rows()查表/算分位点。
    */
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order); // 获取当前特征观测涉及的状态(IMU克隆，标定状态)的边缘协方差矩阵
    // 状态不确定性投影到测量空间 —— 把状态协方差"穿过"测量雅可比Hx，变成残差空间里的协方差
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    // 构造投影后量测的新息协方差 S = H_x * P_marg * H_x^T + σ_pix^2 * I，其中 R = σ_pix^2 * I 是像素噪声协方差
    // 注意： 零空间投影用的是正交矩阵Q2(Givens 旋转)，它不改变白噪声的性质。即各向同性白噪声经过正交变换后仍是各向同性白噪声、方差不变。
    // 所以噪声仍是 isotropic的，投影后 R 仍然是 σ_pix^2 * I，直接加在 S 的对角线上是正确的。
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    // 计算马氏距离平方，数学上等价于卡方统计量 chi2 = r^T * S^-1 * r，但不直接求逆S^-1(数值上慢且不稳定)，
    // 这里用 Cholesky 分解：
    // 1. S.llt()把S分解为下三角矩阵L和其转置L^T，满足 S = L * L^T
    // 2. S.llt().solve(res)求解线性方程 S * x = r，得到 x = S^-1 * r
    // 3. res.dot(S.llt().solve(res)) 就是 r^T * S^-1 * r，
    // 也就是马氏距离平方，即所需要的卡方统计量 chi2
    // 解释：马氏距离是统计学中衡量一个点与分布中心的距离，考虑了协方差。对于高斯分布，马氏距离平方服从卡方分布。
    // 这里的残差 r 是高斯噪声下的随机变量，S 是它的协方差矩阵，所以 r^T S^-1 r  
    // 服从卡方分布，卡方检验就是用这个统计量来判断观测是否异常。
    // 具体公式参考 https://docs.openvins.com/update-feat.html#chi2-check
    double chi2 = res.dot(S.llt().solve(res));
    // 此时残差维度为2m-3，S是2m-3 x 2m-3的矩阵，chi2是标量。 自由度就是res.rows()

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
    if (chi2 > _options.chi2_multipler * chi2_check) { // 该特征为外点（动态物体、误三角化、观测异常），直接删除并跳过堆叠
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0; // 当前特征 H_x 内部的"列游标"(当前状态快在本特征内的起始列索引)
    for (const auto &var : Hx_order) { // 按顺序遍历该特征的每个状态块

      // Ensure that this variable is in our Jacobian
      // 全局去重：这个状态第一次出现才在大矩阵里分配列
      // 因为同一个状态（比如某个 IMU 克隆）会被很多特征同时观测到。
      // 第一个遇到它的特征负责在 Hx_big 里分配列；后面所有特征都通过 Hx_mapping 找到同一列往里填。这样最终 
      // Hx_big 的每一列都只对应一个状态，且 Hx_order_big 记录了这个对应关系，供最后的 EKFUpdate 使用
      // note: Hx_big 的行就是所有好特征的残差堆叠（每一行对应某特征某观测的一个 u 或 v 分量），列是所有涉及状态的总维数
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob}); // 状态 → 全局起始列
        Hx_order_big.push_back(var);        // 记入全局顺序表
        ct_jacob += var->size();            // 推进全局列计数
      }

      // Append to our large Jacobian
      // 把这块从"单特征 H_x"拷进"全局 Hx_big"
      // Hx_big是列共享，行堆叠结构: 同一个状态（列）被多个特征观测，就在不同行各写一段自己的方程块。
      // ct_meas 保证每个特征的行区间互不重叠，因此后拷贝的特征不会覆盖前面的，
      // 反而正是靠这种堆叠把多个特征的观测约束融合进同一个 EKF 更新。
      // 真正的"同一状态写同一块"只在单特征内部出现，而那已经在 get_feature_jacobian_full 里用 += 合并好了
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size(); // 推进单特征列游标
    }

    // Append our residual and move forward
    // 残差也一样拷贝进去
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();  // 推进全局行计数
    it2++;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  // 用完即弃
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  // 收缩矩阵到实际大小 + 保护判断
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  // 开头预分配的max_meas_size/max_hx_size 是上界（卡方剔除 + 零空间投影会让行数变少、去重会让列数变少）。
  // 现在用 conservativeResize 把矩阵裁剪到真实尺寸，省内存也让后续计算维度正确。
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 5. Perform measurement compression 测量压缩
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Our noise is isotropic, so make it here after our compression
  // 压缩用的是正交变换，各向同性噪声保持不变，所以 R 仍是σ_pix^2*I，直接按压缩后的行数构造即可
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
