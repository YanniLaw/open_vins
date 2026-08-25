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
 * 参考 https://docs.openvins.com/update-feat.html#feat-rep
 * 因为这部分只与特征表示和锚点有关、与具体观测无关，所以在 get_feature_jacobian_full 中每个特征只调用一次，
 * 把结果（dpfg_dlambda、dpfg_dx、dpfg_dx_order）提到观测循环外面复用。
 * 随后在观测循环里用链式法则合成: 
 * ∂z/∂x = ∂z/∂C_Pf * ∂C_Pf/∂G_Pf * ∂G_Pf/∂x
 * 这里的 ∂G_Pf/∂x 就是本函数计算的内容
 * 注意: H_x 是 vector，不是矩阵：因为每个锚点相关的状态（克隆、外参）尺寸不同，用 vector<MatrixXd> 逐块存放，
 * 并用 x_order 一一对应记录是哪份状态。后续在 get_feature_jacobian_full 中通过 map_hx 把块放进大矩阵 Hx 的对应列
 * 输出被消费的方式: 
 * 特征部分: Hf^观测 = ∂z/∂G_Pf*Hf^本函数(就是循环里的dz_dpfg * dpfg_dlambda) λ是特征表示相关的状态
 * 锚点状态部分:   ∂z/ ∂x_anchor = ∂z/∂G_Pf*∂G_Pf/∂x_anchor(就是循环里的dz_dpfg * dpfg_dx.at(i)) x是锚点相关的状态
 * @param state 
 * @param feature 
 * @param H_f 输出: 全局特征位置对特征表示的雅可比矩阵
 * @param H_x 输出: 全局特征位置对锚点状态或者标定状态的雅可比矩阵(如果不是锚点表示则不计算，不开启标定也不计算)
 * @param x_order 
 */
void UpdaterHelper::get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<Type>> &x_order) {

  // Global XYZ representation
  // G_Pf = f(λ) = [G_x, G_y, G_z]^T , λ = [G_x, G_y, G_z]^T, ∂f(λ)/∂λ = I_3x3
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
    H_f.resize(3, 3);
    H_f.setIdentity();
    return;
  }

  // Global inverse depth representation 全局球坐标逆深度
  // 参数 λ = [θ,ϕ,ρ]^T (方位角、极角、逆深度)，表示函数为(球坐标): 
  //                 | cosθ sinϕ |
  //  G_Pf = 1 / ρ * | sinθ sinϕ | = f(λ)
  //                 |    cosϕ   |
  // ∂f(λ)/∂λ = [...]3x3
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
  // 下面都是用锚点系表示的一些计算!!!!!
  // 所以除了对特征表示参数求导外，还要额外对锚点状态求导
  // 对锚点状态求导(其实就是对锚点位姿求导)，得到雅可比矩阵 H_x

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
  // FEJ 处理（关键细节）：目的是把锚点相关雅可比固定在 FEJ 线性化点上（保持可观性性质）。
  // 注意它只对锚点状态做 FEJ，外参 R_ItoC/p_IinC 刻意不做
  if (state->_options.do_fej) {
    // "Best" feature in the global frame
    // 用"当前"锚点状态把特征转到全局系，得到"best"全局位置
    Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
    // Transform the best into our anchor frame using FEJ
    // 换成 FEJ（第一估计）锚点状态，再转回锚点相机系
    R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
    p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
    p_FinA = (R_GtoI.transpose() * R_ItoC.transpose()).transpose() * (p_FinG_best - p_IinG) + p_IinC;
  }
  Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_ItoC.transpose();

  // Jacobian for our anchor pose，锚点姿态雅可比
  // 见文档 https://docs.openvins.com/update-feat.html#feat-rep-anchor-xyz
  // G_Pf = f(λ, R_GtoIa，P_IainG, R_ItoC, P_IinC) 
  //      = R_GtoIa^T * R_ItoC^T * (λ - P_IinC) + P_IainG
  // 其中 λ = Ca_Pf = [Ca_x, Ca_y, Ca_z]^T, 是锚点系下的特征位置
  // 所以，∂f(*)/∂λ = R_GtoIa^T * R_ItoC^T
  // 由于锚点位姿涉及到这种特征表示，所以需要对锚点位姿求导，得到雅可比矩阵 H_anc
  // 对锚点克隆位姿的雅可比就是对 R_GtoIa 和 P_IainG 的偏导数：
  // ∂f(*)/∂R_GtoIa = -R_GtoIa^T * skew_x(R_ItoC^T * (λ - P_IinC))
  // ∂f(*)/∂P_IainG = I_3x3
  Eigen::Matrix<double, 3, 6> H_anc;
  H_anc.block(0, 0, 3, 3).noalias() = -R_GtoI.transpose() * skew_x(R_ItoC.transpose() * (p_FinA - p_IinC));
  H_anc.block(0, 3, 3, 3).setIdentity();

  // Add anchor Jacobians to our return vector
  x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
  H_x.push_back(H_anc);

  // Get calibration Jacobians (for anchor clone) 锚点相机外参雅可比，同样见如上的文档
  // ∂f(*)/∂R_ItoC = -R_GtoIa^T * R_ItoC^T * skew_x(λ - P_IinC)
  // ∂f(*)/∂P_IinC = -R_GtoIa^T * R_ItoC^T
  if (state->_options.do_calib_camera_pose) {
    Eigen::Matrix<double, 3, 6> H_calib;
    H_calib.block(0, 0, 3, 3).noalias() = -R_CtoG * skew_x(p_FinA - p_IinC);
    H_calib.block(0, 3, 3, 3) = -R_CtoG;
    x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
    H_x.push_back(H_calib);
  }

  // If we are doing anchored XYZ feature
  // Anchored XYZ
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    // 表示参数: 锚点系XYZ  H_f = R_CtoG
    H_f = R_CtoG;
    return;
  }

  // If we are doing full inverse depth
  // Anchored Inverse Depth
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
    // 见公式 https://docs.openvins.com/update-feat.html#feat-rep-anchor-inv
    // 表示参数  [θ,ϕ,ρ]（锚点系球坐标） H_f = R_CtoG * ∂_Ca_Pf/∂[θ,ϕ,ρ]
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
    // 见公式 https://docs.openvins.com/update-feat.html#feat-rep-anchor-inv2
    // 参数 [α,β,ρ]=[x/z, y/z, 1/z]， H_f = R_CtoG * ∂_Ca_Pf/∂[α,β,ρ]
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
    // 参数 ρ = 1/z， H_f = R_CtoG * ∂_Ca_Pf/∂ρ
    Eigen::Vector3d d_pfinA_drho; // - (1/ρ^2) * bearing
    d_pfinA_drho << -(1.0 / (rho * rho)) * bearing;
    H_f = R_CtoG * d_pfinA_drho;
    return;
  }

  // Failure, invalid representation that is not programmed
  assert(false);
}

/**
 * @brief Compute the full Jacobian for a given feature with respect to the feature itself and the involved states.
 *  一般来说，MSCKF的目标是把下面这个线性化方程：
 *  r = H_f * delta_f + H_x * delta_x
 *  openvins为了构造完整的线性化观测模型，把该特征在所有相机、所有时刻的观测堆叠成一个大的线性方程:
 *  r = H_f * δλ + H_x * δx + n
 * 其中，r 残差(2nx1,每个观测贡献u,v两行)
 * H_f: 对特征参数误差δλ的雅可比矩阵(2nx3或2nx1，视表示方式而定)
 * H_x: 对状态误差的雅可比矩阵(2nxm.m为设计状态总维数)
 * 投影链(即h_d ∘ h_p ∘ h_t):
 * Ii_Pf = R_GtoIi(P_FinG - P_IiinG), Ci_Pf = R_ItoCi*Ii_Pf + P_IinCi
 * zn =  hp(Ci_Pf) = [x/z, y/z]^T  z = hd(zn,ζ)
 * 线性化后对某状态 x 的雅可比就是文档里的链式法则公式
 * ∂z    ∂hd    ∂hp  | ∂Ci_Pf   ∂Ci_Pf     ∂G_Pf |
 * —— =   ——    ——   |   ——   +   ——    *   ——   |
 * ∂x    ∂zn  ∂Ci_Pf |   ∂x     ∂G_Pf       ∂x   |
 *                                  对锚点/表示
 * 其中 ∂hd/∂zn就是dz_dzn
 * ∂hp/∂Ci_Pf就是dzn_dpfc
 * 
 * 对于某一次观测，测量值是带畸变的像素坐标(u,v)，预测值来自投影链:
 * pIi = R_GtoIi * (p_FinG - p_IinG)  // 全局系下的特征坐标 → 当前IMU系下的特征坐标
 * pCi = R_ItoCi * pIi + p_IinCi      // 当前IMU系下的特征坐标 → 当前相机系下的特征坐标
 * z = distort([x/z]) , [x,y,z]^T = pCi // 当前相机系下的特征坐标 → 去畸变后的归一化像素坐标(u,v)
 *             [y/z]
 * 
 * 残差 r = zm - zo,线性化后对状态和特征分别求导，就得到H_x和H_f。
 * 注意，H_x是一个大矩阵，列数是所有涉及状态的总维数，行数是该特征的总观测次数*2。
 * @param state 滤波器状态
 * @param feature 特征（含所有观测、三角化均值）
 * @param H_f 输出：观测对特征误差的雅可比 2m x 3(或2m x 1)  视表达方式而定
 * @param H_x 输出：观测对状态误差的雅可比 2m x 状态维数
 * @param res 输出：测量残差 2m x1 
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

    // If doing calibration intrinsics 相机内参以及畸变 8维
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
  // 这样 H_x 的列数 total_hx 就是所有相关状态维数的总和，而 map_hx 记录每个状态对象在 H_x 中的起始列索引

  //=========================================================================
  //=========================================================================

  // Calculate the position of this feature in the global frame
  // If anchored, then we need to calculate the position of the feature in the global
  // 把特征位置统一到全局系G_Pf
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
    // 通过锚点外参以及锚点克隆位姿把锚点系下的特征点坐标P_FinA变换到全局
    // 其实就是一个 A_Pf(也就是C_Pf) -> I_Pf -> G_Pf 的转换
    // p_FinG = R_GtoI^T * R_ItoC^T * (p_FinA - p_IinC) + p_IinG
    // 这样后面每个观测都能用同一个全局点统一投影
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
  // 这部分只与特征表示和锚点有关，与具体观测无关，所以只算一次，在循环外提取并复用
  Eigen::MatrixXd dpfg_dlambda; // 全局特征位置P_FinG对特征表示参数λ的导数 ∂^Gpf/∂λ
  std::vector<Eigen::MatrixXd> dpfg_dx; // 全局特征位置P_FinG对锚点状态（锚点位姿、锚点外参）的导数，∂^Gpf/∂x_anchor（多块）
  std::vector<std::shared_ptr<Type>> dpfg_dx_order; // 记录 dpfg_dx 中每块对应的状态对象，方便后续在 H_x 中放置雅可比块
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
      // https://docs.openvins.com/update-feat.html#distortion
      // 分别是公式中的∂hd(.)/∂zn,k, ∂hd(.)/∂ζ
      Eigen::MatrixXd dz_dzn, dz_dzeta;
      state->_cam_intrinsics_cameras.at(pair.first)->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

      // Normalized coordinates in respect to projection function
      // 归一化坐标对相机系3D点（投影函数）的导数 ∂zn/∂^Cpf
      Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
      dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

      // Derivative of p_FinCi in respect to p_FinIi
      // p_FC 对 p_FG 的导数（旋转链） ∂^Cpf/∂^Gpf
      Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;

      // Derivative of p_FinCi in respect to camera clone state
      // p_FC 对克隆6自由度位姿的导数（旋转3+平移3） ∂^Cpf/∂克隆位姿
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
      // ③ 锚点状态（锚点位姿/锚点外参）雅可比：累加
      for (size_t i = 0; i < dpfg_dx_order.size(); i++) {
        H_x.block(2 * c, map_hx[dpfg_dx_order.at(i)], 2, dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg * dpfg_dx.at(i);
      }

      //=========================================================================
      //=========================================================================

      // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
      // ④相机外参与内参雅可比 累加
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
      // ⑤ 内参畸变
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
 * 要把Hf这项从方程中消掉，做法是对Hf做QR分解，得到Hf = Q [R; 0]，然后左乘Q^T，得到
 * Hf = [Q1  Q2] [R1] = Q1 R1
 *               [0 ]
 * 其中Q2(2m x (2m-3)) 张成了Hf的左零空间，因为Q2^T*Hf = Q2^T*Q1*R1 = 0，所以左乘Q2^T就可以把Hf消掉，得到
 * Q2^T r = Q2^T Hx δx + Q2^T Hf δf + Q2^T n  
 * 即 r0 = Ho δx + n0
 * 也就是只保留了对状态误差 δx 的约束。这个过程就是零空间投影。
 * 量测维度从2m 降到了 2m-3，之后用r0和Ho去走标准EKF更新即可。
 * 但是为什么代码中用GINVENS旋转来做零空间投影而不是直接用QR分解呢？
 * 因为QR分解会产生一个正交矩阵Q，Q的列数是Hf的行数，Hf是2m x 3(或1)的矩阵，所以Q是2m x 2m的矩阵，
 * 左乘Q^T会把r和Hx都变成2m维的向量和矩阵，而我们只需要保留2m-3维的约束，
 * 所以用Givens旋转可以直接把Hf变成上三角矩阵，然后把下方的零空间投影掉，得到Ho和r0。
 * 代码中用Ginvens旋转，并且不显示构造Q，直接对增广矩阵[Hf | Hx | r]原地做行旋转
 * Givens Rotation 的优势:
 * 1. 不需要显式构造Q矩阵，节省内存和计算量
 * 2. 非常容易同时变换增广矩阵 Hf Hx 和 r，保证线性化方程的一致性
 * 3. 对稀疏矩阵更高效，因为每次旋转只影响两行，计算量小   非常适合这种“小列数、长行数”的矩阵
 * 4. 数值稳定性好，适合在迭代优化中使用
 * 5. 直接原地操作矩阵，cache locality / 内存开销更好
 * @param H_f 残差对特征误差的雅可比 2m x 3(或2m x 1) m是该特征的总观测次数，投影后变为0
 * @param H_x 残差对状态误差的雅可比 2m x 状态维数  投影后变为 2m-3 x 状态维数
 * @param res 残差 2m x 1, 投影后变为 2m-3 x 1
 */
void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  // 核心 Givens 旋转消元过程：: 
  // 对增广矩阵[Hf | Hx | r]原地执行正交变换，将Hf变为上三角矩阵，同时将相同的变换应用到Hx和r上，从而自动实现零空间投影。
  // Givens 旋转 是一种实现 QR 分解的数值稳定算法，它通过一系列平面旋转变换，逐个将 H_f 的下三角元素消为零。
  // 每次旋转只影响两行，计算量小，适合稀疏矩阵
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) { // 从最后一行向上，直到 n+1
      // Givens matrix G
      // 输入两个数a = H_f(m - 1, n), b = H_f(m, n)，构造一个旋转矩阵G，使得G * [a; b] = [r; 0]，即把下方的元素消为零
      // 其中 r = sqrt(a^2 + b^2)，cosθ = a/r，sinθ = -b/r
      // 也就是说，这个旋转把b变为0，同时把a变为r，旋转矩阵G是一个2x2的正交矩阵
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n)); // 对 (m-1, n) 和 (m, n) 这两个元素构造 Givens 旋转
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      // 应用旋转到三个矩阵的相应行
      // 对当前矩阵(如H_f)的第 m-1 行和第 m 行左乘旋转矩阵(的共轭转置)。
      // 由于makeGivens构造的旋转矩阵通常为G = [[c, s], [-s, c]]，它的共轭转置是G^T = [[c, -s], [s, c]]
      // 这正好是保证消元方向正确所需要的
      // 为什么对Hf只从第n列开始应用旋转？因为前n-1列已经被消元为上三角了，如果再对它们应用新的旋转，会破坏已经置零的结构，
      // 而从第 n 列开始应用，可以保证新的旋转只影响当前列及之后的列，不会影响已完成的列;
      // 实际上，由于我们只关心消元，对前 n 列不做变化，它们已满足上三角
      // 为什么对 Hx 和 res 是从整列（第 0 列）开始?
      // 因为这两个矩阵没有类似 H_f 的上三角结构，必须将完整的旋转变换作用到它们的所有列，以保持整个增广矩阵的正交变换一致性。
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }
  // 执行完所有循环后 Hf 被变换为[R; 0]的形式，其中R是上三角矩阵，0是零矩阵。此时Hf的下方部分已经被消掉。
  // 此时，Hx和res也被同样的正交变换旋转为 Hx' = Q^T Hx, res' = Q^T res，其中Q是由所有Givens旋转组成的正交矩阵。
  // 此时，我们可以将增广矩阵分块: 
  // 上半部分(前n行): 对应Hf的行空间，包含特征点信息。这部分数据与特征点有关，在不知道精确特征点时不可靠。
  // 下半部分(后2m-n行): 对应Hf的左零空间，因为Hf'的下半部分是零矩阵，所以这部分数据与特征点无关，可以用于状态更新。

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  // 只保留底部 rows-cols 行
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
}

/* 零空间投影 和 测量压缩 的区别: 
1.作用对象: 
  - 零空间投影: 作用于特征点误差的雅可比 H_f，将其消掉，只保留对状态误差的约束。
  - 测量压缩: 作用于观测对状态误差的雅可比 H_x 和残差 res，将其行数压缩到最多等于状态维数。
2.目的:
  - 零空间投影: 把特征点误差从方程里面删除，使得更新只依赖于状态误差。
  - 测量压缩: 把冗余行压缩掉，减少计算量和内存占用，提高EKF更新的效率，同时保持信息量不变。
3. H矩阵特征:
  - 零空间投影: H_f 是一个 2m x 3(或1) 的矩阵，m是该特征的总观测次数，表示残差对特征误差的雅可比。
  - 测量压缩: H_x 是一个 2m x 状态维数 的矩阵，表示残差对状态误差的雅可比。
3.零空:
  - 零空间投影: 消完后留下下面的零空间行。 
  - 测量压缩: 消完后留下上面的三角行。 
4.维度变化:
  - 零空间投影: 行数2m→2m−3。
  - 测量压缩: 行数：2m→min(2m,n)。
两者共用同一套 Givens 消元机器，只是"保留哪一半"不同：零空间投影留下底部（与特征无关的约束），压缩留下顶部（上三角的独立约束）
*/

/**
 * @brief Compress the measurement Jacobian and residual using Givens rotations.
 * 这个函数用 Givens 正交旋转把"上千行残差"无损压缩到"最多状态维数"，把 EKF 更新里那个要反复求逆的大矩阵从千维缩到几十维
 * 由于上一步Hx_big等拼接完所有好特征后，Hx是一个"又高又瘦"的矩阵（行数 = 上千个残差，列数 = 几十个状态）。
 * 这个函数用正交旋转把它的行数压到最多等于状态维数，而信息量完全不变——因为 EKF 更新其实只需要这么多独立约束。
 * 本质上是在 MSCKF 已经通过零空间投影消掉特征点状态以后，再对“大量视觉残差”做一次 QR 型测量压缩
 * 很多条residual -> 最多和状态维度一样多的独立约束，剩下的都是冗余约束，压缩掉就行了
 * 它不是简单丢测量，也不是降采样，而是在高斯线性模型下把对状态有用的信息完整保留下来，把那些落在 Hx左零空间里的纯噪声方向丢掉
 * 零空间投影后，特征已经不再出现在状态里
 * r = Hx*δx + n 其中Hx mxn r mx1 n~N(0,sigma^2*I)
 * 对Hx做QR分解，得到Hx = Q [R; 0]，其中R是上三角矩阵，0是零矩阵 Q=[Q1 Q2]
 * 又因为Q是正交矩阵，有Q^TQ=I；所以Hx = [Q1 Q2][R; 0] = Q1 R 所以有 Q1^T*Hx = R，Q2^T*Hx = 0
 * 左乘Q^T，得到 Q^T r = Q^T Hx δx + Q^T n = [R; 0] δx + Q^T n
 * 展开 [Q1^T*r] = [R] δx + [Q1^T*n]
 *     [Q2^T*r]   [0]       [Q2^T*n]
 * 第一部分 r1 = R δx + n1，n1 ~ N(0,sigma^2*I) 
 * 第二部分 r2 = 0 δx + n2，n2 ~ N(0,sigma^2*I),其中r2 = Q2^T*r，n2 = Q2^T*n
 * 只保留上半部分，得到 r0 = R δx + n0，其中n0 ~ N(0,sigma^2*I)，r0 mx1，R mxn，m = min(2m, n)，n是状态维数
 * 这样就把原始的2m维残差压缩到最多n维，信息量不变，EKF更新只需要用r0和R就行了
 * @param H_x The measurement Jacobian matrix to be compressed.
 * @param res The residual vector to be compressed.
 */
void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  // 观测约束已经足够少，不需要压缩，直接返回。
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) { // 逐列消元
    for (int m = (int)H_x.rows() - 1; m > n; m--) { // 从最后一行往上
      // Givens matrix G ，用 (m-1,n) 和 (m,n) 两个元素构造旋转，目的是把 (m,n) 消成 0
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      // 最终 Hx变成上三角：前 n 行非零、其余全零
      // 为什么 Hx只从第 n 列开始旋转，而 res 从第 0 列开始？
      // Hx前 n−1 列已经完成消元、对角线以下全是 0（这正是我们想要的三角结构）。
      // 如果对这些已完成列再施转，理论上等价（0 旋转还是 0），但没必要——从第 n 列开始即可，省计算。
      // 而 res 没有这种结构约束，必须整列一起旋转，保证 Q 一致地作用在增广系统上。
      // 把旋转作用到 H_x 的第 m-1、m 两行（从第 n 列开始）
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      // 同样的旋转作用到 res 的第 m-1、m 两行（整列）
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols()); // 丢掉下方全部为零的行，只保留前 r 行
  res.conservativeResize(r, res.cols());
}
