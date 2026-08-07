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

#include "FeatureInitializer.h"

#include "Feature.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;

/**
 * @brief Triangulate a 3D feature point from multiple camera observations using linear least squares.
 * 实现 DLT 式线性三角化：以观测最多的相机为锚点，把每个测量时刻的相机位姿变换到锚点系，
 * 用"视线方向与特征-相机向量平行"的叉积约束累加出线性系统 APf=b，
 * QR 求解后通过条件数 + 距离范围检验，最终把特征位置同时存为锚点系 p_FinA 和全局系 p_FinG。
 * 它给出的是初值，之后由 single_gaussnewton 做非线性精化。
 * 与single_triangulation_1d 的区别
 * 1. 求解量：single_triangulation 求解的是特征点的 3D 坐标(3个自由度)，而 single_triangulation_1d 只求解深度(1个自由度)。
 * 2. 前提: single_triangulation 无特殊前提，而 single_triangulation_1d 假设锚点观测的视线方向是精确的（视为真值）。
 * 3. 线性系统: single_triangulation 构建的是 3x3 的线性系统，而 single_triangulation_1d 构建的是 1x1 的线性系统。
 * 4. 速度: single_triangulation_1d 由于只求解深度，计算量更小，速度更快。
 * 5. 适用: single_triangulation 适用于一般情况，而 single_triangulation_1d 适用于需要速度、且锚点视线可靠时（对应配置 triangulate_1d）。
 * @param feat The feature to be triangulated.
 * @param clonesCAM A map of camera poses at each clone timestamp.
 * @return true If the triangulation was successful.
 * @return false If the triangulation failed (e.g., due to degeneracy or parallel rays).
 */
bool FeatureInitializer::single_triangulation(std::shared_ptr<Feature> feat,
                                              std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // Total number of measurements
  // Also set the first measurement to be the anchor frame
  // 选择观测数最多的相机作为锚点
  int total_meas = 0;   // 该特征点在不同相机上的总观测数
  size_t anchor_most_meas = 0;
  size_t most_meas = 0; // 该特征点在单个相机上的最大观测数
  for (auto const &pair : feat->timestamps) {
    total_meas += (int)pair.second.size();
    if (pair.second.size() > most_meas) {
      anchor_most_meas = pair.first;
      most_meas = pair.second.size();
    }
  }
  feat->anchor_cam_id = anchor_most_meas; // 选择观测数最多的相机作为锚点相机
  feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back(); // 该相机最后一帧

  // Our linear system matrices
  /**
   * @brief 核心思想
   * 可参考https://docs.openvins.com/update-featinit.html#featinit-linear
   * 已知: 特征在若干相机位姿下都有观测，且每个观测给出一个归一化视线方向bi(bearing)。
   * 几何约束: 特征点Pf到相机Pci坐标系下的向量，必须与视线方向bi平行，即差积为零:
   *   bi × (Pf - Pci) = 0
   * 用反对称矩阵[bi]x 表示叉积，即 [bi]x (Pf - Pci) = 0 ==> [bi]x Pf = [bi]x Pci
   * 两边同时左乘 [bi]x^T构造对称正定系统，把所有测量累加成一个大的最小二乘问题: 
   *  [bi]x^T [bi]x Pf = [bi]x^T [bi]x Pci，
   * 即 Ai Pf = Ai Pci，累加所有观测得到 APf = b。
   * A 是从所有观测累加的 3x3 矩阵，b 是从所有观测累加的 3x1 向量。
   * 解这个线性系统即得特征位置。多视角越多，A 越满秩、解越稳定。
   * 通过 QR 分解求解该线性系统，得到特征点的初始 3D 位置。
   */
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero(); // 线性系统的系数矩阵
  Eigen::Vector3d b = Eigen::Vector3d::Zero(); // 线性系统的常数向量

  // Get the position of the anchor pose
  // 锚点 = 观测最多的相机 的最后一张图。三角化结果 p_FinA 以锚点相机位姿为参考系表示
  ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
  const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

  // Loop through each camera for this feature 遍历所有测量，累加线性系统
  for (auto const &pair : feat->timestamps) {

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

      // Get the position of this clone in the global 该测量时刻的相机位姿
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

      // Convert current position relative to anchor
      // 当前相机位姿 → 锚点坐标系（去掉锚点的平移/旋转)
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

      // Get the UV coordinate normal
      // 视线方向：归一化坐标 (u_n, v_n, 1) → 转到锚点系 → 归一化
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
      b_i = R_AtoCi.transpose() * b_i; // 把视线方向从当前相机坐标系转换到锚点相机坐标系
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
      A += Ai;
      b += Ai * p_CiinA;
    }
  }

  // Solve the linear system 带列主元的 QR 分解求解线性系统 APf=b， 数值稳定性较好
  Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

  // Check A and p_f
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
  Eigen::MatrixXd singularValues;
  singularValues.resize(svd.singularValues().rows(), 1);
  singularValues = svd.singularValues();
  double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0); // 条件数 = 最大奇异值 / 最小奇异值

  // std::stringstream ss;
  // ss << feat->featid << " - cond " << std::abs(condA) << " - z " << p_f(2, 0) << std::endl;
  // PRINT_DEBUG(ss.str().c_str());

  // If we have a bad condition number, or it is too close
  // Then set the flag for bad (i.e. set z-axis to nan)
  // 条件数检验是关键：如果各相机视线近乎平行（相机几乎没有位移基线），A 接近奇异，条件数巨大——此时三角化结果不可信，
  // 必须返回 false（在 UpdaterMSCKF::update 里该特征会被剔除）
  if (std::abs(condA) > _options.max_cond_number || p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist ||
      std::isnan(p_f.norm())) {
    return false;
  }

  // Store it in our feature object
  feat->p_FinA = p_f; // 锚点系下的 3D 位置
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG; // 全局系下的 3D 位置
  return true;
}

bool FeatureInitializer::single_triangulation_1d(std::shared_ptr<Feature> feat,
                                                 std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // Total number of measurements
  // Also set the first measurement to be the anchor frame
  int total_meas = 0;
  size_t anchor_most_meas = 0;
  size_t most_meas = 0;
  for (auto const &pair : feat->timestamps) {
    total_meas += (int)pair.second.size();
    if (pair.second.size() > most_meas) {
      anchor_most_meas = pair.first;
      most_meas = pair.second.size();
    }
  }
  feat->anchor_cam_id = anchor_most_meas;
  feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back();
  size_t idx_anchor_bearing = feat->timestamps.at(feat->anchor_cam_id).size() - 1;

  // Our linear system matrices
  double A = 0.0;
  double b = 0.0;

  // Get the position of the anchor pose
  ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
  const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

  // Get bearing in anchor frame
  Eigen::Matrix<double, 3, 1> bearing_inA;
  bearing_inA << feat->uvs_norm.at(feat->anchor_cam_id).at(idx_anchor_bearing)(0),
      feat->uvs_norm.at(feat->anchor_cam_id).at(idx_anchor_bearing)(1), 1;
  bearing_inA = bearing_inA / bearing_inA.norm();

  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

      // Skip the anchor bearing
      if ((int)pair.first == feat->anchor_cam_id && m == idx_anchor_bearing)
        continue;

      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

      // Get the UV coordinate normal
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
      b_i = R_AtoCi.transpose() * b_i;
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Vector3d BperpBanchor = Bperp * bearing_inA;
      A += BperpBanchor.dot(BperpBanchor);
      b += BperpBanchor.dot(Bperp * p_CiinA);
    }
  }

  // Solve the linear system
  double depth = b / A;
  Eigen::MatrixXd p_f = depth * bearing_inA;

  // Then set the flag for bad (i.e. set z-axis to nan)
  if (p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist || std::isnan(p_f.norm())) {
    return false;
  }

  // Store it in our feature object
  feat->p_FinA = p_f;
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  return true;
}

/**
 * @brief Perform single feature triangulation using the Gauss-Newton optimization method.
 * 用 LM 算法在逆深度参数化下最小化所有视角的重投影残差，对 single_triangulation 给的线性初值做非线性精化：
 * 通过 λ 自适应调节在高斯牛顿（快）与梯度下降（稳）之间切换，迭代收敛后把结果转回欧氏坐标，并用"切平面有效基线比"检验退化，
 * 最终输出精化后的 p_FinA/p_FinG——这就是 MSCKF 特征三角化的"线性初值 + 非线性精化"两步法。
 * @param feat 
 * @param clonesCAM 
 * @return true 
 * @return false 
 */
bool FeatureInitializer::single_gaussnewton(std::shared_ptr<Feature> feat,
                                            std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // Get into inverse depth
  // 把3D坐标换成锚点系下的逆深度表示(α, β, ρ),好处: 
  // 1. 特征位置 = 1/ρ * [α, β, 1]^T,参数化更稳定;
  // 2. 逆深度对近/远特征尺度更友好，数值条件更好。
  double rho = 1 / feat->p_FinA(2); // 逆深度参数化：ρ = 1/z，z = feat->p_FinA(2) 是锚点系下的深度
  double alpha = feat->p_FinA(0) / feat->p_FinA(2); // α = x/z
  double beta = feat->p_FinA(1) / feat->p_FinA(2);  // β = y/z

  // Optimization parameters
  double lam = _options.init_lamda;
  double eps = 10000;
  int runs = 0;
  // 优化目标（重投影残差）
  // 对每个观测，用当前 (α,β,ρ) 计算预测的归一化坐标 z，与真实观测 uvs_norm 求差
  // Variables used in the optimization
  bool recompute = true;
  Eigen::Matrix<double, 3, 3> Hess = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 1> grad = Eigen::Matrix<double, 3, 1>::Zero();

  // Cost at the last iteration
  double cost_old = compute_error(clonesCAM, feat, alpha, beta, rho);

  // Get the position of the anchor pose
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

  // Loop till we have either
  // 1. Reached our max iteration count
  // 2. System is unstable
  // 3. System has converged
  while (runs < _options.max_runs && lam < _options.max_lamda && eps > _options.min_dx) {

    // Triggers a recomputation of jacobians/information/gradients
    if (recompute) {

      Hess.setZero();
      grad.setZero();

      double err = 0;

      // Loop through each camera for this feature
      for (auto const &pair : feat->timestamps) {

        // Add CAM_I features
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

          //=====================================================================================
          //=====================================================================================

          // Get the position of this clone in the global
          const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).Rot();
          const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).pos();
          // Convert current position relative to anchor
          Eigen::Matrix<double, 3, 3> R_AtoCi;
          R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
          Eigen::Matrix<double, 3, 1> p_CiinA;
          p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
          Eigen::Matrix<double, 3, 1> p_AinCi;
          p_AinCi.noalias() = -R_AtoCi * p_CiinA;

          //=====================================================================================
          //=====================================================================================

          // Middle variables of the system
          // 中间量：把 (α, β, ρ) 投影到相机 i 系
          double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
          double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
          double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
          // Calculate jacobian
          double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
          double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
          double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
          double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
          double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
          double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
          Eigen::Matrix<double, 2, 3> H;
          H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
          // Calculate residual
          Eigen::Matrix<float, 2, 1> z;
          z << hi1 / hi3, hi2 / hi3; // 预测的归一化坐标（透视投影）
          Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z; // 残差 = 观测 - 预测

          //=====================================================================================
          //=====================================================================================

          // Append to our summation variables
          err += std::pow(res.norm(), 2); // 代价
          grad.noalias() += H.transpose() * res.cast<double>(); // 梯度
          Hess.noalias() += H.transpose() * H; // 信息矩阵(高斯牛顿近似海森矩阵)
        }
      }
    }

    // Solve Levenberg iteration
    Eigen::Matrix<double, 3, 3> Hess_l = Hess;
    for (size_t r = 0; r < (size_t)Hess.rows(); r++) {
      Hess_l(r, r) *= (1.0 + lam);
    }

    Eigen::Matrix<double, 3, 1> dx = Hess_l.colPivHouseholderQr().solve(grad);
    // Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

    // Check if error has gone down
    double cost = compute_error(clonesCAM, feat, alpha + dx(0, 0), beta + dx(1, 0), rho + dx(2, 0));

    // Debug print
    // std::stringstream ss;
    // ss << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;
    // PRINT_DEBUG(ss.str().c_str());

    // Check if converged
    if (cost <= cost_old && (cost_old - cost) / cost_old < _options.min_dcost) {
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      eps = 0;
      break;
    }

    // If cost is lowered, accept step
    // Else inflate lambda (try to make more stable)
    if (cost <= cost_old) {
      recompute = true;
      cost_old = cost;
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      runs++;
      lam = lam / _options.lam_mult;
      eps = dx.norm();
    } else {
      recompute = false;
      lam = lam * _options.lam_mult;
      continue;
    }
  }

  // Revert to standard, and set to all  逆深度 → 欧氏坐标
  feat->p_FinA(0) = alpha / rho;
  feat->p_FinA(1) = beta / rho;
  feat->p_FinA(2) = 1 / rho;

  // Get tangent plane to x_hat
  // 做基线比（baseline ratio）检验——这是这里特有的退化检测
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->p_FinA);
  Eigen::MatrixXd Q = qr.householderQ();

  // Max baseline we have between poses
  double base_line_max = 0.0;

  // Check maximum baseline
  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {
    // Loop through the other clones to see what the max baseline is
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 1> p_CiinA = R_GtoA * (p_CiinG - p_AinG);
      // Dot product camera pose and nullspace
      double base_line = ((Q.block(0, 1, 3, 2)).transpose() * p_CiinA).norm();
      if (base_line > base_line_max)
        base_line_max = base_line;
    }
  }
  // std::stringstream ss;
  // ss << feat->featid << " - max base " << (feat->p_FinA.norm() / base_line_max) << " - z " << feat->p_FinA(2) << std::endl;
  // PRINT_DEBUG(ss.str().c_str());

  // Check if this feature is bad or not
  // 1. If the feature is too close
  // 2. If the feature is invalid
  // 3. If the baseline ratio is large
  if (feat->p_FinA(2) < _options.min_dist || feat->p_FinA(2) > _options.max_dist ||
      (feat->p_FinA.norm() / base_line_max) > _options.max_baseline || std::isnan(feat->p_FinA.norm())) {
    return false;
  }

  // Finally get position in global frame
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  return true;
}

double FeatureInitializer::compute_error(std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                         std::shared_ptr<Feature> feat, double alpha, double beta, double rho) {

  // Total error
  double err = 0;

  // Get the position of the anchor pose
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {
    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

      //=====================================================================================
      //=====================================================================================

      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
      Eigen::Matrix<double, 3, 1> p_AinCi;
      p_AinCi.noalias() = -R_AtoCi * p_CiinA;

      //=====================================================================================
      //=====================================================================================

      // Middle variables of the system
      double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
      double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
      double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
      // Calculate residual
      Eigen::Matrix<float, 2, 1> z;
      z << hi1 / hi3, hi2 / hi3;
      Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z;
      // Append to our summation variables
      err += pow(res.norm(), 2);
    }
  }

  return err;
}