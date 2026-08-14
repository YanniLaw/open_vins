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
  // std::unordered_map<size_t, std::vector<double>> timestamps; // 该特征点在每个相机上的观测时间戳列表
  for (auto const &pair : feat->timestamps) {
    total_meas += (int)pair.second.size(); // 统计该特征点在所有相机上的总观测数
    if (pair.second.size() > most_meas) {
      anchor_most_meas = pair.first;
      most_meas = pair.second.size();
    }
  }
  feat->anchor_cam_id = anchor_most_meas; // 选择观测数最多的相机作为锚点相机
  feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back(); // 以锚点相机最后一帧观测作为锚点时间戳

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
  // 为什么代码特意选锚点系而不是世界系?
  // 1. 数值稳定性：锚点系以相机位姿为原点，坐标量级小、靠近原点；而世界系坐标可能很大（尤其经过长时间积分），直接求解 
  // APf = b 会引入较大的数值误差。这正是代码里把 p_CiinG 做 R_GtoA * (p_CiinG - p_AinG) 平移的目的。
  // 2. 后续精化本来就要锚点系结果：single_gaussnewton 用的是锚点系下的逆深度参数化 (α,β,ρ)，它直接从 feat->p_FinA 取初值,
  // double rho = 1 / feat->p_FinA(2); 所以线性三角化直接输出锚点系结果，省去一次坐标变换
  // 3. 需要世界系时最后再转回，也就是函数结尾
  ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
  const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

  // Loop through each camera for this feature 遍历所有测量，累加线性系统
  // std::unordered_map<size_t, std::vector<double>> timestamps
  for (auto const &pair : feat->timestamps) { // 遍历所有相机的观测时间戳列表，pair.first = 相机索引，pair.second = 该相机的观测时间戳列表

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) { // 遍历该相机的每个观测时间戳

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
  // 这里的 p_f 是锚点系下的特征点位置，因为在前面的循环中，所有的观测都被转换到了锚点坐标系下!!!
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
  // 质量检测: 
  // 1. 条件数检验是关键：如果各相机视线近乎平行（相机几乎没有位移基线），A 接近奇异，条件数巨大——此时三角化结果不可信，
  // 必须返回 false（在 UpdaterMSCKF::update 里该特征会被剔除）
  // 2. 距离范围检验: z 坐标超出 [min_dist, max_dist]（特征太近或太远）也判为失败
  // 3. NaN 检验: 如果三角化结果为 NaN，也判为失败
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
 * 为什么需要它?
 * 线性三角化（DLT）只是最小二乘闭式解，对噪声敏感、也不是最小化重投影误差意义上的最优，还可能会产生退化情况（例如相机视线接近共线）。
 * single_gaussnewton 把问题变成非线性最小二乘，用迭代优化精化初值，提高精度以及鲁棒性。
 * @param feat 
 * @param clonesCAM 
 * @return true 
 * @return false 
 */
bool FeatureInitializer::single_gaussnewton(std::shared_ptr<Feature> feat,
                                            std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // Get into inverse depth
  // 把锚点系下的3D坐标换成锚点系下的逆深度表示(α, β, ρ),好处: 
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

  // Get the position of the anchor pose 锚点坐标系位姿
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
          /*
          计算预测的归一化坐标 z=[hi1/hi3, hi2/hi3] 关于θ=(α, β, ρ) 的偏导数
          这里其实是用的链式求导法则: z(θ) = z(h(θ))，先对内层 hi1, hi2, hi3 求导，再对外层θ 求导，最后相乘。
          1) 内层求导: hi1, hi2, hi3 关于 (α, β, ρ) 的偏导数
          内层求导: hi1, hi2, hi3 关于 (α, β, ρ) 的偏导数
          已知h的表达式)
          h1 = R00* α + R01* β + R02 + ρ * p0
          h2 = R10* α + R11* β + R12 + ρ * p1
          h3 = R20* α + R21* β + R22 + ρ * p2
          h是参数α, β, ρ 的线性函数，所以偏导数就是系数本身: 
          ∂h1/∂α = R00, ∂h1/∂β = R01, ∂h1/∂ρ = p0
          ∂h2/∂α = R10, ∂h2/∂β = R11, ∂h2/∂ρ = p1
          ∂h3/∂α = R20, ∂h3/∂β = R21, ∂h3/∂ρ = p2
          写成矩阵就是
          [ R00  R01  p0 ]
          [ R10  R11  p1 ] = ∂h / ∂θ
          [ R20  R21  p2 ] 
          2) 外层求导: z 关于 hi1, hi2, hi3 的偏导数
          外层(透视投影) z1 = hi1 / hi3, z2 = hi2 / hi3
          对外层求雅可比，我们需要用到商法则，即 (u/v)' = (u'v - uv') / v^2
          对z1 = h1 / h3, 有 ∂z1/∂h1 = 1/h3, ∂z1/∂h2 = 0, ∂z1/∂h3 = -h1/(h3^2)
          对z2 = h2 / h3, 有 ∂z2/∂h1 = 0, ∂z2/∂h2 = 1/h3, ∂z2/∂h3 = -h2/(h3^2)
          写成矩阵就是
          [ 1/ h3  0    -h1/(h3^2) ] = ∂z / ∂h
          [ 0     1/h3  -h2/(h3^2) ]
          3) 链式求导: ∂z/∂θ = ∂z/∂h * ∂h/∂θ (两个矩阵相乘即可)
          ∂z1/∂α = 1/ h3 * R00 + (-h1/(h3^2)) * R20
          ∂z1/∂β = 1/ h3 * R01 + (-h1/(h3^2)) * R21
          ∂z1/∂ρ = 1/ h3 * p0 + (-h1/(h3^2)) * p2
          ∂z2/∂α = 1/ h3 * R10 + (-h2/(h3^2)) * R20
          ∂z2/∂β = 1/ h3 * R11 + (-h2/(h3^2)) * R21
          ∂z2/∂ρ = 1/ h3 * p1 + (-h2/(h3^2)) * p2
          */
          // Middle variables of the system
          // 计算公式见 https://docs.openvins.com/update-featinit.html#featinit-nonlinear
          // hi1, hi2, hi3 是特征点在相机 i 系下的齐次坐标(除以了锚点深度A_Zf)
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
          Eigen::Matrix<double, 2, 3> H; // 其实是残差对变量的雅可比矩阵 (Jacobian)
          H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
          // Calculate residual
          Eigen::Matrix<float, 2, 1> z;
          z << hi1 / hi3, hi2 / hi3; // 预测的归一化坐标（透视投影）
          Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z; // 残差 = 观测 - 预测

          //=====================================================================================
          //=====================================================================================

          // Append to our summation variables
          err += std::pow(res.norm(), 2); // 代价
          grad.noalias() += H.transpose() * res.cast<double>(); // J^T * r
          Hess.noalias() += H.transpose() * H; // J^T * J  高斯牛顿近似海森矩阵
          // 用 J^T * J 近似海森矩阵，本质上是牺牲掉残差带来的二阶曲率信息，换取绝对的正定性保证和一阶导数的廉价计算。
          /* 为什么可以用J^T * J 近似海森矩阵？
          假设目标函数是残差的平方和 F(x) = 1/2 * r(x)^T * r(x)
          其中 r(x) 是残差向量，J = ∂r/∂x 是雅可比矩阵
          则梯度 ∇F(x) = J^T * r(x)
          海森矩阵 H = ∇^2 F(x) = J^T * J + Σ ri * ∇^2 ri
          当残差较小时，第二项可以忽略，因此 H ≈ J^T * J
          用 J^T * J 近似海森矩阵有什么好处??？
          1. 正定性: 真实海森矩阵H可能不是正定的(比如鞍点或者非凸区域)，而 J^T * J 总是半正定的，加上阻尼项就是正定，保证永远是下降方向;
          2. 计算代价: 计算海森矩阵H需要计算复杂的二阶偏导, 而 J^T * J 只需一阶导数，比求二阶导快得多，且极度适合并行计算；
          3. 存储代价: 海森矩阵H需要存储二阶张量（3维数据），内存爆炸，而 J^T * J 只需存储一阶导数(二维矩阵)，节省大量内存。
          
          note: 当残差r很大时，扔掉第二项会导致近似误差极大，算法收敛及慢甚至发散。
          补救措施 --Levenberg-Marquardt 算法，LM算法
          在J^T * J 的基础上加入阻尼项，即 H ≈ J^T * J + λI，可以在残差较大时提高算法的稳定性，这就是 LM 算法的核心思想。
          当λ很大时，退化为梯度下降(稳健但慢，适合远离最优解时); 
          当λ很小时，接近高斯牛顿J^T * J (快，但可能发散，依赖初始值，适合接近最优解时);
          */
        }
      }
    }

    // Solve Levenberg iteration
    // λ 大 → 接近梯度下降（稳、慢）；
    // λ 小 → 接近高斯牛顿（快、但可能发散）
    // 代码通过代价是否下降来自适应调节
    Eigen::Matrix<double, 3, 3> Hess_l = Hess;
    for (size_t r = 0; r < (size_t)Hess.rows(); r++) {
      // 注意，这里不是教科书上常见的J^T * J+λI 的形式，而是Marquardt 原始论文里的变体——"按对角线缩放"(被注释掉的那行才是标准λI版本)
      // 为什么用 λ·diag(H) 而不是 λI？
      // 因为参数 𝛼,𝛽, ρ的量纲和数值量级差异巨大: 𝛼,𝛽是归一化坐标（量级 ~1),而 ρ是逆深度（量级可能是 0.01~10）。
      // 用 λI 时，同样的阻尼强度对 ρ 和 𝛼 效果完全不同；
      // 而 λdiag(H) 会按每个参数自身曲率（H_ii 反映了该方向的信息量）自适应地加阻尼——曲率大的方向阻尼大，天然对参数缩放不变（scale-invariant），数值上更稳健。
      Hess_l(r, r) *= (1.0 + lam); // 阻尼处理，对角线放大(1+λ)，这是 LM 与纯高斯牛顿的区别
    }
    // 解线性方程组求步长 dx，阻尼后Hess_l严格正定，解出的dx一定是下降方向
    // colPivHouseholderQr是带列主元的 QR 分解求解器：数值稳定性好，不需要矩阵正定前提，对 3×3 小矩阵开销可忽略。
    // 相比直接求逆或 Cholesky（LLT/LDLT），QR 对病态/接近奇异的 海森矩阵H 更鲁棒
    Eigen::Matrix<double, 3, 1> dx = Hess_l.colPivHouseholderQr().solve(grad);
    // Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

    // Check if error has gone down
    double cost = compute_error(clonesCAM, feat, alpha + dx(0, 0), beta + dx(1, 0), rho + dx(2, 0));

    // Debug print
    // std::stringstream ss;
    // ss << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;
    // PRINT_DEBUG(ss.str().c_str());

    // Check if converged 收敛判断(提前退出)
    // 代价确实没有上升 且 相对代价下降率小于最小阈值(改进已经微不足道，说明接近最优解，继续迭代无意义)，认为收敛
    if (cost <= cost_old && (cost_old - cost) / cost_old < _options.min_dcost) {
      alpha += dx(0, 0);  // 更新 alpha
      beta += dx(1, 0);   // 更新 beta
      rho += dx(2, 0);    // 更新 rho
      eps = 0; // 让 while 条件 eps > min_dx 失效
      break;
    }

    // If cost is lowered, accept step
    // Else inflate lambda (try to make more stable)
    // 代价下降 → 接受步长，λ 减小，下一步更接近高斯牛顿；
    // 代价上升 → 拒绝步长，λ 增大，下一步更接近梯度下降
    if (cost <= cost_old) {
      recompute = true; // 角度θ 更新，需要重新计算雅可比和残差
      cost_old = cost;
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      runs++;
      lam = lam / _options.lam_mult; // 减小阻尼λ ，下一步更接近高斯牛顿
      eps = dx.norm();  // 记录步长，供下次收敛/退出判断
    } else {
      // recompute = false（省计算的关键）
      // 步长被拒绝意味着 θ 根本没变，既然 θ 没变，累加的Hessian和梯度仍然有效，无需重新计算,只是把λ放大后重新解一遍阻尼法方程再求dx
      // λ 增大：步长走过头了，说明当前 λ 太激进，放大 λ 让系统更接近梯度下降（保守、稳），期望下一步更小更安全
      // 注意：拒绝分支不更新 eps 也不 runs++。反复被拒绝时，λ 不断翻倍，最终会触发 while 条件 lam < max_lamda 失效而退出——这是"优化失败"的退出路径。
      recompute = false; // θ 没变，雅可比仍然有效！
      lam = lam * _options.lam_mult; // λ 增大 → 更接近梯度下降
      continue; // 回到循环顶部，但不重算雅可比
    }
  }

  // 质量检验
  // LM 全程在逆深度参数化下优化，迭代结束得到 (α,β,ρ)，按定义还原回锚点系 3D 坐标
  // Revert to standard, and set to all  逆深度 → 欧氏坐标
  feat->p_FinA(0) = alpha / rho;
  feat->p_FinA(1) = beta / rho;
  feat->p_FinA(2) = 1 / rho;

  // Get tangent plane to x_hat
  // 做基线比（baseline ratio）检验——这是这里特有的退化检测
  // QR分解 p = QR,其中 Q (3*3)是正交矩阵，R 是上三角矩阵，Q 的第一列就是 p 的归一化方向；Q 的后两列张成与 p 垂直的二维平面
  // 对3x1向量A_Pf 进行QR分解，对一个向量v来说，QR得到的正交矩阵Q满足: 
  // Q^T v = [||v||, 0, 0]^T, 
  // 也就是说，Q的第0列 ∝v（沿视线方向），第1，第2列张成与v垂直的平面----这个平面就是过锚点，垂直于观测视线的切平面
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->p_FinA);
  Eigen::MatrixXd Q = qr.householderQ();
  // 为什么要投影到切平面？
  // 因为三角化估计深度靠的是视差（parallax），而只有垂直于视线方向的基线分量才产生视差
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
      // Dot product camera pose and nullspace 取第1.2列 Q.block(0, 1, 3, 2)，即Q_⊥，其列张成与视线方向垂直的切平面
      // 把相机位置投影到切平面上，norm 就是"有效基线"
      // beff=||Q_⊥^T * A_P_Ci|| ， Q_⊥^T * A_P_Ci是相机位姿在切平面上的2D投影
      // 它的模长 = 相机 Ci相对锚点的有效基线（垂直于视线那部分）
      // 若所有相机都几乎沿视线方向排列（基线平行于视线），beff≈0 → 无视差 → 深度不可观测
      double base_line = ((Q.block(0, 1, 3, 2)).transpose() * p_CiinA).norm();
      if (base_line > base_line_max) // 切平面投影 → 有效基线，取所有观测中的最大值,它衡量"这次三角化能提供多少视差信息"
        base_line_max = base_line;
    }
  }
  // std::stringstream ss;
  // ss << feat->featid << " - max base " << (feat->p_FinA.norm() / base_line_max) << " - z " << feat->p_FinA(2) << std::endl;
  // PRINT_DEBUG(ss.str().c_str());

  // Check if this feature is bad or not
  // 1. If the feature is too close
  // 2. If the feature is invalid
  // 3. If the baseline ratio is large 基线比过大->退化
  // 4. 数值异常（NaN）
  // 基线比为什么重要？
  // 因为ratio = ||p_FinA|| / base_line_max = 特征距离 / 最大有效基线
  // 物理含义：特征离相机有多远 vs 相机之间能提供多少视差
  // 如果特征距离远大于有效基线（比如 ratio > 100），说明视差极其微小，深度方向几乎不可观测。
  // 此时 LM 优化可能把特征"推"到无穷远来糊弄残差（z→∞ 时投影趋于固定方向，能压低代价），但这是退化解；
  // 该检验正是为了拦截这种"看似代价很低、实则深度不可信"的结果
  // note: 这也解释了为什么前面 single_triangulation 用"条件数"检测、而精化后用"基线比"——因为优化收敛到退化区时，
  // J^T*J 可能依然良态，但几何上视差已经不足以支撑深度估计，基线比能从几何信息量角度兜底。
  if (feat->p_FinA(2) < _options.min_dist || feat->p_FinA(2) > _options.max_dist ||
      (feat->p_FinA.norm() / base_line_max) > _options.max_baseline || std::isnan(feat->p_FinA.norm())) {
    return false;
  }

  // Finally get position in global frame
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  return true;
}

/**
 * @brief Compute the error for a given feature and its associated camera poses.
 * https://docs.openvins.com/update-featinit.html#featinit-nonlinear 
 * @param clonesCAM 
 * @param feat 
 * @param alpha 
 * @param beta 
 * @param rho 
 * @return double 
 */
double FeatureInitializer::compute_error(std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                         std::shared_ptr<Feature> feat, double alpha, double beta, double rho) {

  // Total error
  double err = 0;

  // Get the position of the anchor pose 该特征点的锚点坐标系位姿
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

  // Loop through each camera for this feature 遍历观测到该特征点的所有相机
  for (auto const &pair : feat->timestamps) {
    // Add CAM_I features 遍历每个相机对该特征点的所有观测
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
      // 公式中的h本质上是特征点在Ci坐标系下的3D坐标乘以锚点深度A_Zf的倒数(即除以锚点深度)
      // 这个缩放对计算预测的归一化坐标 z=[ui,vi] 没有影响
      double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
      double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
      double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
      // Calculate residual
      Eigen::Matrix<float, 2, 1> z;
      z << hi1 / hi3, hi2 / hi3; // 预测的归一化坐标（透视投影）
      Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z;
      // Append to our summation variables
      err += pow(res.norm(), 2);
    }
  }

  return err;
}