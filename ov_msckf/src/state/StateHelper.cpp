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

#include "StateHelper.h"

#include "state/State.h"

#include "types/Landmark.h"
#include "utils/colors.h"
#include "utils/print.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

/**
 * @brief 传播状态协方差矩阵
 * 只传播协方差、不传播均值
 * @param state     滤波器状态
 * @param order_NEW 传播后的状态块
 * @param order_OLD 传播前的状态块
 * @param Phi       状态转移矩阵
 * @param Q         过程噪声协方差矩阵，是一个方阵
 */
void StateHelper::EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                                 const std::vector<std::shared_ptr<Type>> &order_OLD, const Eigen::MatrixXd &Phi,
                                 const Eigen::MatrixXd &Q) {

  // We need at least one old and new variable
  if (order_NEW.empty() || order_OLD.empty()) {
    PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with empty variable arrays!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Loop through our Phi order and ensure that they are continuous in memory
  // 确保参与传播的状态变量在全局状态向量中是连续存储的，这是块操作正确性的前提
  int size_order_NEW = order_NEW.at(0)->size();
  for (size_t i = 0; i < order_NEW.size() - 1; i++) {
    if (order_NEW.at(i)->id() + order_NEW.at(i)->size() != order_NEW.at(i + 1)->id()) {
      PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with non-contiguous state elements!\n" RESET);
      PRINT_ERROR(
          RED "StateHelper::EKFPropagation() - This code only support a state transition which is in the same order as the state\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    size_order_NEW += order_NEW.at(i + 1)->size();
  }

  // Size of the old phi matrix
  int size_order_OLD = order_OLD.at(0)->size();
  for (size_t i = 0; i < order_OLD.size() - 1; i++) {
    size_order_OLD += order_OLD.at(i + 1)->size();
  }

  // Assert that we have correct sizes
  // Φ 矩阵本身就是稠密连续矩阵，不是块对角稀疏矩阵。它的列按照 order_OLD 的顺序排列，行按照 order_NEW 的顺序排列。
  // 比如 order = [imu(15维), calib_dw(6维), calib_da(6维)]，那么 Φ 就是一个 27 x 27 的稠密矩阵，
  // 前 15 行对应 imu，接下来的 6 行对应 calib_dw，最后 6 行对应 calib_da；
  // 前 15 列对应 imu，接下来的 6 列对应 calib_dw，最后 6 列对应 calib_da。
  assert(size_order_NEW == Phi.rows());
  assert(size_order_OLD == Phi.cols());
  assert(size_order_NEW == Q.cols());
  assert(size_order_NEW == Q.rows());

  // Get the location in small phi for each measuring variable
  int current_it = 0;
  std::vector<int> Phi_id;  // Phi_id 记录了每个 order_OLD 中的变量在 Φ 矩阵中对应的列起始位置
  for (const auto &var : order_OLD) {
    Phi_id.push_back(current_it);
    current_it += var->size();
  }

  // Loop through all our old states and get the state transition times it
  // Cov_PhiT = [ Pxx ] [ Phi' ]'
  Eigen::MatrixXd Cov_PhiT = Eigen::MatrixXd::Zero(state->_Cov.rows(), Phi.rows());
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Cov_PhiT.noalias() +=
        state->_Cov.block(0, var->id(), state->_Cov.rows(), var->size()) * Phi.block(0, Phi_id[i], Phi.rows(), var->size()).transpose();
  }

  // Get Phi_NEW*Covariance*Phi_NEW^t + Q
  // 从矩阵 Q 的上三角部分构建一个完整的对称矩阵，并将结果存储到 Phi_Cov_PhiT 中
  Eigen::MatrixXd Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>();
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Phi_Cov_PhiT.noalias() += Phi.block(0, Phi_id[i], Phi.rows(), var->size()) * Cov_PhiT.block(var->id(), 0, var->size(), Phi.rows());
  }

  // We are good to go!
  int start_id = order_NEW.at(0)->id();
  int phi_size = Phi.rows();
  int total_size = state->_Cov.rows();
  state->_Cov.block(start_id, 0, phi_size, total_size) = Cov_PhiT.transpose();
  state->_Cov.block(0, start_id, total_size, phi_size) = Cov_PhiT;
  state->_Cov.block(start_id, start_id, phi_size, phi_size) = Phi_Cov_PhiT;

  // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
  // 提取矩阵 state->_Cov 的主对角线元素，并将它们复制到一个新的动态双精度向量 diags 中
  Eigen::VectorXd diags = state->_Cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.rows(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "StateHelper::EKFPropagation() - diagonal at %d is %.2f\n" RESET, i, diags(i));
      found_neg = true;
    }
  }
  if (found_neg) {
    std::exit(EXIT_FAILURE);
  }
}

/**
 * @brief Perform an EKF update step.
 *
 * @param state The current state to be updated.完整的误差卡尔曼滤波状态
 * @param H_order The order of the measurement variables.每列块对应的状态变量列表
 * @param H The measurement Jacobian matrix.压缩后的测量矩阵
 * @param res The measurement residual vector.压缩后的残差向量
 * @param R The measurement noise covariance matrix.测量噪声协方差矩阵
 */
void StateHelper::EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  assert(res.rows() == R.rows());
  assert(H.rows() == res.rows());
  // 预计算 M = P*H^T，P 是全局协方差矩阵，H 是压缩后的测量矩阵
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;   // 计算每一个state block在小jacobian中的位置，建立 H_order → H 中列偏移 的映射
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  // 一般来说，H并不是一个包含了完整的状态宽度的矩阵，而是只包含了参与当前观测的状态变量。
  // 在计算M的时候，其实一般做法是将H扩展为一个完整的状态宽度矩阵（其余列补零），然后计算M = P * H^T。
  // 但是 Mi = ∑_j P_{ij} * H_j^T，其中 j 只遍历参与观测的状态变量，因此可以直接用协方差矩阵中对应的块来计算，而不需要扩展 H。
  // 也就是说，M 的每一行对应一个状态变量，每一列对应一个观测残差，而 M 的每一行只需要累加它对所有观测状态块的贡献。
  // 这样做的好处是节省了内存和计算，因为不需要构造一个完整的稀疏矩阵 H，而是直接利用协方差矩阵中的块来计算 M。
  // 具体来说，对于每一个状态变量 var，它对 M 的贡献是 ∑_i P_{var, H_order[i]} * H_{*，i}^⊤ —— 只用协方差中"该状态 × 量测状态"那一小块，其他块乘出来也是 0，直接跳过。
  // H_id[i] 记录了第 i 个量测状态在 H 里的起始列，和之前 get_feature_jacobian_full 的 map_hx、 UpdaterMSCKF 的 Hx_mapping 是同一套"列-状态记账"
  // ================================
  // 对每个状态变量 var，累加它对所有量测状态块的贡献
  // M=PH^⊤的行覆盖所有状态，但列只涉及 H_order 里的状态
  // 对每个状态 var，它对 M 的贡献是 ∑_i P_{var, H_order[i]} * H_{*，i}^⊤ —— 只用协方差中"该状态 × 量测状态"那一小块，其他块乘出来也是 0，直接跳过
  // H_id[i] 记录了第 i 个量测状态在 H 里的起始列，和之前 get_feature_jacobian_full 的 map_hx、UpdaterMSCKF 的 Hx_mapping 是同一套"列-状态记账"逻辑的延续。
  // 结果 M_a 是一个"行覆盖所有状态、列覆盖量测状态"的矩阵，后续用它和 S^{-1} 相乘得到 Kalman 增益 K。
  // Ma = P*H^⊤ (尺寸为 N_state × N_res) -> M^T = HP
  for (const auto &var : state->_variables) {
    // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
    Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<Type> meas_var = H_order[i];
      M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                       H.block(0, H_id[i], H.rows(), meas_var->size()).transpose();
    }
    M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
  }
  // 为什么没被观测的状态也会更新?
  // EKF 更新不是只更新观测方程中显式出现的变量，而是通过 covariance cross-correlation 更新整个相关状态

  //==========================================================
  //==========================================================
  // Get covariance of the involved terms
  // 从完整 covariance 中抽出 measurement 涉及变量对应的 covariance
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // Residual covariance S = H*Cov*H' + R
  // 计算新息协方差
  // 为什么只写入上三角？因为 S 是对称矩阵，写入上三角就够了，节省计算和内存。
  Eigen::MatrixXd S(R.rows(), R.rows());
  S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
  S.triangularView<Eigen::Upper>() += R; // 只将这个结果矩阵的上三角部分（Upper Triangle）写入 S
  // Eigen::MatrixXd S = H * P_small * H.transpose() + R;

  // Invert our S (should we use a more stable method here??)
  // 如果不需要求逆，可以直接用 K = llt.solve(M_a.transpose()).transpose(); 一步求取卡尔曼增益
  // 用 Cholesky 分解求逆，避免直接求逆带来的数值不稳定性。S 是对称正定矩阵，适合用 Cholesky 分解。
  // llt() 是 Eigen 提供的 Cholesky 分解方法，S = LL^T (S是对称正定矩阵)
  // solveInPlace() 求解S*X = I (即对单位阵逐列求解)，结果X = S^-1直接写回Sinv。是在原地求解线性方程组，避免了额外的内存开销。
  // 好处：稳定、快，且是原地操作，不额外分配大矩阵
  Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
  S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
  // Sinv是对称矩阵，仍然用 selfadjointView<Eigen::Upper>() 来读取上半部分
  Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>(); // 组装卡尔曼增益 
  // Eigen::MatrixXd K = M_a * S.inverse();

  // Update Covariance
  // 更新协方差矩阵 P ,标准形式 P+ = (I - K*H)*P'
  // Joseph 稳定形式（对称保持形式） P+ = (I - K*H)*P'*(I - K*H)^T + K*R*K^T
  // 本质上是把(I-KH)平方了一下，再加上测量噪声项，确保了数值稳定性和对称性，但是计算量是标准形式的3倍
  // 还有一种方法，平方根/因式分解更新(数值黄金标准)，不直接更新P矩阵，而是更新协方差的 Cholesky 因子(即下三角矩阵L，满足P = LL^T)，
  // 这种方法始终保持 P 为正定矩阵，甚至能处理条件数极差的病态矩阵，数值稳定性最好，但实现复杂，计算量也大。
  // 本文用的方法是简化形式的EKF协方差更新，在增益取最优时与 Joseph 形式等价，数值稳定性略差于Joseph但计算量小很多。
  // 它的核心是利用了 M_a = P*H^T，S = H*P*H^T + R，最终更新公式可以简化为 P+ = P - K*M_a^T。
  //  KM^T = M*S^{-1}*M^T = PH^T*S^{-1}*H*P = KSK^T
  // 由于Ma = PH^T, 则Ma^T = HP, 所以 K*M_a^T = K*HP 
  // 因此 P = P - KHP = P - PH^T(HPH^T+R)^{-1}HP = P - PH^TS^{-1}HP = P - KSK^T
  state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>(); // 同样只保留上三角，保证对称性
  // Cov -= K * M_a.transpose();
  // Cov = 0.5*(Cov+Cov.transpose());

  // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
  Eigen::VectorXd diags = state->_Cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.rows(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET, i, diags(i));
      found_neg = true;
    }
  }
  if (found_neg) {
    std::exit(EXIT_FAILURE);
  }

  // Calculate our delta and update all our active states
  // dx 的布局和协方差/状态向量完全一致：每个状态变量在 dx 中占据 [id, id+size)
  Eigen::VectorXd dx = K * res; // 计算误差状态修正量
  for (size_t i = 0; i < state->_variables.size(); i++) {
    // 更新每个状态变量的值，dx.block(...) 提取出对应状态变量的误差修正量
    // 每个变量的 update() 会按自身类型处理
    state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
  }

  // If we are doing online intrinsic calibration we should update our camera objects
  // NOTE: is this the best place to put this update logic??? probably..
  if (state->_options.do_calib_camera_intrinsics) {
    for (auto const &calib : state->_cam_intrinsics) {
      state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
    }
  }
}

/**
 * @brief 把“初始化器给出的协方差子块”写回到全局状态协方差里，并保持矩阵对称
 * 
 * @param state 全局滤波器状态
 * @param covariance 初始协方差矩阵
 * @param order 协方差矩阵中各变量的顺序
 */
void StateHelper::set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                         const std::vector<std::shared_ptr<ov_type::Type>> &order) {

  // We need to loop through each element and overwrite the current covariance values
  // For example consider the following:
  // x = [ ori pos ] -> insert into -> x = [ ori bias pos ]
  // P = [ P_oo P_op ] -> P = [ P_oo  0   P_op ]
  //     [ P_po P_pp ]        [  0    P*    0  ]
  //                          [ P_po  0   P_pp ]
  // The key assumption here is that the covariance is block diagonal (cross-terms zero with P* can be dense)
  // This is normally the care on startup (for example between calibration and the initial state

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  // 它不是“重建整个协方差”，而是“把你指定变量集合的协方差块覆盖进去”。
  // 未在 order 里的其他变量块保持原值不动

  /*  假设状态向量为 x = [q, p, v, bg, ba, dw, da, dt]，但初始化器只估计了 order = {imu(q, p, v, bg, ba)}
      全局协方差（初始化前）:
          q  p  v  bg ba dw da dt
      q  [0  0  0  0  0  0  0  0]
      p  [0  0  0  0  0  0  0  0]
      v  [0  0  0  0  0  0  0  0]
      bg [0  0  0  0  0  0  0  0]
      ba [0  0  0  0  0  0  0  0]
      dw [0  0  0  0  0  0  0  0]
      da [0  0  0  0  0  0  0  0]
      dt [0  0  0  0  0  0  0  0]

      初始化器给出的局部协方差（order = {q, p, v, bg, ba}）:
           q    p    v    bg    ba
      q  [Pqq  Pqp  Pqv  Pqbg  Pqba]
      p  [Ppq  Ppp  Ppv  Ppbg  Ppba]
      v  [Pvq  Pvp  Pvv  Pvbg  Pvba]
      bg [Pbgq Pbgp Pbgv Pbgbg Pbgba]
      ba [Pbqa Pbap Pbav Pbabg Pgaba]

      覆盖后的全局协方差:
           q    p    v    bg    ba  dw da dt
      q  [Pqq  Pqp  Pqv  Pqbg  Pqba  0  0  0]
      p  [Ppq  Ppp  Ppv  Ppbg  Ppba  0  0  0]
      v  [Pvq  Pvp  Pvv  Pvbg  Pvba  0  0  0]
      bg [Pbgq Pbgp Pbgv Pbgbg Pbgba 0  0  0]
      ba [Pbqa Pbap Pbav Pbabg Pbaba 0  0  0]
      dw [ 0    0    0     0     0   0  0  0]   ← 不动
      da [ 0    0    0     0     0   0  0  0]   ← 不动
      dt [ 0    0    0     0     0   0  0  0]   ← 不动 
  */
  int i_index = 0;
  for (size_t i = 0; i < order.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < order.size(); k++) {
      // order[i]->id() 是变量在全局状态向量中的起始位置，i_index 是其在局部协方差中的起始位置
      state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
          covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
      k_index += order[k]->size();
    }
    i_index += order[i]->size();
  }
  // 用上三角自动镜像出下三角，得到严格对称矩阵（避免后续 LLT/SVD 等算法因为微小不对称而数值不稳）
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
}

/**
 * @brief 从全局协方差里“抠”出指定状态子集的协方差子块
 * 主要用在两个地方:
 * 1. EKF 更新的新息协方差 S = H*P_small*H' + R 因为量测只跟 H_order 里的那部分状态有关，不需要拿全协方差去算
 * 2. 更新前的卡方检验 chi2 = r^T*(Hz*P_marg*Hz^T + σ^2*I)^{-1}*r 每帧有几百上千个特征，
 * 每个特征都拿全协方差做一次S^-1是不现实的，所以只取它涉及的那几个状态
 * 原理：为什么“子块就是边缘协方差”?
 * 对高斯分布，边缘化掉其他变量后，剩余变量的协方差恰好就是联合协方差中对应的子块（这是高斯分布的性质，不需要 Schur 补）:
 * x = [x1, x2]^T ~ N([μ1, μ2]^T, [[P11, P12], [P21, P22]])
 * 则 x1 ~ N(μ1, P11)，x2 ~ N(μ2, P22)，即边缘协方差就是联合协方差的子块
 * 所以这个函数不需要做任何“真正的边缘化计算”，直接按块拷贝 state->_Cov 里对应位置即可。
 * 注意： 
 * 1. 它不是“条件化/真正边缘化状态”：它不改变 state->_Cov、不删除任何变量，只是返回一个拷贝，是只读操作。
 * 真正删状态用的是 marginalize（那是做块删除，也仍然不做 Schur 补，MSCKF 里旧的克隆信息不再需要）。
 * 2. 对称性：全局 state->_Cov 始终被维护为对称矩阵（各写回函数最后都 selfadjointView<Upper>() 镜像对称），所以抠出来的子块理论上也是对称的。
 * 代码末尾那行被注释掉的 // Small_cov = 0.5*(Small_cov+Small_cov.transpose()); 只是防御性的保险，平时不开启
 * 3. 性能：双循环是O(N^2) 次块拷贝，但每个特征只涉及 2~3 个状态，块都很小，开销可忽略；对比直接拿全协方差做 S^-1 要省得多
 * @param state 当前滤波器状态
 * @param small_variables 指定的小状态变量列表
 * @return Eigen::MatrixXd 对应的小状态变量的边缘协方差矩阵（含所有交叉项）
 */
Eigen::MatrixXd StateHelper::get_marginal_covariance(std::shared_ptr<State> state,
                                                     const std::vector<std::shared_ptr<Type>> &small_variables) {

  // Calculate the marginal covariance size we need to make our matrix
  int cov_size = 0; // 返回矩阵的维数
  for (size_t i = 0; i < small_variables.size(); i++) {
    cov_size += small_variables[i]->size();
  }

  // Construct our return covariance
  Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < small_variables.size(); i++) { // 行方向：小矩阵的第 i 个块
    int k_index = 0;
    for (size_t k = 0; k < small_variables.size(); k++) { // 列方向：小矩阵的第 k 个块
      Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
          // 全局协方差的起始行列
          state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
      k_index += small_variables[k]->size();
    }
    i_index += small_variables[i]->size();
  }

  // Return the covariance
  // Small_cov = 0.5*(Small_cov+Small_cov.transpose());
  return Small_cov;
}

Eigen::MatrixXd StateHelper::get_full_covariance(std::shared_ptr<State> state) {

  // Size of the covariance is the active
  int cov_size = (int)state->_Cov.rows();

  // Construct our return covariance
  Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // Copy in the active state elements
  full_cov.block(0, 0, state->_Cov.rows(), state->_Cov.rows()) = state->_Cov;

  // Return the covariance
  return full_cov;
}

/**
 * @brief 边缘化指定的状态变量
 * 从协方差矩阵和状态变量列表中删除指定变量对应的行和列，并正确更新其他变量的 id 索引
 * 假设状态向量分为三部分: x1（被删变量之前的）、xm（要删的）、x2（被删变量之后的）
 * 协方差矩阵 P 的形式为：P = [ P11  P1m  P12 ]
 *                         [ Pm1  Pmm  Pm2 ]
 *                         [ P21  P2m  P22 ]
 * 边缘化 xm 后，新的协方差矩阵 P_new 为：
 * P’ = [ P11  P12 ]
 *         [ P21  P22 ]
 * 其中 P11、P12、P21、P22 分别对应 x1 和 x2 的协方差块，P1m、Pm1、Pm2、P2m 被删除。
 * 这里没有做 Schur complement 之类的操作——OpenVINS 的 MSCKF 实现中，边缘化克隆就是直接把该克隆的行列删掉。
 * 这不损失信息，因为后续不再需要这些克隆来约束其他状态。
 * @param state 当前滤波器状态
 * @param marg 要边缘化的状态变量 _clones_IMU 中保存的 imu pose变量
 */
void StateHelper::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg) {

  // Check if the current state has the element we want to marginalize
  // 首先检查是否在滤波器的状态向量列表中找到要边缘化的变量
  if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
    PRINT_ERROR(RED "StateHelper::marginalize() - Called on variable that is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Generic covariance has this form for x_1, x_m, x_2. If we want to remove x_m:
  //
  //  P_(x_1,x_1) P(x_1,x_m) P(x_1,x_2)
  //  P_(x_m,x_1) P(x_m,x_m) P(x_m,x_2)
  //  P_(x_2,x_1) P(x_2,x_m) P(x_2,x_2)
  //
  //  to
  //
  //  P_(x_1,x_1) P(x_1,x_2)
  //  P_(x_2,x_1) P(x_2,x_2)
  //
  // i.e. x_1 goes from 0 to marg_id, x_2 goes from marg_id+marg_size to Cov.rows() in the original covariance

  int marg_size = marg->size(); // 被删状态变量的维度
  int marg_id = marg->id();     // 被删状态变量在协方差中的起始位置
  int x2_size = (int)state->_Cov.rows() - marg_id - marg_size; // 被删变量之后 变量x2(严格来说是剩余变量)的维度
  // 边缘化之后的协方差矩阵维度为 原来的rows - marg_size
  Eigen::MatrixXd Cov_new(state->_Cov.rows() - marg_size, state->_Cov.rows() - marg_size);
  // 更新新的协方差矩阵的各个块
  // P_(x_1,x_1)
  Cov_new.block(0, 0, marg_id, marg_id) = state->_Cov.block(0, 0, marg_id, marg_id);

  // P_(x_1,x_2)
  Cov_new.block(0, marg_id, marg_id, x2_size) = state->_Cov.block(0, marg_id + marg_size, marg_id, x2_size);

  // P_(x_2,x_1)
  Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();

  // P(x_2,x_2)
  Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state->_Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);

  // Now set new covariance
  // state->_Cov.resize(Cov_new.rows(),Cov_new.cols());
  state->_Cov = Cov_new;
  // state->Cov() = 0.5*(Cov_new+Cov_new.transpose());
  assert(state->_Cov.rows() == Cov_new.rows());

  // Now we keep the remaining variables and update their ordering
  // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
  std::vector<std::shared_ptr<Type>> remaining_variables;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    // Only keep non-marginal states
    if (state->_variables.at(i) != marg) {
      if (state->_variables.at(i)->id() > marg_id) {
        // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
        // 边缘化过后，处于被删变量之后的所有变量的 id(在协方差矩阵中的位置) 都需要向前移动 marg_size
        state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
      }
      remaining_variables.push_back(state->_variables.at(i));
    }
  }

  // Delete the old state variable to free up its memory
  // NOTE: we don't need to do this any more since our variable is a shared ptr
  // NOTE: thus this is automatically managed, but this allows outside references to keep the old variable
  // delete marg;
  // 将被删变量的 id 设为 -1，表示它已不在状态向量中，防止后续误用
  marg->set_local_id(-1);

  // Now set variables as the remaining ones
  state->_variables = remaining_variables;
}

/**
 * @brief 这个函数接收一个现有状态变量（如 IMU 位姿），完整复制一份追加到状态向量末尾，并正确填充新变量与所有其他变量的协方差
 * 
 * @param state 误差卡尔曼滤波状态
 * @param variable_to_clone 要克隆的状态变量(单一变量或子变量)
 * @return std::shared_ptr<Type> 
 */
std::shared_ptr<Type> StateHelper::clone(std::shared_ptr<State> state, std::shared_ptr<Type> variable_to_clone) {

  // Get total size of new cloned variables, and the old covariance size
  int total_size = variable_to_clone->size(); // 要克隆的状态变量的维度
  int old_size = (int)state->_Cov.rows(); // 克隆前的全局状态协方差矩阵的维度
  int new_loc = (int)state->_Cov.rows();  // 克隆后新变量在全局状态向量中的起始位置（即原协方差矩阵的末尾）

  // Resize both our covariance to the new size,扩充协方差矩阵
  // 协方差从 old_size × old_size 扩展为 (old_size+total_size) × (old_size+total_size)，新区域全部填充 0
  // conservativeResizeLike 的核心功能是将当前矩阵或数组的大小，调整为与另一个矩阵或数组（other）完全相同
  // 函数名中的“保守”（Conservative）是其行为的关键，主要体现在数据保留上: 
  // 1. 保留左上角数据：调整大小时，原矩阵/数组中位于“左上角”的数据会被尽可能地保留下来
  // 2. 扩大 (New Size > Old Size)：原有数据保留在左上角，但新增加的行和列的元素是未初始化的，其值是未定义的（“垃圾值”）!!!!
  // 3. 缩小 (New Size < Old Size)：矩阵/数组会被截断，只保留左上角的部分数据，超出新尺寸范围的数据将丢失
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));

  // What is the new state, and variable we inserted
  const std::vector<std::shared_ptr<Type>> new_variables = state->_variables; // not used
  std::shared_ptr<Type> new_clone = nullptr;

  // Loop through all variables, and find the variable that we are going to clone
  // 在状态变量列表中定位当前被克隆变量
  for (size_t k = 0; k < state->_variables.size(); k++) {

    // Skip this if it is not the same
    // First check if the top level variable is the same, then check the sub-variables
    // 从当前滤波器状态中查找匹配的克隆状态变量（支持子变量查找）
    // check_if_subvariable 会返回一个指向匹配子变量的 shared_ptr，如果没有匹配则返回 nullptr
    // 这一句没有起到什么作用啊
    std::shared_ptr<Type> type_check = state->_variables.at(k)->check_if_subvariable(variable_to_clone);
    if (state->_variables.at(k) == variable_to_clone) {
      type_check = state->_variables.at(k);
    } else if (type_check != variable_to_clone) {
      continue;
    }

    // So we will clone this one // 被克隆变量在协方差中的位置
    int old_loc = type_check->id();

    // Copy the covariance elements
    // 复制三个块: 自有协方差块、新克隆变量与其他变量的交叉协方差块(行和列)
    state->_Cov.block(new_loc, new_loc, total_size, total_size) = state->_Cov.block(old_loc, old_loc, total_size, total_size);
    state->_Cov.block(0, new_loc, old_size, total_size) = state->_Cov.block(0, old_loc, old_size, total_size);
    state->_Cov.block(new_loc, 0, total_size, old_size) = state->_Cov.block(old_loc, 0, total_size, old_size);

    // Create clone from the type being cloned
    new_clone = type_check->clone();  // 深拷贝
    new_clone->set_local_id(new_loc); // 设置新克隆变量在协方差矩阵中的起始位置
    break;
  }

  // Check if the current state has this variable
  if (new_clone == nullptr) {
    PRINT_ERROR(RED "StateHelper::clone() - Called on variable is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Add to variable list and return
  state->_variables.push_back(new_clone);
  return new_clone;
}

/**
 * @brief Initialize a new variable in the state with a given measurement model and noise
 * 给定一批观测到"新路标"的量测，用 Givens 正交旋转把线性系统一分为二： 
 * 上半部分用来初始化路标（均值 + 协方差 + 与已有状态的交叉项），下半部分用来对已有状态做一次 EKF 更新
 * 线性化后的初始测量模型为：
 * r = H_R * x + H_L * f + n, n ~ N(0, R)
 * 其中 r 是残差，x 是已有状态，f 是新路标，n 是测量噪声，R 是噪声协方差矩阵
 * 我们需要做两件事情：
 * 1.求出新路标的最优估计f^及其协方差P_ff和与已有状态的交叉协方差P_xf
 * 2.利用不依赖路标的那部分信息顺便更新已有状态
 * 难点:
 * H_L是高瘦矩阵(2mx3)，不能直接求逆，所以先用正交旋转把它变为"上半可逆，下半为零"
 * 通过 Givens QR 分解把 H_f 变为上三角
 * 1. 上半部分: H_finit * f + H_xinit * x = rinit, 用来初始化新路标 f
 * 2. 下半部分: H_up * x = r_up, 用来对已有状态 x 做一次 EKF 更新
 * @param state 
 * @param new_variable landmark
 * @param H_order 与该new_variable相关的状态变量列表，顺序与 H_R/H_L 中的列对应
 * @param H_R H_x 观测相对相关状态变量的雅可比矩阵
 * @param H_L H_f 观测相对路标变量的雅可比矩阵
 * @param R 噪声协方差矩阵
 * @param res 残差向量
 * @param chi_2_mult 卡方检测阈值倍数，通常取 1.0~3.0，越大越宽松
 * @return true 
 * @return false 
 */
bool StateHelper::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                             const std::vector<std::shared_ptr<Type>> &H_order, Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L,
                             Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {

  // Check that this new variable is not already initialized
  // 该路标已经在状态向量中存在，说明重复初始化了
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  // 噪声必须是各向同性（对角线同值、非对角线为零）
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // First we perform QR givens to seperate the system
  // The top will be a system that depends on the new state, while the bottom does not
  size_t new_var_size = new_variable->size(); // 路标的维度(三维或者一维)
  assert((int)new_var_size == H_L.cols()); // 检查观测对路标的雅可比矩阵列数是否与路标维度一致

  // 这里实质上构造了一个全局正交旋转矩阵 Q(由一系列 Givens 旋转连乘而成)，使得: 
  // Q[H_L; H_R; res] = [R_L, H_R,1, res1] 
  //                    [0,   H_R,2, res2]，
  // 其中 R_L 是上三角矩阵，H_R,1/H_R,2 分别是上半部分/下半部分的 H_R
  // 也就是把原来的线性系统分为两部分：上半部分用来初始化新路标，下半部分用来对已有状态做一次 EKF 更新
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_L.cols(); ++n) {
    for (int m = (int)H_L.rows() - 1; m > n; m--) { // 自下而上，逐行消元
      // Givens matrix G
      //计算一个 2x2 的 Givens 正交旋转矩阵，将向量[H_L(m - 1, n), H_L(m, n)]中的 H_L(m, n) 被消为 0
      tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      // 同步行变换(左乘G^T)
      (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // Separate into initializing and updating portions
  // 这里将观测拆成两个子系统
  // 原方程 r = Hx * δx + Hf * δf + n，分解后变为：
  // [r_init] = [H_xinit] * δx + [H_finit] * δf + n_init
  // [r_up  ] = [H_up   ]      + [0      ]      + n_up
  // 1. Invertible initializing system
  // 上半部（含路标，可逆）→ 用于初始化路标
  Eigen::MatrixXd Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
  Eigen::MatrixXd H_finit = H_L.block(0, 0, new_var_size, new_var_size);
  Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
  Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

  // 2. Nullspace projected updating system
  // 下半部（无路标）→ 用于更新已有状态
  Eigen::MatrixXd Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
  Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
  Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

  //==========================================================
  //==========================================================

  // Do mahalanobis distance testing
  // 对下半部残差做卡方检验，防止异常值影响已有状态的更新
  Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order);
  assert(Rup.rows() == Hup.rows());
  assert(Hup.cols() == P_up.cols());
  Eigen::MatrixXd S = Hup * P_up * Hup.transpose() + Rup;
  double chi2 = resup.dot(S.llt().solve(resup));

  // Get what our threshold should be
  // 用不依赖路标的下半部做检验：如果这部分残差都大到离谱，说明该特征初始化不可靠，直接拒绝
  boost::math::chi_squared chi_squared_dist(res.rows());
  double chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
  if (chi2 > chi_2_mult * chi2_check) {
    return false;
  }

  //==========================================================
  //==========================================================
  // Finally, initialize it in our state 用上半部分初始化路标
  StateHelper::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit);

  // Update with updating portion
  // initialization 后为什么还有一次 EKF update？
  // 前面经过QR之后 r 分为上半部分的r_init(这部分拿来去初始化路标点了)与下半部分的r_up(完全不依赖于路标)
  // 所以剩余的r_up也不能浪费，拿来继续更新我们的state,与msckf思想一致
  if (Hup.rows() > 0) {
    StateHelper::EKFUpdate(state, H_order, Hup, resup, Rup);  // 用下半部分（不依赖路标）对已有状态做一次标准更新
  }
  return true;
}

/**
 * @brief Initialize a new variable in the state using an invertible measurement model
 * 现在初始化部分是:
 * r_init = Hx*δx + Hf *δf + n 
 * 由于经过QR后  Hf ∈ R_3x3，并且假设可逆
 * 于是 δf = Hf^{-1}(r_init - Hx*δx -n)
 * 也就是 δf = -Hf^{-1}*Hx*δx + Hf^{-1}*r_init - Hf^{-1}*n
 * 可以将这个公式分为两部分:
 * 1. Hf^{-1}*r_init 确定性部分，类似于均值
 * 2. -Hf^{-1}*Hx*δx - Hf^{-1}*n  随机部分，类似于误差
 * EKF随机变量传播里是均值和协方差分开处理，所以
 * 协方差计算:
 * δf = -Hf^{-1}*Hx*δx - Hf^{-1}*n
 * 令A = -Hf^{-1}*Hx, B = -Hf^{-1}, 则 δf = A*δx + B*n
 * 于是 P_ff = A*P_xx*A^T + B*R*B^T, P_xf = P_xx*A^T
 * 其中 P_xx 是已有状态的协方差，R 是测量噪声协方差
 * 所以 Pff = Hf^{-1}*Hx*P_xx*Hx^T*Hf^{-T} + Hf^{-1}*R*Hf^{-T}, P_xf = -P_xx*Hx^T*Hf^{-T}
 * 也就是新路标的协方差和已有状态的交叉协方差
 * @param state The current state
 * @param new_variable The new variable to be initialized 路标
 * @param H_order The order of variables in the measurement
 * @param H_R The right part of the measurement matrix after QR decomposition 上半部分H_xinit
 * @param H_L The left part of the measurement matrix after QR decomposition 上半部分H_finit
 * @param R The measurement noise covariance 上半部分噪声
 * @param res The measurement residual 上半部分残差
 */
void StateHelper::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                        const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H_R,
                                        const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {

  // Check that this new variable is not already initialized
  // 重复初始化检查
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // 各向同性噪声检查（对角线同值、非对角为零）
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  assert(res.rows() == R.rows());
  assert(H_L.rows() == res.rows());
  assert(H_L.rows() == H_R.rows());
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id; // H_id: H_order → H_R 中列偏移
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  // M的行覆盖所有状态、列只涉及量测状态，按"状态块 × 量测状态块"稀疏累加
  for (const auto &var : state->_variables) {
    // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
    Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
    for (size_t i = 0; i < H_order.size(); i++) {
      std::shared_ptr<Type> meas_var = H_order.at(i);
      M_i += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
             H_R.block(0, H_id[i], H_R.rows(), meas_var->size()).transpose();
    }
    M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
  }

  //==========================================================
  //==========================================================
  // Get covariance of this small jacobian
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // M = H_R*Cov*H_R' + R
  Eigen::MatrixXd M(H_R.rows(), H_R.rows());
  M.triangularView<Eigen::Upper>() = H_R * P_small * H_R.transpose();
  M.triangularView<Eigen::Upper>() += R;

  // Covariance of the variable/landmark that will be initialized
  // 计算路标点的自有协方差 P_LL = H_L^{-1} * M * H_L^{-T}
  // 为什么计算 landmark covariance 时忽略 residual 的 deterministic part？
  // 因为协方差描述的是: 随机变量相对于均值的波动，而不是均值本身。
  // 定义上 Pff = E[(δf - E[δf])(δf - E[δf])^T]，其中 E[δf] 是均值，(δf - E[δf]) 是零均值的随机变量
  // 现在 δf = -H_L^{-1}*H_R*δx + H_L^{-1}*r - H_L^{-1}*n，
  // 当前这次计算中残差 r 已经是一个确定的数值，他不是随机变量，所以H_L^{-1}*r 是一个确定量，记 c = H_L^{-1}*r
  // 而随机部分记成 w = -H_L^{-1}*H_R*δx - H_L^{-1}*n，w是零均值的随机变量，E[w] = 0
  // 那么 δf = c + w，又 E[w] = 0 则E[δf] = c + E[w] = c，所以 δf - E[δf] = c +w -c = w
  // 所以Cov[δf] = Cov(w) = E[w w^T]，即Pff = H_L^{-1}*H_R*Pxx*H_R^T*H_L^{-T} + H_L^{-1}*R*H_L^{-T}
  assert(H_L.rows() == H_L.cols()); // H_L 必须是方阵（且大小为路标维度 × 路标维度）
  assert(H_L.rows() == new_variable->size());
  Eigen::MatrixXd H_Linv = H_L.inverse();
  Eigen::MatrixXd P_LL = H_Linv * M.selfadjointView<Eigen::Upper>() * H_Linv.transpose();

  // Augment the covariance matrix
  // 扩增协方差矩阵，增加新路标的协方差块和与已有状态的交叉协方差块
  // Pnew = [ Pxx  Pxf ]
  //        [ Pfx  Pff ]
  // 这里为什么不能写成 Pnew = [Pxx  0]
  //                        [0    Pff] 呢？
  // 因为这样写的话就相当于告诉EKF滤波器新路标与已有camera pose状态是独立的，但是实际上landmark是由这些相机位姿三角化得到的。
  // 所以新路标的协方差 Pff 与已有状态的交叉协方差 Pxf 不能为零，否则滤波器无法利用已有状态对新路标进行约束，导致初始化不准确。
  // delayed initialization 最关键的并不是“三角化晚一点”，而是正确做 correlated state augmentation
  size_t oldSize = state->_Cov.rows();
  state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize + new_variable->size(), oldSize + new_variable->size()));
  state->_Cov.block(0, oldSize, oldSize, new_variable->size()).noalias() = -M_a * H_Linv.transpose();
  state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
  state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_LL;

  // Update the variable that will be initialized (invertible systems can only update the new variable).
  // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
  // 更新新路标的状态值，使用 H_L 的逆乘以残差向量 res 得到最优估计增量
  // δf = H_L^{-1} * res  → 这里计算的是条件均值 δf^ = E[δf] = H_L^{-1} * res
  // 在EKF当前线性化点，E[δx] = 0, E[n] = 0, 所以只有res留下来
  // 但是如果是真实的landmark error f^~,那就绝对不止跟r有关了，而是
  // f^~ = Hf^{-1}*r -Hf^{-1}*Hx*x^~ - Hf^{-1}*n
  // 而初始化完成、把 mean correction 吸收到 nominal state 以后，新的 zero-mean error 实际上变成
  // f^~+ = -Hf^{-1}*Hx*x^~ - Hf^{-1}*n
  // 所以它和 camera/IMU state uncertainty 的关系，完整地保存在 Pxf 里了
  new_variable->update(H_Linv * res);

  // Now collect results, and add it to the state variables
  new_variable->set_local_id(oldSize);
  state->_variables.push_back(new_variable);

  // std::stringstream ss;
  // ss << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;
  // PRINT_DEBUG(ss.str().c_str());
}

/**
 * @brief 状态克隆（clone）是指在滤波器中添加一个新的状态变量，该变量的初始值和协方差与现有状态变量相同。
 * 这个函数实现了状态克隆的过程，并且在时间校准的情况下，还会根据IMU的角速度和线速度来调整协方差矩阵，以反映时间偏移对状态估计的不确定性影响。
 * 
 * @param state 误差滤波器状态
 * @param last_w IMU角速度
 */
void StateHelper::augment_clone(std::shared_ptr<State> state, Eigen::Matrix<double, 3, 1> last_w) {

  // We can't insert a clone that occured at the same timestamp!
  // state->_timestamp 表明当前滤波器已经传播到了这个时间点，
  // 如果在这个时间点已经有一个克隆存在，那么就会导致状态冲突，因此需要检查是否已经存在该克隆。
  if (state->_clones_IMU.find(state->_timestamp) != state->_clones_IMU.end()) {
    PRINT_ERROR(RED "TRIED TO INSERT A CLONE AT THE SAME TIME AS AN EXISTING CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our cloner and add it to our vector of types
  // NOTE: this will clone the clone pose to the END of the covariance...
  // 克隆imu位姿: 并将其添加到状态变量列表中，同时扩展协方差矩阵以包含新克隆的状态变量。
  std::shared_ptr<Type> posetemp = StateHelper::clone(state, state->_imu->pose());

  // Cast to a JPL pose type, check if valid
  std::shared_ptr<PoseJPL> pose = std::dynamic_pointer_cast<PoseJPL>(posetemp);
  if (pose == nullptr) {
    PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Append the new clone to our clone vector， 克隆IMU位姿，即在滤波器状态中添加新的克隆IMU状态
  state->_clones_IMU[state->_timestamp] = pose;

  // If we are doing time calibration, then our clones are a function of the time offset
  // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
  // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
  if (state->_options.do_calib_camera_timeoffset) {
    // Jacobian to augment by
    Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
    dnc_dt.block(0, 0, 3, 1) = last_w;
    dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    // Augment covariance with time offset Jacobian
    // TODO: replace this with a call to the EKFPropagate function instead....
    state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
        state->_Cov.block(0, state->_calib_dt_CAMtoIMU->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
    state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
        dnc_dt * state->_Cov.block(state->_calib_dt_CAMtoIMU->id(), 0, 1, state->_Cov.rows());
  }
}

/**
 * @brief 边缘化最旧的克隆IMU状态
 * 该函数一般每次 EKF 更新后被调用
 * @param state 当前滤波器状态
 */
void StateHelper::marginalize_old_clone(std::shared_ptr<State> state) {
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    double marginal_time = state->margtimestep();
    // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
    // 这个可能会在特征跟踪等线程中被加锁
    std::lock_guard<std::mutex> lock(state->_mutex_state);
    assert(marginal_time != INFINITY);
    StateHelper::marginalize(state, state->_clones_IMU.at(marginal_time));
    // Note that the marginalizer should have already deleted the clone
    // Thus we just need to remove the pointer to it from our state
    // 从误差滤波器状态中的克隆map中删除最旧的克隆IMU状态
    state->_clones_IMU.erase(marginal_time);
  }
}

/**
 * @brief 边缘化标记为 should_marg 的 普通SLAM 路标
 *
 * @param state 当前滤波器状态
 */
void StateHelper::marginalize_slam(std::shared_ptr<State> state) {
  // Remove SLAM features that have their marginalization flag set
  // We also check that we do not remove any aruoctag landmarks
  int ct_marginalized = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    // (*it0).first 是特征的 ID，这里只边缘化普通SLAM特征(aruco受到保护)
    if ((*it0).second->should_marg && (int)(*it0).first > 4 * state->_options.max_aruco_features) {
      StateHelper::marginalize(state, (*it0).second);
      it0 = state->_features_SLAM.erase(it0);
      ct_marginalized++;
    } else {
      it0++;
    }
  }
}