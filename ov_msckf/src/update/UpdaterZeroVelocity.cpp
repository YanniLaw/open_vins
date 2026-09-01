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

#include "UpdaterZeroVelocity.h"

#include "UpdaterHelper.h"

#include "feat/FeatureDatabase.h"
#include "feat/FeatureHelper.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;
/*
OpenVINS 里的零速更新（Zero Velocity Update, ZUPT）和很多惯导系统里“直接把速度观测设成 0”的 ZUPT 不太一样。
OpenVINS 当前默认实现的核心其实更接近“静止/恒速检测 + IMU 伪测量更新”：
利用“静止时真实角速度为 0、真实线加速度只剩重力”这个事实，对姿态、陀螺 bias、加速度计 bias 做 EKF 更新；
然后再通过当前速度大小和图像视差避免把匀速运动误判为静止。
官方文档自己也特别强调：这并不是严格意义上直接约束v=0，而首先是在约束“zero acceleration / zero angular velocity”
文档参考https://docs.openvins.com/update-zerovelocity.html

OpenVINS 为什么需要 ZUPT? 
假设机器人停在那里，对于单目VIO：
相机没有运动；特征点没有视差；三角化退化；MSCKF 更新的信息量迅速下降；但是 IMU 仍然不断积分；很小的 b_g,b_a 误差仍会继续导致速度和姿态漂移。
更麻烦的是，机器人虽然静止，但环境可能在动。例如汽车停在路口，周围车辆经过。此时视觉更新甚至可能产生错误运动估计。
所以 OpenVINS 的思路是：
一旦确认平台静止，就不要依赖视觉视差去估计运动，而是直接利用 IMU 的静止物理模型给 EKF 强约束。

OPENVINS的IMU模型（参考https://docs.openvins.com/propagation.html）
陀螺仪 wm = w + b_g + n_g
加速度计 am = a + R_ItoG* g + b_a + n_a
其中 w 是真实角速度，a 是真实线加速度，g 是重力加速度，b_g,b_a 是陀螺和加速度计的偏置，
n_g,n_a 是陀螺和加速度计的高斯噪声。
假设平台静止，则 w=0, a=0
所以有:
a = am - R_ItoG* g - b_a - n_a = 0
w = wm - b_g - n_g = 0
构建如下残差(subtract the synthetic "measurement" and our measurement function):
r = [a - (am - R_ItoG* g - b_a - n_a)] = [- (am - b_a - R_ItoG* g - n_a)]
    [w - (wm - b_g - n_g)]             = [- (wm - b_g - n_g)]

于是 得到雅可比矩阵
∂r/∂R_ItoG = -[R_ItoG*gx]
∂r/∂b_a = ∂r/∂b_g = -I_3x3
*/

UpdaterZeroVelocity::UpdaterZeroVelocity(UpdaterOptions &options, NoiseManager &noises, std::shared_ptr<ov_core::FeatureDatabase> db,
                                         std::shared_ptr<Propagator> prop, double gravity_mag, double zupt_max_velocity,
                                         double zupt_noise_multiplier, double zupt_max_disparity)
    : _options(options), _noises(noises), _db(db), _prop(prop), _zupt_max_velocity(zupt_max_velocity),
      _zupt_noise_multiplier(zupt_noise_multiplier), _zupt_max_disparity(zupt_max_disparity) {

  // Gravity
  _gravity << 0.0, 0.0, gravity_mag;

  // Save our raw pixel noise squared
  _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
  _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
  _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
  _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  // 为什么这里表的上限是 1000，而 MSCKF/SLAM 的一些表上限是 500？
  // 因为 ZUPT 可能积累一段时间内的很多 IMU 样本；每个相邻 IMU 时间段贡献 6 维约束：
  for (int i = 1; i < 1000; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

/**
 * @brief Try to update the state using zero-velocity constraints
 * 构造经过噪声白化的伪量测；再通过卡方距离、当前速度和图像视差确认静止；
 * 一旦通过，就传播偏置随机游走并调用 EKFUpdate 校正状态。
 * 默认模式下，它更像是“静止 IMU 校准器”：利用静止期修正姿态和零偏，抑制惯导漂移，同时跳过缺乏视差价值的普通视觉更新。
 * 步骤:
 * 1. 在两个相机时刻之间收集 IMU；
 * 2. 用“静止时陀螺和加速度计应满足的物理规律”构造伪量测；
 * 3. 用卡方检验、速度阈值和图像视差判断是否真的静止；
 * 4. 若成立，用 EKF 校正姿态、IMU 零偏，以及通过协方差关联间接校正其他状态；
 * 5. 把状态时间推进到当前相机时刻，而不走普通的视觉更新/克隆流程。
 * @param state The current state of the system
 * @param timestamp The timestamp at which to apply the zero-velocity update 传入的相机的时间戳
 * @return true If the update was successfully applied
 * @return false If the update could not be applied
 */
bool UpdaterZeroVelocity::try_update(std::shared_ptr<State> state, double timestamp) {

  // Return if we don't have any imu data yet
  // 无 IMU缓存 数据，无法判断静止
  if (imu_data.empty()) {
    last_zupt_state_timestamp = 0.0;
    return false;
  }

  // Return if the state is already at the desired time
  // 当前状态已经在目标时刻，无需再处理
  if (state->_timestamp == timestamp) {
    last_zupt_state_timestamp = 0.0;
    return false;
  }

  // Set the last time offset value if we have just started the system up
  if (!have_last_prop_time_offset) { // 刚启动系统，last_prop_time_offset 还没有值
    last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0); // 上一次传播的时间偏移量
    have_last_prop_time_offset = true;
  }

  // assert that the time we are requesting is in the future
  // assert(timestamp > state->_timestamp);

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_calib_dt_CAMtoIMU->value()(0); // 最新估计的 IMU-CAM 时间偏移量

  // First lets construct an IMU vector of measurements we need
  // double time0 = state->_timestamp+t_off_new;
  double time0 = state->_timestamp + last_prop_time_offset; // 区间起点使用上次传播时的偏移
  double time1 = timestamp + t_off_new; // 区间终点使用当前估计的偏移

  // Select bounding inertial measurements
  std::vector<ov_core::ImuData> imu_recent = Propagator::select_imu_readings(imu_data, time0, time1);

  // Move forward in time
  last_prop_time_offset = t_off_new;

  // Check that we have at least one measurement to propagate with
  if (imu_recent.size() < 2) {
    PRINT_WARNING(RED "[ZUPT]: There are no IMU data to check for zero velocity with!!\n" RESET);
    last_zupt_state_timestamp = 0.0;
    return false;
  }

  // If we should integrate the acceleration and say the velocity should be zero
  // Also if we should still inflate the bias based on their random walk noises
  // 正常运行时，通过原始 IMU 静止约束更新姿态和零偏，然后把时间推进到 timestamp
  bool integrated_accel_constraint = false; // untested 使用“静止比力/重力一致”约束，不直接积分加速度约束速度
  bool model_time_varying_bias = true;         // 接受 ZUPT 前，允许 IMU 零偏按随机游走继续增长协方差
  bool override_with_disparity_check = true;   // 图像视差足够小时，可覆盖 IMU 卡方/速度拒绝条件
  bool explicitly_enforce_zero_motion = false; // 是否显式加入“位置不变、姿态不变、速度为零”的 9 维硬约束

  // Order of our Jacobian
  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_imu->q());    // 姿态误差 3 维
  Hx_order.push_back(state->_imu->bg());   // 陀螺零偏 3 维
  Hx_order.push_back(state->_imu->ba());   // 加速度计零偏 3 维
  if (integrated_accel_constraint) {
    Hx_order.push_back(state->_imu->v());  // 速度误差 3 维  not used
  }

  // Large final matrices used for update (we will compress these)
  int h_size = (integrated_accel_constraint) ? 12 : 9;
  int m_size = 6 * ((int)imu_recent.size() - 1); // 每个相邻 IMU 时间间隔贡献 6 维观测约束
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m_size, h_size);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(m_size);

  // IMU intrinsic calibration estimates (static)
  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());

  // Loop through all our IMU and construct the residual and Jacobian
  // TODO: should add jacobians here in respect to IMU intrinsics!!
  // State order is: [q_GtoI, bg, ba, v_IinG]
  // Measurement order is: [w_true = 0, a_true = 0 or v_k+1 = 0]
  // w_true = w_m - bw - nw
  // a_true = a_m - ba - R*g - na
  // v_true = v_k - g*dt + R^T*(a_m - ba - na)*dt
  double dt_summed = 0;
  for (size_t i = 0; i < imu_recent.size() - 1; i++) {

    // Precomputed values
    double dt = imu_recent.at(i + 1).timestamp - imu_recent.at(i).timestamp;
    Eigen::Vector3d a_hat = state->_calib_imu_ACCtoIMU->Rot() * Da * (imu_recent.at(i).am - state->_imu->bias_a());
    Eigen::Vector3d w_hat = state->_calib_imu_GYROtoIMU->Rot() * Dw * (imu_recent.at(i).wm - state->_imu->bias_g() - Tg * a_hat);

    // Measurement noise (convert from continuous to discrete)
    // NOTE: The dt time might be different if we have "cut" any imu measurements
    // NOTE: We are performing "whittening" thus, we will decompose R_meas^-1 = L*L^t
    // NOTE: This is then multiplied to the residual and Jacobian (equivalent to just updating with R_meas)
    // NOTE: See Maybeck Stochastic Models, Estimation, and Control Vol. 1 Equations (7-21a)-(7-21c)
    double w_omega = std::sqrt(dt) / _noises.sigma_w;
    double w_accel = std::sqrt(dt) / _noises.sigma_a;
    double w_accel_v = 1.0 / (std::sqrt(dt) * _noises.sigma_a);

    // Measurement residual (true value is zero)
    // 代码中的残差 r = [r_omega, r_accel]
    // 其中 r_omega = - (wm - b_g - n_g)
    //     r_accel = - (am - b_a - R_ItoG* g - n_a)
    // 静止时，w_true = 0, a_true = 0, 所以残差就是负的测量函数
    // 预测measurement  hw(x) = w^  (这里w^就是前面计算出来的w_hat)
    // 伪measurement  zw = 0
    // 按照openvins的residual约定，rm = zw - hw(x) = 0 - w^ = -w^ = -(wm - b_g)
    // 同样，对于加速度计，预测measurement  ha(x) = a^ - R_ItoG* g
    // 伪measurement  za = 0
    // ra = za - ha(x) = 0 - (a^ - R_ItoG* g) = - (a^ - R_ItoG* g) = -(am - b_a - R_ItoG* g)

    // NOTE: 加速度 residual 也能更新姿态！！！
    // 这是因为姿态和加速度计零偏是耦合的，长期静止和足够好的先验方差，可以把他们慢慢分开
    // 所以 VIO初始化质量、IMU 零偏初始值 和 ZUPT效果高度相关
    // ra = R_ItoG*g - a_m + b_a 
    // 其中R_ItoG 就跟姿态 q_GtoI 有关，b_a 是加速度计零偏
    // 采用小角度误差，R(δθ) ≈ (I - skew(δθ))R，则
    // 所以 δ(R_ItoG*g) = -skew(δθ)*R_ItoG*g = -skew(R_ItoG*g)*δθ
    // 所以加速度残差对姿态误差的雅可比 ∂ra/∂(δθ) = -skew(R_ItoG*g)
    // rw = - (wm - b_g)
    // 所以陀螺残差对陀螺零偏的雅可比 ∂rw/∂(δb_g) = -I_3x3

    // NOTE: 这里的残差是 whitened 的，乘了噪声协方差的平方根
    // 为什么有白化？因为后续要做卡方检验，白化后的残差才符合标准正态分布
    // 连续时间白噪声离散到一个时间间隔 Δt 后，瞬时测量的方差近似与 1/Δt 成正比
    // Rw ≈ σ_w^2 / Δt * I_3x3, Ra ≈ σ_a^2 / Δt * I_3x3 
    // 其逆标准差就是 R_w^{-1/2} ≈ sqrt(Δt)/σ_w * I_3x3, R_a^{-1/2} ≈ sqrt(Δt)/σ_a * I_3x3
    // 所以将残差和雅可比都乘上这个逆标准差，就得到了白化后的残差和雅可比
    // w_omega = sqrt(dt)/σ_w, w_accel = sqrt(dt)/σ_a
    // 等价于正确使用原始噪声协方差，但后续可以统一把测量噪声视为单位阵
    // 而且白化后的残差符合标准正态分布，便于后续做卡方检验
    res.block(6 * i + 0, 0, 3, 1) = -w_omega * w_hat;
    if (!integrated_accel_constraint) {
      res.block(6 * i + 3, 0, 3, 1) = -w_accel * (a_hat - state->_imu->Rot() * _gravity);
    } else {
      res.block(6 * i + 3, 0, 3, 1) = -w_accel_v * (state->_imu->vel() - _gravity * dt + state->_imu->Rot().transpose() * a_hat * dt);
    }

    // Measurement Jacobian
    // 默认计算顺序，[δθ, δb_g, δb_a] (δv(if have))
    Eigen::Matrix3d R_GtoI_jacob = (state->_options.do_fej) ? state->_imu->Rot_fej() : state->_imu->Rot();
    H.block(6 * i + 0, 3, 3, 3) = -w_omega * Eigen::Matrix3d::Identity();
    if (!integrated_accel_constraint) {
      H.block(6 * i + 3, 0, 3, 3) = -w_accel * skew_x(R_GtoI_jacob * _gravity);
      H.block(6 * i + 3, 6, 3, 3) = -w_accel * Eigen::Matrix3d::Identity();
    } else {
      H.block(6 * i + 3, 0, 3, 3) = -w_accel_v * R_GtoI_jacob.transpose() * skew_x(a_hat) * dt;
      H.block(6 * i + 3, 6, 3, 3) = -w_accel_v * R_GtoI_jacob.transpose() * dt;
      H.block(6 * i + 3, 9, 3, 3) = w_accel_v * Eigen::Matrix3d::Identity();
    }
    dt_summed += dt;
  }

  // Compress the system (we should be over determined)
  // 一帧 camera interval 可能产生几十维甚至上百维 ZUPT residual,但默认状态只有9维，大部分 measurement rows 实际是冗余的
  // 用 Givens 正交旋转把系统压缩为至多 9 行 ，压缩后至多保留Hc ∈ R_9x9，只保留上方非零的独立行。
  // 这不会改变白化系统的信息量，却能显著降低后续卡方检验和 EKF 更新中矩阵分解的规模
  UpdaterHelper::measurement_compress_inplace(H, res);
  if (H.rows() < 1) {
    return false;
  }

  // Multiply our noise matrix by a fixed amount
  // We typically need to treat the IMU as being "worst" to detect / not become overconfident
  // 由于前面已对白化，理论上可取 R = I，但为了避免过度自信，OpenVINS 允许人为放大噪声协方差
  // λ越大: 认为 ZUPT 伪量测更不可靠，更新更弱、更保守；
  // λ越小: 更相信“静止”假设，更新更强，也更容易过度自信
  // 这是一种工程上的保守调参手段。静止检测误判时，过强的 ZUPT 会造成很大伤害，因此一般不建议把该值设得过小
  // 理论 IMU white noise 往往解释不了实际静止状态的:
  // 电机振动;风扇震动; 车体共振; 地板震动; ADC 量化; 未建模噪声; 温漂
  // λ越大(表明实际上IMU会datasheet更吵)会让 ZUPT detection 更宽容，同时 update 也更保守
  Eigen::MatrixXd R = _zupt_noise_multiplier * Eigen::MatrixXd::Identity(res.rows(), res.rows());

  // Next propagate the biases forward in time
  // NOTE: G*Qd*G^t = dt*Qd*dt = dt*(1/dt*Qc)*dt = dt*Qc
  // 零偏随机游走协方差
  Eigen::MatrixXd Q_bias = Eigen::MatrixXd::Identity(6, 6);
  Q_bias.block(0, 0, 3, 3) *= dt_summed * _noises.sigma_wb_2;
  Q_bias.block(3, 3, 3, 3) *= dt_summed * _noises.sigma_ab_2;

  // Chi2 distance check
  // 但仅靠 chi-square 可能会把匀速运动判成静止，所以后面还需要视差判断
  // NOTE: we also append the propagation we "would do before the update" if this was to be accepted (just the bias evolution)
  // NOTE: we don't propagate first since if we fail the chi2 then we just want to return and do normal logic
  Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
  if (model_time_varying_bias) {
    P_marg.block(3, 3, 6, 6) += Q_bias; // 如果接受 ZUPT，那么偏置在这段时间内本来也应该积累这份不确定性
  }
  Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
  double chi2 = res.dot(S.llt().solve(res));

  // Get our threshold (we precompute up to 1000 but handle the case that it is more)
  double chi2_check;
  if (res.rows() < 1000) {
    chi2_check = chi_squared_table[res.rows()];
  } else {
    boost::math::chi_squared chi_squared_dist(res.rows());
    chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    PRINT_WARNING(YELLOW "[ZUPT]: chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
  }

  // Check if the image disparity
  bool disparity_passed = false;
  if (override_with_disparity_check) {

    // Get the disparity statistics from this image to the previous
    // 它比较前后两帧中共同特征的平均像素位移
    double time0_cam = state->_timestamp;
    double time1_cam = timestamp;
    int num_features = 0;
    double disp_avg = 0.0;
    double disp_var = 0.0;
    FeatureHelper::compute_disparity(_db, time0_cam, time1_cam, disp_avg, disp_var, num_features);

    // Check if this disparity is enough to be classified as moving
    disparity_passed = (disp_avg < _zupt_max_disparity && num_features > 20);
    if (disparity_passed) {
      PRINT_INFO(CYAN "[ZUPT]: passed disparity (%.3f < %.3f, %d features)\n" RESET, disp_avg, _zupt_max_disparity, (int)num_features);
    } else {
      PRINT_DEBUG(YELLOW "[ZUPT]: failed disparity (%.3f > %.3f, %d features)\n" RESET, disp_avg, _zupt_max_disparity, (int)num_features);
    }
  }

  // Check if we are currently zero velocity
  // We need to pass the chi2 and not be above our velocity threshold
  // 图像视差检测为何能覆盖 IMU 检测? 
  // 因为 IMU 卡方检验和速度阈值只能判断“是否静止”，但无法判断“是否匀速运动”。
  // 而图像视差检测可以判断相机是否有运动，哪怕是匀速运动。
  // 所以当前的策略是: 
  // 1. 如果图像视差检测通过，则认为静止，直接接受 ZUPT；
  // 2. 如果图像视差检测不通过，则再用 IMU 卡方检验和速度阈值判断是否静止，若不通过则拒绝 ZUPT。
  // 这是一个务实但需要留意的策略。相机若面对低纹理区域、重复纹理或远景，低视差不必然代表静止；
  // 反过来，设备静止时 IMU 强振动又可能使卡方检验失败。因此这里是在两种传感器判据间做工程折中。
  // 对轮式机器人而言，最好IMU ∩ WheelEncoder ∩ Camera 三传感器同时判断静止，才能更可靠地触发 ZUPT。
  // 至少 (IMU ∩ WheelEncoder) ∪ (IMU ∩ Camera) 组合，而不是让camera单独override

  if (!disparity_passed && (chi2 > _options.chi2_multipler * chi2_check || state->_imu->vel().norm() > _zupt_max_velocity)) {
    last_zupt_state_timestamp = 0.0;
    last_zupt_count = 0;
    PRINT_DEBUG(YELLOW "[ZUPT]: rejected |v_IinG| = %.3f (chi2 %.3f > %.3f)\n" RESET, state->_imu->vel().norm(), chi2,
                _options.chi2_multipler * chi2_check);
    return false;
  }
  PRINT_INFO(CYAN "[ZUPT]: accepted |v_IinG| = %.3f (chi2 %.3f < %.3f)\n" RESET, state->_imu->vel().norm(), chi2,
             _options.chi2_multipler * chi2_check);

  // Do our update, only do this update if we have previously detected
  // If we have succeeded, then we should remove the current timestamp feature tracks
  // This is because we will not clone at this timestep and instead do our zero velocity update
  // NOTE: We want to keep the tracks from the second time we have called the zv-upt since this won't have a clone
  // NOTE: All future times after the second call to this function will also *not* have a clone, so we can remove those
  // 连续 ZUPT 时为何清理特征轨迹?
  // 设备若连续静止，系统不会按普通流程在每个时刻创建新的 clone。
  // 继续保留这些时刻的视觉轨迹，会得到无法对应到 clone 的测量，或在之后形成大量没有有效视差的冗余轨迹
  // 因此从连续 ZUPT 的第三次开始，清掉前一 ZUPT 时刻的精确观测记录。
  // last_zupt_count 用于保留刚开始静止时仍可能需要的那部分轨迹，避免过早清理
  if (last_zupt_count >= 2) {
    _db->cleanup_measurements_exact(last_zupt_state_timestamp);
  }

  // Else we are good, update the system
  // 1) update with our IMU measurements directly
  // 2) propagate and then explicitly say that our ori, pos, and vel should be zero
  // 注意：默认系统没有直接把速度硬设为零，也没有显式约束位置不动
  // 它直接观测的是: 
  // 角速度为0；加速度与重力一致。
  // 因此它主要校正: 
  // 陀螺仪零偏; 加速度计零偏; roll/pitch 姿态；与它们存在协方差关联的其他状态（位置、速度、偏置等）间接校正。
  if (!explicitly_enforce_zero_motion) {
    // 默认不显式加入“位置不变、姿态不变、速度为零”的 9 维硬约束，而是通过 IMU 静止约束间接更新状态
    // Next propagate the biases forward in time
    // NOTE: G*Qd*G^t = dt*Qd*dt = dt*Qc
    if (model_time_varying_bias) { // 先让零偏随机游走协方差正式传播(在静止过程中也还是要允许零偏游走)
      Eigen::MatrixXd Phi_bias = Eigen::MatrixXd::Identity(6, 6);
      std::vector<std::shared_ptr<Type>> Phi_order;
      Phi_order.push_back(state->_imu->bg());
      Phi_order.push_back(state->_imu->ba());
      StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_bias, Q_bias);
    }

    // Finally move the state time forward
    // 然后用伪量测做 EKF 更新
    StateHelper::EKFUpdate(state, Hx_order, H, res, R);
    // 状态时间直接跳到 timestamp，位置通常保持不变。这符合静止阶段的物理假设，但它并不是“强制所有运动量严格归零”
    state->_timestamp = timestamp;

  } else {
    // 显式加入“位置不变、姿态不变、速度为零”的 9 维硬约束(强制零运动)，直接把状态推进到 timestamp
    // Propagate the state forward in time
    double time0_cam = last_zupt_state_timestamp;
    double time1_cam = timestamp;
    _prop->propagate_and_clone(state, time1_cam);

    // Create the update system!
    H = Eigen::MatrixXd::Zero(9, 15);
    res = Eigen::VectorXd::Zero(9);
    R = Eigen::MatrixXd::Identity(9, 9);

    // residual (order is ori, pos, vel)
    Eigen::Matrix3d R_GtoI0 = state->_clones_IMU.at(time0_cam)->Rot();
    Eigen::Vector3d p_I0inG = state->_clones_IMU.at(time0_cam)->pos();
    Eigen::Matrix3d R_GtoI1 = state->_clones_IMU.at(time1_cam)->Rot();
    Eigen::Vector3d p_I1inG = state->_clones_IMU.at(time1_cam)->pos();
    res.block(0, 0, 3, 1) = -log_so3(R_GtoI0 * R_GtoI1.transpose());
    res.block(3, 0, 3, 1) = p_I1inG - p_I0inG;
    res.block(6, 0, 3, 1) = state->_imu->vel();
    res *= -1;

    // jacobian (order is q0, p0, q1, p1, v0)
    Hx_order.clear();
    Hx_order.push_back(state->_clones_IMU.at(time0_cam));
    Hx_order.push_back(state->_clones_IMU.at(time1_cam));
    Hx_order.push_back(state->_imu->v());
    if (state->_options.do_fej) {
      R_GtoI0 = state->_clones_IMU.at(time0_cam)->Rot_fej();
    }
    H.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    H.block(0, 6, 3, 3) = -R_GtoI0;
    H.block(3, 3, 3, 3) = -Eigen::Matrix3d::Identity();
    H.block(3, 9, 3, 3) = Eigen::Matrix3d::Identity();
    H.block(6, 12, 3, 3) = Eigen::Matrix3d::Identity();

    // noise (order is ori, pos, vel)
    R.block(0, 0, 3, 3) *= std::pow(1e-2, 2);
    R.block(3, 3, 3, 3) *= std::pow(1e-1, 2);
    R.block(6, 6, 3, 3) *= std::pow(1e-1, 2);

    // finally update and remove the old clone
    StateHelper::EKFUpdate(state, Hx_order, H, res, R);
    StateHelper::marginalize(state, state->_clones_IMU.at(time1_cam));
    state->_clones_IMU.erase(time1_cam);
  }

  // Finally return 最后记录当前静止状态
  // 一旦拒绝 ZUPT，则归零，相当于打断连续静止段，下次一接受会重新从第一轮ZUPT计数
  last_zupt_state_timestamp = timestamp;
  last_zupt_count++;
  return true;
}