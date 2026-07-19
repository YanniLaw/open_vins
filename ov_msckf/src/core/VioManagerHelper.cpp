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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "types/LandmarkRepresentation.h"
#include "utils/print.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void VioManager::initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate) {

  // Initialize the system
  state->_imu->set_value(imustate.block(1, 0, 16, 1));
  state->_imu->set_fej(imustate.block(1, 0, 16, 1));

  // Fix the global yaw and position gauge freedoms
  // TODO: Why does this break out simulation consistency metrics?
  std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu};
  Eigen::MatrixXd Cov = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(state->_imu->size(), state->_imu->size());
  Cov.block(0, 0, 3, 3) = std::pow(0.017, 2) * Eigen::Matrix3d::Identity(); // q
  Cov.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();  // p
  Cov.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity();  // v (static)
  StateHelper::set_initial_covariance(state, Cov, order);

  // Set the state time
  state->_timestamp = imustate(0, 0);
  startup_time = imustate(0, 0);
  is_initialized_vio = true;

  // Cleanup any features older then the initialization time
  trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
  }

  // Print what we init'ed with
  PRINT_DEBUG(GREEN "[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!\n" RESET);
  PRINT_DEBUG(GREEN "[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
              state->_imu->quat()(2), state->_imu->quat()(3));
  PRINT_DEBUG(GREEN "[INIT]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
              state->_imu->bias_g()(2));
  PRINT_DEBUG(GREEN "[INIT]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
  PRINT_DEBUG(GREEN "[INIT]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
              state->_imu->bias_a()(2));
  PRINT_DEBUG(GREEN "[INIT]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
}

bool VioManager::try_to_initialize(const ov_core::CameraData &message) {

  // Directly return if the initialization thread is running
  // Note that we lock on the queue since we could have finished an update
  // And are using this queue to propagate the state forward. We should wait in this case
  // 核心设计思想：初始化是耗时操作，不能阻塞主线程接收 IMU/相机数据。camera_queue_init 作为"缓冲区"，
  // 让初始化完成后能快速追上当前时间，而不是从初始化时刻一帧帧重新处理所有丢弃的帧。

  // 初始化可能需要几百毫秒到几秒（收集足够的 IMU/Jerk 数据），但是在这段时间内新的相机帧仍在不断到来。
  // 当初始化线程已经在运行时，后续每一帧camera运行到 try_to_initialize 时都会把当前帧的时间戳缓存起来(避免数据丢失)，
  // 同时会返回 false，导致这些帧在 track_image_and_update 中被直接跳过。
  // 如果初始化成功，则缓存的时间戳会被依次处理。用缓存时间戳快速前推状态到最新时刻（按 clone_rate 跳步，不逐帧做全量更新）
  // 如果初始化失败，则clear() 清空缓存，下次重试从头开始。

  if (thread_init_running) {
    std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
    camera_queue_init.push_back(message.timestamp); // 缓存当前时间戳
    return false;
  }

  // If the thread was a success, then return success! 初始化成功了也直接返回成功，避免第二次初始化
  if (thread_init_success) {
    return true;
  }

  // Run the initialization in a second thread so it can go as slow as it desires
  thread_init_running = true;
  std::thread thread([&] {
    // Returns from our initializer
    double timestamp; // VIO系统初始化的时间戳
    Eigen::MatrixXd covariance;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    auto init_rT1 = boost::posix_time::microsec_clock::local_time();

    // Try to initialize the system
    // We will wait for a jerk if we do not have the zero velocity update enabled
    // Otherwise we can initialize right away as the zero velocity will handle the stationary case
    // 如果没有启用零速度更新，则必须等待jerk(加加速度)来初始化; 否则可以立即初始化，因为零速度更新将处理静止情况
    // 为什么需要jerk？
    // 初始化时需要估计 IMU 的陀螺仪偏置和加速度计偏置
    // 静止状态：仅有重力加速度，无法区分偏置和真实加速度
    // 有加速度突变（Jerk）：系统观察到动态变化，能分离偏置成分
    bool wait_for_jerk = (updaterZUPT == nullptr);
    // 如果是静止初始化，则timestamp为静止的前半段最后一个测量的时间戳
    bool success = initializer->initialize(timestamp, covariance, order, state->_imu, wait_for_jerk);

    // If we have initialized successfully we will set the covariance and state elements as needed
    // TODO: set the clones and SLAM features here so we can start updating right away...
    if (success) {

      // Set our covariance (state should already be set in the initializer)
      StateHelper::set_initial_covariance(state, covariance, order);

      // Set the state time
      state->_timestamp = timestamp;
      startup_time = timestamp;

      // Cleanup any features older than the initialization time
      // Also increase the number of features to the desired amount during estimation
      // NOTE: we will split the total number of features over all cameras uniformly
      // 既然已经初始化完成了，就清理掉特征数据库中比初始化时间更早的测量数据
      trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
      // 为什么要增加特征数量？
      // 因为初始化阶段和正常估计阶段用的是两套目标数。
      // 创建前端追踪器时，先用的是 init_max_features，专门给初始化阶段降负载
      // 初始化成功后，再切到估计阶段目标 num_pts / num_cameras
      // 为什么要这么设计？
      // 初始化更重（要解初始姿态/速度/偏置等），用更少特征可降低计算开销、加快初始化。
      // 一旦初始化完成，系统进入持续估计，需要更多特征提升鲁棒性和精度，所以把追踪目标调高
      trackFEATS->set_num_features(std::floor((double)params.num_pts / (double)params.state_options.num_cameras));
      if (trackARUCO != nullptr) {
        trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
      }

      // If we are moving then don't do zero velocity update4
      if (state->_imu->vel().norm() > params.zupt_max_velocity) {
        has_moved_since_zupt = true;
      }

      // Else we are good to go, print out our stats
      auto init_rT2 = boost::posix_time::microsec_clock::local_time();
      PRINT_INFO(GREEN "[init]: successful initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
      PRINT_INFO(GREEN "[init]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
                 state->_imu->quat()(2), state->_imu->quat()(3));
      PRINT_INFO(GREEN "[init]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
                 state->_imu->bias_g()(2));
      PRINT_INFO(GREEN "[init]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
      PRINT_INFO(GREEN "[init]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
                 state->_imu->bias_a()(2));
      PRINT_INFO(GREEN "[init]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));

      // Remove any camera times that are order then the initialized time
      // This can happen if the initialization has taken a while to perform
      // 把系统从“初始化完成时刻”快速推进到“当前前端时间”(当前初始化线程执行时前面一直在缓存最新camera时间戳)，减少初始化线程造成的时间落后
      // 初始化可能花了几百毫秒到几秒，这段时间新来的图像时间戳先被缓存了，现在要把这些“晚于初始化时刻”的帧用于状态前推
      // 早于等于 timestamp 的时刻会被丢弃，因为状态已经在那个时间点上了
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      std::vector<double> camera_timestamps_to_init;
      // camera_queue_init是调用try_to_initialize时缓存的相机时间戳(如果当前线程花了很久才完成初始化的话，这些时间戳会被缓存起来)
      for (size_t i = 0; i < camera_queue_init.size(); i++) {
        if (camera_queue_init.at(i) > timestamp) { // 将初始化时刻之后的相机时间戳加入待初始化列表
          camera_timestamps_to_init.push_back(camera_queue_init.at(i));
        }
      }

      // Now we have initialized we will propagate the state to the current timestep
      // In general this should be ok as long as the initialization didn't take too long to perform
      // Propagating over multiple seconds will become an issue if the initial biases are bad
      // 目的不是每一帧都 clone，而是按步长跳着取，避免一次性加太多 clone 让状态爆炸，
      // 其实如果缓存的时间戳不超过 max_clone_size，那么每个时间戳都会去处理的
      // 因为缓存的时间戳数量可能远大于滑窗容量（如 10 秒视频对应 300 帧，但 max_clone_size 可能只有 15 个）
      // 不用每帧都做完整传播+克隆，而是等间隔跳取，使克隆数量受控，每传播一帧后立即 marginalize_old_clone() 保证不超过最大数量

      // Note: 为什么不做完整更新而只做传播？
      // 初始化线程内部没有执行 do_feature_propagate_update 的完整流程（MSCKF/SLAM 更新、重新三角化等），原因：
      // 缓存帧的特征已被丢弃：初始化时已经 cleanup_measurements(timestamp)，缓存帧对应的特征观测数据已被清理
      // 避免重复计算：完整的 MSCKF/SLAM 更新计算量大，追赶时做全量更新是浪费
      // 尽快交付状态：追赶的目的是让状态时间尽快对齐到当前时刻，后续正常帧会做完整的更新
      // 追赶完成后，下一帧正常相机到来时，track_image_and_update 中的 do_feature_propagate_update 才会执行完整的传播+更新+三角化+边缘化流程。

      size_t clone_rate = (size_t)((double)camera_timestamps_to_init.size() / (double)params.state_options.max_clone_size) + 1;
      for (size_t i = 0; i < camera_timestamps_to_init.size(); i += clone_rate) {
        // 从当前状态时间，利用缓存的 IMU 数据，将 IMU 状态（姿态、速度、位置）前推到缓存帧的时间点
        // 同时传播协方差，更新状态转移和噪声累积
        propagator->propagate_and_clone(state, camera_timestamps_to_init.at(i));
        StateHelper::marginalize_old_clone(state); // 如果 clone 超过上限，立即边缘化最老的，保持滑窗大小受控
      }
      PRINT_DEBUG(YELLOW "[init]: moved the state forward %.2f seconds\n" RESET, state->_timestamp - timestamp);
      thread_init_success = true;
      camera_queue_init.clear();

    } else {
      auto init_rT2 = boost::posix_time::microsec_clock::local_time();
      PRINT_DEBUG(YELLOW "[init]: failed initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
      thread_init_success = false;
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      camera_queue_init.clear(); // 如果初始化失败，则清空缓存，下次重试从头开始
    }

    // Finally, mark that the thread has finished running
    thread_init_running = false;
  });

  // If we are single threaded, then run single threaded
  // Otherwise detach this thread so it runs in the background!
  if (!params.use_multi_threading_subs) {
    thread.join();
  } else {
    thread.detach();
  }
  return false;
}

void VioManager::retriangulate_active_tracks(const ov_core::CameraData &message) {

  // Start timing
  boost::posix_time::ptime retri_rT1, retri_rT2, retri_rT3;
  retri_rT1 = boost::posix_time::microsec_clock::local_time();

  // Clear old active track data
  assert(state->_clones_IMU.find(message.timestamp) != state->_clones_IMU.end());
  active_tracks_time = message.timestamp;
  active_image = cv::Mat();
  trackFEATS->display_active(active_image, 255, 255, 255, 255, 255, 255, " ");
  if (!active_image.empty()) {
    active_image = active_image(cv::Rect(0, 0, message.images.at(0).cols, message.images.at(0).rows));
  }
  active_tracks_posinG.clear();
  active_tracks_uvd.clear();

  // Current active tracks in our frontend
  // TODO: should probably assert here that these are at the message time...
  auto last_obs = trackFEATS->get_last_obs();
  auto last_ids = trackFEATS->get_last_ids();

  // New set of linear systems that only contain the latest track info
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A_new;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b_new;
  std::map<size_t, int> active_feat_linsys_count_new;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG_new;

  // Append our new observations for each camera
  std::map<size_t, cv::Point2f> feat_uvs_in_cam0;
  for (auto const &cam_id : message.sensor_ids) {

    // IMU historical clone
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(active_tracks_time)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(active_tracks_time)->pos();

    // Calibration for this cam_id
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

    // Convert current CAMERA position relative to global
    Eigen::Matrix3d R_GtoCi = R_ItoC * R_GtoI;
    Eigen::Vector3d p_CiinG = p_IinG - R_GtoCi.transpose() * p_IinC;

    // Loop through each measurement
    assert(last_obs.find(cam_id) != last_obs.end());
    assert(last_ids.find(cam_id) != last_ids.end());
    for (size_t i = 0; i < last_obs.at(cam_id).size(); i++) {

      // Record this feature uv if is seen from cam0
      size_t featid = last_ids.at(cam_id).at(i);
      cv::Point2f pt_d = last_obs.at(cam_id).at(i).pt;
      if (cam_id == 0) {
        feat_uvs_in_cam0[featid] = pt_d;
      }

      // Skip this feature if it is a SLAM feature (the state estimate takes priority)
      if (state->_features_SLAM.find(featid) != state->_features_SLAM.end()) {
        continue;
      }

      // Get the UV coordinate normal
      cv::Point2f pt_n = state->_cam_intrinsics_cameras.at(cam_id)->undistort_cv(pt_d);
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << pt_n.x, pt_n.y, 1;
      b_i = R_GtoCi.transpose() * b_i;
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
      Eigen::Vector3d bi = Ai * p_CiinG;
      if (active_feat_linsys_A.find(featid) == active_feat_linsys_A.end()) {
        active_feat_linsys_A_new.insert({featid, Ai});
        active_feat_linsys_b_new.insert({featid, bi});
        active_feat_linsys_count_new.insert({featid, 1});
      } else {
        active_feat_linsys_A_new[featid] = Ai + active_feat_linsys_A[featid];
        active_feat_linsys_b_new[featid] = bi + active_feat_linsys_b[featid];
        active_feat_linsys_count_new[featid] = 1 + active_feat_linsys_count[featid];
      }

      // For this feature, recover its 3d position if we have enough observations!
      if (active_feat_linsys_count_new.at(featid) > 3) {

        // Recover feature estimate
        Eigen::Matrix3d A = active_feat_linsys_A_new[featid];
        Eigen::Vector3d b = active_feat_linsys_b_new[featid];
        Eigen::MatrixXd p_FinG = A.colPivHouseholderQr().solve(b);
        Eigen::MatrixXd p_FinCi = R_GtoCi * (p_FinG - p_CiinG);

        // Check A and p_FinCi
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
        Eigen::MatrixXd singularValues;
        singularValues.resize(svd.singularValues().rows(), 1);
        singularValues = svd.singularValues();
        double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

        // If we have a bad condition number, or it is too close
        // Then set the flag for bad (i.e. set z-axis to nan)
        if (std::abs(condA) <= params.featinit_options.max_cond_number && p_FinCi(2, 0) >= params.featinit_options.min_dist &&
            p_FinCi(2, 0) <= params.featinit_options.max_dist && !std::isnan(p_FinCi.norm())) {
          active_tracks_posinG_new[featid] = p_FinG;
        }
      }
    }
  }
  size_t total_triangulated = active_tracks_posinG.size();

  // Update active set of linear systems
  active_feat_linsys_A = active_feat_linsys_A_new;
  active_feat_linsys_b = active_feat_linsys_b_new;
  active_feat_linsys_count = active_feat_linsys_count_new;
  active_tracks_posinG = active_tracks_posinG_new;
  retri_rT2 = boost::posix_time::microsec_clock::local_time();

  // Return if no features
  if (active_tracks_posinG.empty() && state->_features_SLAM.empty())
    return;

  // Append our SLAM features we have
  for (const auto &feat : state->_features_SLAM) {
    Eigen::Vector3d p_FinG = feat.second->get_xyz(false);
    if (LandmarkRepresentation::is_relative_representation(feat.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(feat.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->Rot();
      Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->Rot();
      Eigen::Vector3d p_IinG = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feat.second->get_xyz(false) - p_IinC) + p_IinG;
    }
    active_tracks_posinG[feat.second->_featid] = p_FinG;
  }

  // Calibration of the first camera (cam0)
  std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(0);
  std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0);
  Eigen::Matrix<double, 3, 3> R_ItoC = calibration->Rot();
  Eigen::Matrix<double, 3, 1> p_IinC = calibration->pos();

  // Get current IMU clone state
  std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(active_tracks_time);
  Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
  Eigen::Vector3d p_IiinG = clone_Ii->pos();

  // 4. Next we can update our variable with the global position
  //    We also will project the features into the current frame
  for (const auto &feat : active_tracks_posinG) {

    // For now skip features not seen from current frame
    // TODO: should we publish other features not tracked in cam0??
    if (feat_uvs_in_cam0.find(feat.first) == feat_uvs_in_cam0.end())
      continue;

    // Calculate the depth of the feature in the current frame
    // Project SLAM feature and non-cam0 features into the current frame of reference
    Eigen::Vector3d p_FinIi = R_GtoIi * (feat.second - p_IiinG);
    Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
    double depth = p_FinCi(2);
    Eigen::Vector2d uv_dist;
    if (feat_uvs_in_cam0.find(feat.first) != feat_uvs_in_cam0.end()) {
      uv_dist << (double)feat_uvs_in_cam0.at(feat.first).x, (double)feat_uvs_in_cam0.at(feat.first).y;
    } else {
      Eigen::Vector2d uv_norm;
      uv_norm << p_FinCi(0) / depth, p_FinCi(1) / depth;
      uv_dist = state->_cam_intrinsics_cameras.at(0)->distort_d(uv_norm);
    }

    // Skip if not valid (i.e. negative depth, or outside of image)
    if (depth < 0.1) {
      continue;
    }

    // Skip if not valid (i.e. negative depth, or outside of image)
    int width = state->_cam_intrinsics_cameras.at(0)->w();
    int height = state->_cam_intrinsics_cameras.at(0)->h();
    if (uv_dist(0) < 0 || (int)uv_dist(0) >= width || uv_dist(1) < 0 || (int)uv_dist(1) >= height) {
      // PRINT_DEBUG("feat %zu -> depth = %.2f | u_d = %.2f | v_d = %.2f\n",(*it2)->featid,depth,uv_dist(0),uv_dist(1));
      continue;
    }

    // Finally construct the uv and depth
    Eigen::Vector3d uvd;
    uvd << uv_dist, depth;
    active_tracks_uvd.insert({feat.first, uvd});
  }
  retri_rT3 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds for triangulation (%zu tri of %zu active)\n" RESET,
            (retri_rT2 - retri_rT1).total_microseconds() * 1e-6, total_triangulated, active_feat_linsys_A.size());
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds for re-projection into current\n" RESET, (retri_rT3 - retri_rT2).total_microseconds() * 1e-6);
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds total\n" RESET, (retri_rT3 - retri_rT1).total_microseconds() * 1e-6);
}

cv::Mat VioManager::get_historical_viz_image() {

  // Return if not ready yet
  if (state == nullptr || trackFEATS == nullptr)
    return cv::Mat();

  // Build an id-list of what features we should highlight (i.e. SLAM)
  std::vector<size_t> highlighted_ids;
  for (const auto &feat : state->_features_SLAM) {
    highlighted_ids.push_back(feat.first);
  }

  // Text we will overlay if needed
  std::string overlay = (did_zupt_update) ? "zvupt" : "";
  overlay = (!is_initialized_vio) ? "init" : overlay;

  // Get the current active tracks
  cv::Mat img_history;
  trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255, highlighted_ids, overlay);
  if (trackARUCO != nullptr) {
    trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255, highlighted_ids, overlay);
    // trackARUCO->display_active(img_history, 0, 255, 255, 255, 255, 255, overlay);
  }

  // Finally return the image
  return img_history;
}

std::vector<Eigen::Vector3d> VioManager::get_features_SLAM() {
  std::vector<Eigen::Vector3d> slam_feats;
  for (auto &f : state->_features_SLAM) {
    if ((int)f.first <= 4 * state->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      slam_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      slam_feats.push_back(f.second->get_xyz(false));
    }
  }
  return slam_feats;
}

std::vector<Eigen::Vector3d> VioManager::get_features_ARUCO() {
  std::vector<Eigen::Vector3d> aruco_feats;
  for (auto &f : state->_features_SLAM) {
    if ((int)f.first > 4 * state->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      aruco_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      aruco_feats.push_back(f.second->get_xyz(false));
    }
  }
  return aruco_feats;
}
