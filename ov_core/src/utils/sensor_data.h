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

#ifndef OV_CORE_SENSOR_DATA_H
#define OV_CORE_SENSOR_DATA_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ov_core {

/**
 * @brief Struct for a single imu measurement (time, wm, am)
 */
struct ImuData {

  /// Timestamp of the reading
  double timestamp;

  /// Gyroscope reading, angular velocity (rad/s)
  Eigen::Matrix<double, 3, 1> wm;

  /// Accelerometer reading, linear acceleration (m/s^2)
  Eigen::Matrix<double, 3, 1> am;

  /// Sort function to allow for using of STL containers
  bool operator<(const ImuData &other) const { return timestamp < other.timestamp; }
};

/**
 * @brief Struct for a collection of camera measurements.
 * openvins支持多相机系统，每个CameraData实例可以包含来自多个相机的图像数据。
 * 每个图像数据都与一个唯一的相机ID相关联，以便在处理时区分不同的相机数据流。
 * 该结构体包含一个时间戳、一个相机ID列表和一个图像列表。
 * 时间戳表示图像数据的采集时间，相机ID列表用于标识不同的相机数据流，而图像列表则包含了来自各个相机的图像数据。
 * 通过这种方式，系统能够同时处理来自多个相机的数据，并且能够根据相机ID进行区分和处理。
 * For each image we have a camera id and timestamp that it occured at.
 * If there are multiple cameras we will treat it as pair-wise stereo tracking.
 */
struct CameraData {

  /// Timestamp of the reading
  double timestamp;

  /// Camera ids for each of the images collected
  // size等于1时候是单目，size等于2时候是双目
  std::vector<int> sensor_ids;

  /// Raw image we have collected for each camera
  // size等于1时候是单目，size等于2时候是双目
  std::vector<cv::Mat> images;

  /// Tracking masks for each camera we have，掩码mask，用于跟踪指定区域特征点
  // size等于1时候是单目，size等于2时候是双目
  std::vector<cv::Mat> masks;

  /// Sort function to allow for using of STL containers
  /// 如果时间戳相同，则根据sensor_ids中的最小值进行排序，确保同一时间戳的图像按照相机ID顺序处理
  /// 否则根据时间戳进行排序，确保图像按照时间顺序处理
  bool operator<(const CameraData &other) const {
    if (timestamp == other.timestamp) {
      int id = *std::min_element(sensor_ids.begin(), sensor_ids.end());
      int id_other = *std::min_element(other.sensor_ids.begin(), other.sensor_ids.end());
      return id < id_other;
    } else {
      return timestamp < other.timestamp;
    }
  }
};

} // namespace ov_core

#endif // OV_CORE_SENSOR_DATA_H