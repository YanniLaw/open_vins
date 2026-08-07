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

#include "Feature.h"

using namespace ov_core;

/**
 * @brief Cleans old feature measurements that are not in the list of valid times.
 * 删除那些不在 clonetimes 中的特征观测，确保每个测量都对应一个已知的克隆位姿
 * 为什么要这样做？
 * Feature 里存的观测是前端跟踪时按时间顺序累积的，可能包含各种杂散时刻的测量：
 * - 多相机异步处理时，某相机的帧可能晚到
 * - 中间有些帧没成功跟踪到该特征
 * 而 MSCKF 更新时，需要每个测量都能对应一个克隆时刻的相机位姿（用于三角化和雅可比计算）。如果测量发生在没有克隆的时刻，就无法使用。
 * @param valid_times A vector of valid timestamps. Measurements not in this list will be removed.
 */
void Feature::clean_old_measurements(const std::vector<double> &valid_times) {

  // Loop through each of the cameras we have 遍历每个相机
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement 一致性检查: 三个数组长度必须相等
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators 三组平行迭代器，同步前进/删除
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are not in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      // 时间戳不在给定的 valid_times列表里 → 三组数据一起删
      if (std::find(valid_times.begin(), valid_times.end(), *it1) == valid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_invalid_measurements(const std::vector<double> &invalid_times) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(invalid_times.begin(), invalid_times.end(), *it1) != invalid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

// 删除指定时间戳之前的观测数据
void Feature::clean_older_measurements(double timestamp) {

  // Loop through each of the cameras we have
  // std::unordered_map<size_t, std::vector<double>> timestamps;
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    // pair.first 为相机ID，pair.second 为该相机的观测时间戳数组
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are older then the specified one
    while (it1 != timestamps[pair.first].end()) {
      if (*it1 <= timestamp) {
        // 调用 erase 后，被删位置以及其后的迭代器都会失效，但是 erase 会返回下一个有效元素的新迭代器
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}
