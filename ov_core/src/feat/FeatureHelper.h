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

#ifndef OV_CORE_FEATURE_HELPER_H
#define OV_CORE_FEATURE_HELPER_H

#include <Eigen/Eigen>
#include <memory>
#include <mutex>
#include <vector>

#include "Feature.h"
#include "FeatureDatabase.h"
#include "utils/print.h"

namespace ov_core {

/**
 * @brief Contains some nice helper functions for features.
 *
 * These functions should only depend on feature and the feature database.
 */
class FeatureHelper {

public:
  /**
   * @brief This functions will compute the disparity between common features in the two frames.
   *
   * First we find all features in the first frame.
   * Then we loop through each and find the uv of it in the next requested frame.
   * Features are skipped if no tracked feature is found (it was lost).
   * NOTE: this is on the RAW coordinates of the feature not the normalized ones.
   * NOTE: This computes the disparity over all cameras!
   *
   * @param db Feature database pointer
   * @param time0 First camera frame timestamp
   * @param time1 Second camera frame timestamp
   * @param disp_mean Average raw disparity
   * @param disp_var Variance of the disparities
   * @param total_feats Total number of common features
   */
  static void compute_disparity(std::shared_ptr<ov_core::FeatureDatabase> db, double time0, double time1, double &disp_mean,
                                double &disp_var, int &total_feats) {

    // Get features seen from the first image
    std::vector<std::shared_ptr<Feature>> feats0 = db->features_containing(time0, false, true);

    // Compute the disparity
    std::vector<double> disparities;
    for (auto &feat : feats0) {

      // Get the two uvs for both times
      for (auto &campairs : feat->timestamps) {

        // First find the two timestamps
        size_t camid = campairs.first;
        auto it0 = std::find(feat->timestamps.at(camid).begin(), feat->timestamps.at(camid).end(), time0);
        auto it1 = std::find(feat->timestamps.at(camid).begin(), feat->timestamps.at(camid).end(), time1);
        if (it0 == feat->timestamps.at(camid).end() || it1 == feat->timestamps.at(camid).end())
          continue;
        auto idx0 = std::distance(feat->timestamps.at(camid).begin(), it0);
        auto idx1 = std::distance(feat->timestamps.at(camid).begin(), it1);

        // Now lets calculate the disparity
        Eigen::Vector2f uv0 = feat->uvs.at(camid).at(idx0).block(0, 0, 2, 1);
        Eigen::Vector2f uv1 = feat->uvs.at(camid).at(idx1).block(0, 0, 2, 1);
        disparities.push_back((uv1 - uv0).norm());
      }
    }

    // If no disparities, just return
    if (disparities.size() < 2) {
      disp_mean = -1;
      disp_var = -1;
      total_feats = 0;
    }

    // Compute mean and standard deviation in respect to it
    disp_mean = 0;
    for (double disp_i : disparities) {
      disp_mean += disp_i;
    }
    disp_mean /= (double)disparities.size();
    disp_var = 0;
    for (double &disp_i : disparities) {
      disp_var += std::pow(disp_i - disp_mean, 2);
    }
    disp_var = std::sqrt(disp_var / (double)(disparities.size() - 1));
    total_feats = (int)disparities.size();
  }

  /**
   * @brief This functions will compute the disparity over all features we have
   * 计算时间窗口内特征点的平均视差（pixel displacement），判断相机是否在运动
   * NOTE: this is on the RAW coordinates of the feature not the normalized ones.
   * NOTE: This computes the disparity over all cameras!
   *
   * @param db Feature database pointer 特征数据库
   * @param disp_mean Average raw disparity 输出: 平均像素视差
   * @param disp_var Variance of the disparities 输出: 视差方差
   * @param total_feats Total number of common features 输出: 参与计算的总特征数
   * @param newest_time Only compute disparity for ones older (-1 to disable) 时间上界（-1 = 无限制）
   * @param oldest_time Only compute disparity for ones newer (-1 to disable) 时间下界（-1 = 无限制）
   */
  static void compute_disparity(std::shared_ptr<ov_core::FeatureDatabase> db, double &disp_mean, double &disp_var, int &total_feats,
                                double newest_time = -1, double oldest_time = -1) {

    // Compute the disparity
    std::vector<double> disparities;
    // std::unordered_map<size_t, std::shared_ptr<Feature>> 特征数据库类型:特征ID与特征数据类型(包含该特征的所有观测)映射
    for (auto &feat : db->get_internal_data()) {
      // std::unordered_map<size_t, std::vector<double>> timestamps 相机ID与观测时间戳映射
      for (auto &campairs : feat.second->timestamps) {

        // Skip if only one observation 
        if (campairs.second.size() < 2)
          continue;

        // Now lets calculate the disparity (assumes time array is monotonic) 时间数组是单调递增
        size_t camid = campairs.first;
        bool found0 = false;
        bool found1 = false;
        Eigen::Vector2f uv0 = Eigen::Vector2f::Zero();
        Eigen::Vector2f uv1 = Eigen::Vector2f::Zero();
        // 遍历该相机ID下该特征点的所有观测时间
        for (size_t idx = 0; idx < feat.second->timestamps.at(camid).size(); idx++) {
          double time = feat.second->timestamps.at(camid).at(idx);
          // 第一个条件：找时间窗口内的第一个观测
          if ((oldest_time == -1 || time > oldest_time) && !found0) {
            uv0 = feat.second->uvs.at(camid).at(idx).block(0, 0, 2, 1);
            found0 = true;
            continue;
          }
          // 第二个条件：找时间窗口内的最后一个观测(在uv0之后，且在时间上界之前)
          // 如果有多个满足要求条件的时间观测则会多次进入这里，然后就可以得到最后一个满足条件要求的观测
          if ((newest_time == -1 || time < newest_time) && found0) {
            uv1 = feat.second->uvs.at(camid).at(idx).block(0, 0, 2, 1);
            found1 = true;
            continue;
          }
        }

        // If we found both an old and a new time, then we are good!
        if (!found0 || !found1)
          continue;
        disparities.push_back((uv1 - uv0).norm()); // 像素视差值
      }
    }

    // If no disparities, just return
    if (disparities.size() < 2) {
      disp_mean = -1;
      disp_var = -1;
      total_feats = 0;
      // return; // 后面的统计没有意义了
    }

    // Compute mean and standard deviation in respect to it
    disp_mean = 0;
    for (double disp_i : disparities) {
      disp_mean += disp_i;
    }
    disp_mean /= (double)disparities.size();
    disp_var = 0;
    for (double &disp_i : disparities) {
      disp_var += std::pow(disp_i - disp_mean, 2);
    }
    disp_var = std::sqrt(disp_var / (double)(disparities.size() - 1)); // 标准差
    total_feats = (int)disparities.size();
  }

private:
  // Cannot construct this class
  FeatureHelper() {}
};

} // namespace ov_core

#endif /* OV_CORE_FEATURE_HELPER_H */