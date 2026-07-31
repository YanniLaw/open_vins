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

#ifndef OV_CORE_GRIDER_GRID_H
#define OV_CORE_GRIDER_GRID_H

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils/opencv_lambda_body.h"

namespace ov_core {

/**
 * @brief Extracts FAST features in a grid pattern.
 *
 * As compared to just extracting fast features over the entire image,
 * we want to have as uniform of extractions as possible over the image plane.
 * Thus we split the image into a bunch of small grids, and extract points in each.
 * We then pick enough top points in each grid so that we have the total number of desired points.
 */
class Grider_GRID {

public:
  /**
   * @brief Compare keypoints based on their response value.
   * @param first First keypoint
   * @param second Second keypoint
   *
   * We want to have the keypoints with the highest values!
   * See: https://stackoverflow.com/a/10910921
   */
  static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) { return first.response > second.response; }

  /**
   * @brief This function will perform grid extraction using FAST.
   * 在图像上按网格均匀地提取 FAST 特征点。与全图一次性提取 FAST 不同，它把图像切分成 grid_x × grid_y 个网格单元，每个单元独立检测，
   * 再从每个单元挑选响应值最高的特征点，从而保证特征在图像平面上的空间分布尽可能均匀。
   * @param img Image we will do FAST extraction on
   * @param mask Region of the image we do not want to extract features in (255 = do not detect features)与原图大小相同的掩码，255=禁止检测，0=允许检测
   * @param valid_locs Valid 2d grid locations we will extract in (instead of the whole image) 允许提取新特征的粗网格单元位置
   * @param pts vector of extracted points we will return 新提取的特征点(亚像素坐标)
   * @param num_features max number of features we want to extract 一帧图片提取的特征点数量(包括已有的特征点和新提取的特征点)
   * @param grid_x size of grid in the x-direction / u-direction 粗网格单元在x方向的数量
   * @param grid_y size of grid in the y-direction / v-direction 粗网格单元在y方向的数量
   * @param threshold FAST threshold paramter (10 is a good value normally) FAST特征点检测阈值，越大检测到的特征点越少，越小检测到的特征点越多
   * @param nonmaxSuppression if FAST should perform non-max suppression (true normally) 是否进行非极大值抑制，通常为true
   *
   * Given a specified grid size, this will try to extract fast features from each grid.
   * It will then return the best from each grid in the return vector.
   */
  static void perform_griding(const cv::Mat &img, const cv::Mat &mask, const std::vector<std::pair<int, int>> &valid_locs,
                              std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold,
                              bool nonmaxSuppression) {

    // Return if there is nothing to extract
    if (valid_locs.empty())
      return;

    // We want to have equally distributed features
    // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
    // NOTE: Thus if we extract 1 point per grid we have
    // NOTE:    -> 1 = num_features / (grid_x * grid_y)
    // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
    // NOTE:    -> grid_y = sqrt(num_features / ratio)
    // 自适应调整网格数: 如果特征数不足以覆盖所有网格，缩小网格保证每格≥1点
    // 正常情况下每个网格单元至少提取1个特征点，但是如果总特征点数量不足以覆盖所有网格，那么平均每格不到1个特征点
    // 则根据总特征点数量和网格比例重新计算每个方向的网格数量(缩小网格，保持宽高比不变)，保证每个网格单元至少有一个特征点
    // 推导: num_features = grid_x * grid_y = (ratio * grid_y) * grid_y
    // 所以 grid_y = sqrt(num_features / ratio), grid_x = ratio * grid_y
    if (num_features < grid_x * grid_y) {
      double ratio = (double)grid_x / (double)grid_y;
      grid_y = std::ceil(std::sqrt(num_features / ratio));
      grid_x = std::ceil(grid_y * ratio);
    }
    // 每个粗网格单元"理想情况下"应该有多少个特征点
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    assert(grid_x > 0);
    assert(grid_y > 0);
    assert(num_features_grid > 0);

    // Calculate the size our extraction boxes should be 提取特征点的粗网格尺寸大小(一个粗网格对应的原图像素数目)
    int size_x = img.cols / grid_x;
    int size_y = img.rows / grid_y;

    // Make sure our sizes are not zero
    assert(size_x > 0);
    assert(size_y > 0);

    // Parallelize our 2d grid extraction!!
    /** cv::parallel_for_ 是一个自动化的任务分发器。它会把一个大的、独立的循环任务拆分成多个小块，
     *  再分给后台的线程池并行处理，最后再把结果组合起来。这个过程通常被称为 Fork-Join 模型。
     * OpenCV 不自己实现多线程，而是作为一个统一的接口。
     * 它会根据编译时的配置和你的运行环境，自动选择最合适的底层并行库，比如 Intel TBB、OpenMP、苹果的 GCD 等
     * void cv::parallel_for_(
        const cv::Range& range,                                 // [1] 循环范围
        const std::function<void(const cv::Range&)>& functor,   // [2] 循环体（Lambda 或函数对象）
        double nstripes = -1                                    // [3] 任务分割粒度（可选）
      );
     * 
     */
    // 经典的 fork-join + 结果分槽 模式，并行阶段无锁，合并阶段串行，兼顾性能和正确性
    // Fork：parallel_for_ 把 valid_locs 索引范围拆成小块分给线程池
    // 无锁并行：每个线程只往自己专属的 collection[r] 槽里写，没有数据竞争，不需要锁
    // Join：并行结束后，串行遍历合并
    std::vector<std::vector<cv::KeyPoint>> collection(valid_locs.size());
    parallel_for_(cv::Range(0, (int)valid_locs.size()), LambdaBody([&](const cv::Range &range) {
                    for (int r = range.start; r < range.end; r++) {

                      // Calculate what cell xy value we are in
                      auto grid = valid_locs.at(r); // 对应粗网格单元的位置(索引)
                      int x = grid.first * size_x;  // 粗网格单元左上角在原图中的像素坐标
                      int y = grid.second * size_y;

                      // Skip if we are out of bounds 检查粗网格单元是否越界
                      if (x + size_x > img.cols || y + size_y > img.rows)
                        continue;

                      // Calculate where we should be extracting from
                      // 这里其实就是一个粗网格单元大小
                      cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

                      // Extract FAST features for this part of the image，从ROI中提取特征点
                      // 注意: cv::FAST 在 ROI子图中检测，返回的是局部坐标!!!
                      std::vector<cv::KeyPoint> pts_new;
                      cv::FAST(img(img_roi), pts_new, threshold, nonmaxSuppression);

                      // Now lets get the top number from this 按响应值降序排序，取最强的几个
                      std::sort(pts_new.begin(), pts_new.end(), Grider_FAST::compare_response);

                      // Append the "best" ones to our vector
                      // Note that we need to "correct" the point u,v since we extracted it in a ROI
                      // So we should append the location of that ROI in the image
                      // 相当于取std::min(num_features_grid, pts_new.size())
                      // 保证每个网格提取的特征点数量不超过num_features_grid且不超过实际检测到的点数
                      // 每个单元最多取 num_features_grid 个点（按 FAST 响应值从高到低），
                      // 这就是"均匀性"的来源 —— 每个网格公平竞争，而不是全图按响应值全局排序（否则纹理丰富区域会垄断所有特征名额）。
                      for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++) {

                        // Create keypoint，加上ROI的偏移，得到在整张图像中的坐标
                        // 关键点：cv::FAST 在 ROI 子图上检测，返回的坐标是相对于子图左上角的局部坐标，必须加回 ROI 偏移(x, y)才是原图坐标
                        cv::KeyPoint pt_cor = pts_new.at(i);
                        pt_cor.pt.x += (float)x;
                        pt_cor.pt.y += (float)y;

                        // Reject if out of bounds (shouldn't be possible...) 越界检查，防止异常
                        if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x > img.cols || (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y > img.rows)
                          continue;

                        // Check if it is in the mask region 
                        // 掩码检查: 这里不在已经提取的特征点附近的膨胀掩码区域内才允许提取新特征点
                        // 在 perform_detection_monocular 函数的 while 循环里，每个保留的旧特征都画了膨胀矩形到 mask0_updated 上
                        // 新点 vs 旧特征之间的间距，确实已经由膨胀掩码保证了
                        // NOTE: mask has max value of 255 (white) if it should be removed
                        if (mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127) // 原图大小的掩码 mask0_updated
                          continue;
                        collection.at(r).push_back(pt_cor);
                      }
                    }
                  }));

    // Combine all the collections into our single vector
    for (size_t r = 0; r < collection.size(); r++) {
      pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
    }

    // Return if no points
    if (pts.empty())
      return;

    // Sub-pixel refinement parameters 亚像素细化
    // FAST 检测返回的角点坐标是整数像素级别的，精度有限。后续 KLT 光流跟踪对坐标精度要求很高，亚像素级别的精度能显著提升跟踪质量。
    // cornerSubPix 利用角点处"窗口内所有像素的梯度方向与该像素到角点的向量正交"这一约束，迭代求解亚像素位置
    cv::Size win_size = cv::Size(5, 5);     // 亚像素窗口大小
    cv::Size zero_zone = cv::Size(-1, -1);  // 不使用零区，即整个窗口都参与亚像素计算
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001); // 迭代最多 20 次，收敛阈值 0.001 像素

    // Get vector of points
    // 需要中间转换 Point2f 的原因是 cornerSubPix 不接受 KeyPoint 类型，只接受 Point2f，所以需要先提取坐标、精化后再写回。
    std::vector<cv::Point2f> pts_refined;
    for (size_t i = 0; i < pts.size(); i++) {
      pts_refined.push_back(pts.at(i).pt);
    }

    // Finally get sub-pixel for all extracted features
    // cornerSubPix 的原理是：在每个角点周围的窗口内，利用图像梯度正交约束迭代求解精确位置。数学上它要求真实角点处满足：
    // ∑▽I(qi)·(qi - p)^T = 0
    // 即窗口内所有像素的梯度向量与"像素位置到角点的向量"正交，通过迭代把 p 收敛到亚像素位置
    cv::cornerSubPix(img, pts_refined, win_size, zero_zone, term_crit);

    // Save the refined points!
    for (size_t i = 0; i < pts.size(); i++) {
      pts.at(i).pt = pts_refined.at(i);
    }
  }
};

} // namespace ov_core

#endif /* OV_CORE_GRIDER_GRID_H */
