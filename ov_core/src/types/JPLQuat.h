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

#ifndef OV_TYPE_TYPE_JPLQUAT_H
#define OV_TYPE_TYPE_JPLQUAT_H

#include "Type.h"
#include "utils/quat_ops.h"

namespace ov_type {

/**
 * @brief Derived Type class that implements JPL quaternion
 *
 * This quaternion uses a left-multiplicative error state and follows the "JPL convention".
 * Please checkout our utility functions in the quat_ops.h file.
 * We recommend that people new quaternions check out the following resources:
 * - http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
 * - ftp://naif.jpl.nasa.gov/pub/naif/misc/Quaternion_White_Paper/Quaternions_White_Paper.pdf
 *
 *
 * We need to take special care to handle edge cases when converting to and from other rotation formats.
 * All equations are based on the following tech report @cite Trawny2005TR :
 *
 * > Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter for 3D attitude estimation."
 * > University of Minnesota, Dept. of Comp. Sci. & Eng., Tech. Rep 2 (2005): 2005.
 * > http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
 *
 * @section jplquat_define JPL Quaternion Definition
 *
 * We define the quaternion as the following linear combination:
 * @f[
 *  \bar{q} = q_4+q_1\mathbf{i}+q_2\mathbf{j}+q_3\mathbf{k}
 * @f]
 * Where i,j,k are defined as the following:
 * @f[
 *  \mathbf{i}^2=-1~,~\mathbf{j}^2=-1~,~\mathbf{k}^2=-1
 * @f]
 * @f[
 *  -\mathbf{i}\mathbf{j}=\mathbf{j}\mathbf{i}=\mathbf{k}
 *  ~,~
 *  -\mathbf{j}\mathbf{k}=\mathbf{k}\mathbf{j}=\mathbf{i}
 *  ~,~
 *  -\mathbf{k}\mathbf{i}=\mathbf{i}\mathbf{k}=\mathbf{j}
 * @f]
 * As noted in @cite Trawny2005TR this does not correspond to the Hamilton notation, and follows the "JPL Proposed Standard Conventions".
 * The q_4 quantity is the "scalar" portion of the quaternion, while q_1, q_2, q_3 are part of the "vector" portion.
 * We split the 4x1 vector into the following convention:
 * @f[
 *  \bar{q} = \begin{bmatrix}q_1\\q_2\\q_3\\q_4\end{bmatrix} = \begin{bmatrix}\mathbf{q}\\q_4\end{bmatrix}
 * @f]
 * It is also important to note that the quaternion is constrained to the unit circle:
 * @f[
 *  |\bar{q}| = \sqrt{\bar{q}^\top\bar{q}} = \sqrt{|\mathbf{q}|^2+q_4^2} = 1
 * @f]
 *
 *
 * @section jplquat_errorstate Error State Definition
 *
 * It is important to note that one can prove that the left-multiplicative quaternion error is equivalent to the SO(3) error.
 * If one wishes to use the right-hand error, this would need to be implemented as a different type then this class!
 * This is noted in Eq. (71) in @cite Trawny2005TR .
 * Specifically we have the following:
 * \f{align*}{
 * {}^{I}_G\bar{q} &\simeq \begin{bmatrix} \frac{1}{2} \delta \boldsymbol{\theta} \\ 1 \end{bmatrix} \otimes {}^{I}_G\hat{\bar{q}}
 * \f}
 * which is the same as:
 * \f{align*}{
 * {}^{I}_G \mathbf{R} &\simeq \exp(-\delta \boldsymbol{\theta}) {}^{I}_G \hat{\mathbf{R}} \\
 * &\simeq (\mathbf{I} - \lfloor \delta \boldsymbol{\theta} \rfloor) {}^{I}_G \hat{\mathbf{R}} \\
 * \f}
 *
 */
class JPLQuat : public Type {
/**
 * @brief JPLQuat class
 *
 * 自己实现的JPL四元数类，继承自Type类，表示一个JPL四元数类型，用于表示旋转。
 * 该类包含了四元数的定义、更新方法以及与旋转矩阵的转换等功能。
 * JPL四元数的存储顺序为 [q1, q2, q3, q4]，其中q4是标量部分，q1、q2、q3是向量部分。
 * 该类还包含了一个左乘误差状态定义，表示四元数的误差状态与SO(3)误差状态之间的关系。
 */
public:
  JPLQuat() : Type(3) { // 四元数的误差状态是3维的，因为它与SO(3)误差状态等价，SO(3)误差状态是一个3维的轴角表示。
    Eigen::Vector4d q0 = Eigen::Vector4d::Zero();
    q0(3) = 1.0; // [0,0,0,1] is the identity rotation in JPL convention
    // 构造函数中不能安全调用虚函数(set_value)，因此我们直接调用内部函数来设置初始值和FEJ值。
    set_value_internal(q0);
    set_fej_internal(q0);
  }

  ~JPLQuat() {}

  /**
   * @brief Implements update operation by left-multiplying the current
   * quaternion with a quaternion built from a small axis-angle perturbation.
   *
   * @f[
   * \bar{q}=norm\Big(\begin{bmatrix} \frac{1}{2} \delta \boldsymbol{\theta}_{dx} \\ 1 \end{bmatrix}\Big) \otimes \hat{\bar{q}}
   * @f]
   *
   * @param dx Axis-angle representation of the perturbing quaternion
   */
  void update(const Eigen::VectorXd &dx) override {

    assert(dx.rows() == _size);

    // Build perturbing quaternion，论文formula 68 69 70
    // δq = [0.5*δθ, 1]^T
    Eigen::Matrix<double, 4, 1> dq;
    dq << .5 * dx, 1.0; // 构造扰动四元数，左乘误差状态定义中，误差状态是轴角的一半，所以这里乘以0.5
    dq = ov_core::quatnorm(dq);

    // Update estimate and recompute R
    // q_new = δq ⊗ q_old
    set_value(ov_core::quat_multiply(dq, _value));
  }

  /**
   * @brief 两层设计的职责分离：
   *  - set_value() 和 set_fej() 作为对外接口，支持多态调用，允许通过基类指针调用这些方法。
   *  - set_value_internal() 和 set_fej_internal() 作为内部函数，非虚函数，构造安全，直接设置值并计算旋转矩阵。这些函数不依赖于虚函数机制，因此在构造函数中调用是安全的。
   * 这种设计确保了在构造函数中初始化对象时不会调用虚函数，从而避免了潜在的未定义行为，同时仍然提供了对外接口的多态性。
   * 
   * @brief Sets the value of the estimate and recomputes the internal rotation matrix
   * @param new_value New value for the quaternion estimate (JPL quat as x,y,z,w)
   */
  void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

  /**
   * @brief Sets the fej value and recomputes the fej rotation matrix
   * @param new_value New value for the quaternion estimate (JPL quat as x,y,z,w)
   */
  void set_fej(const Eigen::MatrixXd &new_value) override { set_fej_internal(new_value); }
  // 返回基类指针，允许通过基类指针调用这些方法，实现多态性。
  std::shared_ptr<Type> clone() override {
    auto Clone = std::shared_ptr<JPLQuat>(new JPLQuat()); // 新对象
    Clone->set_value(value()); // 深拷贝
    Clone->set_fej(fej());
    return Clone;
  }

  /// Rotation access
  Eigen::Matrix<double, 3, 3> Rot() const { return _R; }

  /// FEJ Rotation access
  Eigen::Matrix<double, 3, 3> Rot_fej() const { return _Rfej; }

protected:
  // 为什么这里是存储旋转矩阵？ 这是一个性能与数值稳定性权衡的设计决策
  // 1. EKF 计算中旋转矩阵使用频率远高于四元数， 如果存储的是四元数，每次需要旋转矩阵时都要进行转换，效率较低。
  // 2. FEJ要求在整个滑动窗口生命周期内保持旋转矩阵不变，如果存储四元数，每次更新都需要重新计算旋转矩阵，增加了计算开销。
  // 为什么不用 Eigen::Quaterniond？
  // Eigen::Quaterniond 的存储顺序是 [x,y,z,w]（Hamilton），
  // 而 OpenVINS 用的是 JPL 约定 [x,y,z,w] 但语义不同（JPL 的 w 对应 Hamilton 的 -w 在某些操作上）。
  // 混用会引入难以排查的符号错误。
  // 整个代码库统一用 Eigen::Matrix<double, 4, 1> 存四元数，用 Eigen::Matrix<double, 3, 3> 存旋转矩阵，
  // 约定明确、不依赖 Eigen 内部的四元数语义。
  // 本质上是以空间换时间的缓存策略，_R 和 _Rfej 是 _value 和 _fej 的派生缓存，由 set_value_internal / set_fej_internal 保证始终同步。
  
  // Stores the rotation
  Eigen::Matrix<double, 3, 3> _R;

  // Stores the first-estimate rotation
  Eigen::Matrix<double, 3, 3> _Rfej;

  /**
   * @brief Sets the value of the estimate and recomputes the internal rotation matrix
   * @param new_value New value for the quaternion estimate
   */
  void set_value_internal(const Eigen::MatrixXd &new_value) {
    // 传入的是JPL四元数，存储顺序为 [x,y,z,w]
    assert(new_value.rows() == 4);
    assert(new_value.cols() == 1);

    _value = new_value;

    // compute associated rotation
    _R = ov_core::quat_2_Rot(new_value);
  }

  /**
   * @brief Sets the fej value and recomputes the fej rotation matrix
   * @param new_value New value for the quaternion estimate
   */
  void set_fej_internal(const Eigen::MatrixXd &new_value) {
    // 传入的是JPL四元数，存储顺序为 [x,y,z,w]
    assert(new_value.rows() == 4);
    assert(new_value.cols() == 1);

    _fej = new_value;

    // compute associated rotation
    _Rfej = ov_core::quat_2_Rot(new_value);
  }
};

} // namespace ov_type

#endif // OV_TYPE_TYPE_JPLQUAT_H
