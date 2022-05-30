#ifndef __MATVEC_3X3_SIMPLE_HPP__
#define __MATVEC_3X3_SIMPLE_HPP__

#include <cmath>
#include <cstring>
#include "vector3d.hpp"

namespace dso {

struct Mat3x3 {
  double data[9] = {1e0, 0e0, 0e0, 0e0, 1e0, 0e0, 0e0, 0e0, 1e0};

  constexpr double &operator()(int i, int j) noexcept {
    return data[i * 3 + j];
  }
  
  constexpr const double &operator()(int i, int j) const noexcept {
    return data[i * 3 + j];
  }

  /// @brief Multiply two 3x3 matrices (aka this * b)
  Mat3x3 operator*(const Mat3x3 &b) const noexcept;

  /// @brief Multiply a vector by a matrix.
  Vector3 operator*(const Vector3 &vec) const noexcept {
    double v0 =
        data[0] * vec.data[0] + data[1] * vec.data[1] + data[2] * vec.data[2];
    double v1 =
        data[3] * vec.data[0] + data[4] * vec.data[1] + data[5] * vec.data[2];
    double v2 =
        data[6] * vec.data[0] + data[7] * vec.data[1] + data[8] * vec.data[2];
    return Vector3{{v0, v1, v2}};
  }

  /// @brief Multiply two 3x3 matrices (aka this * b) and store result in
  ///        this instance
  void mult_inplace(const Mat3x3 &b) noexcept;

  /// @brief Transpose a 3x3 matric (in place)
  Mat3x3& transpose_inplace() noexcept;

  /// @brief Transpose a 3x3 matric
  Mat3x3 transpose() noexcept;

  /// @brief Set to identity matrix
  void set_identity() noexcept;

  /// @brief Rotate an r-matrix about the x-axis.
  /// This will actually perform the operation R = Rx * R, with Rx =
  ///  1        0            0
  ///  0   + cos(phi)   + sin(phi)
  ///  0   - sin(phi)   + cos(phi)
  /// @param[in] angle (radians)
  void rotx(double angle) noexcept;

  /// @brief Rotate an r-matrix about the y-axis.
  /// This will actually perform the operation R = Ry * R, with Rx =
  ///  + cos(phi)     0      - sin(phi)
  ///       0           1           0
  ///  + sin(phi)     0      + cos(phi)
  /// @param[in] angle (radians)
  void roty(double) noexcept;

  /// @brief Rotate an r-matrix about the z-axis.
  /// This will actually perform the operation R = Rz * R, with Rx =
  ///  + cos(psi)   + sin(psi)     0
  ///  - sin(psi)   + cos(psi)     0
  ///       0            0         1
  /// @param[in] angle (radians)
  void rotz(double) noexcept;
}; // Mat3x3
} // dso

#endif
