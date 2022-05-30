#ifndef __VECTOR_3X3_SIMPLE_HPP__
#define __VECTOR_3X3_SIMPLE_HPP__

#include <cmath>
#include <cstring>

namespace dso {

struct Vector3 {
  double data[3] = {0e0, 0e0, 0e0};
  constexpr const double &x() const noexcept { return data[0]; }
  constexpr const double &y() const noexcept { return data[1]; }
  constexpr const double &z() const noexcept { return data[2]; }
  constexpr double &x() noexcept { return data[0]; }
  constexpr double &y() noexcept { return data[1]; }
  constexpr double &z() noexcept { return data[2]; }

  static constexpr Vector3 to_vec3(const double *vec) noexcept {
    return Vector3{{vec[0], vec[1], vec[2]}};
  }

  constexpr void zero_out() noexcept { data[0] = data[1] = data[2] = 0e0; }

  constexpr double norm_squared() const noexcept {
    return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
  }

  constexpr double norm() const noexcept {
    return std::sqrt(this->norm_squared());
  }

  constexpr double dot_product(const Vector3 &v) const noexcept {
    return data[0] * v.data[0] + data[1] * v.data[1] + data[2] * v.data[2];
  }

  constexpr Vector3 cross_product(const Vector3 &v) const noexcept {
    const double s1 = data[1] * v.data[2] - data[2] * v.data[1];
    const double s2 = data[2] * v.data[0] - data[0] * v.data[2];
    const double s3 = data[0] * v.data[1] - data[1] * v.data[0];
    return Vector3{{s1, s2, s3}};
  }

  constexpr void cross_product(const Vector3 &v, Vector3 &out) const noexcept {
    out.data[0] = data[1] * v.data[2] - data[2] * v.data[1];
    out.data[1] = data[2] * v.data[0] - data[0] * v.data[2];
    out.data[2] = data[0] * v.data[1] - data[1] * v.data[0];
  }

  constexpr Vector3 normalize() const noexcept {
    const double norm = this->norm();
    return Vector3{{data[0] / norm, data[1] / norm, data[2] / norm}};
  }

  constexpr void normalize() noexcept {
    const double norm = this->norm();
    data[0] /= norm;
    data[1] /= norm;
    data[2] /= norm;
  }

  constexpr Vector3 operator-(const Vector3 &v) const noexcept {
    return {{data[0] - v.data[0], data[1] - v.data[1], data[2] - v.data[2]}};
  }

  constexpr Vector3 operator+(const Vector3 &v) const noexcept {
    return {{data[0] + v.data[0], data[1] + v.data[1], data[2] + v.data[2]}};
  }

  constexpr Vector3 operator/(double scalar) const noexcept {
    return {{data[0] / scalar, data[1] / scalar, data[2] / scalar}};
  }

  constexpr Vector3 operator*(double scalar) const noexcept {
    return {{data[0] * scalar, data[1] * scalar, data[2] * scalar}};
  }

  constexpr Vector3 &operator+=(const Vector3 &v) noexcept {
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return *this;
  }

  constexpr Vector3 &operator-=(const Vector3 &v) noexcept {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return *this;
  }

  constexpr Vector3 &operator*=(double scalar) noexcept {
    data[0] *= scalar;
    data[1] *= scalar;
    data[2] *= scalar;
    return *this;
  }

  constexpr Vector3 &operator/=(double scalar) noexcept {
    data[0] /= scalar;
    data[1] /= scalar;
    data[2] /= scalar;
    return *this;
  }

  constexpr bool operator==(const Vector3 &v) const noexcept {
    return ((data[0] == v.data[0]) &&
            (data[1] == v.data[1] && data[2] == v.data[2]));
  }

  constexpr bool operator!=(const Vector3 &v) const noexcept {
    return !(this->operator==(v));
  }
};

inline Vector3 operator*(double scalar, const Vector3 &vec) noexcept {
  return vec * scalar;
}

} // namespace dso
#endif
