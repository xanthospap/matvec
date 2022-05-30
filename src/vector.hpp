#ifndef __VECTOR_SIMPLE_HPP__
#define __VECTOR_SIMPLE_HPP__

#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#ifdef INCLUDE_EIGEN
#include "Eigen/Dense"
#endif

namespace dso {

struct Vector {
  std::vector<double> _vec;

  Vector(int sz, double val = 0e0) noexcept : _vec{sz, val} {};
  Vector(const Vector &other, int from, int to)
      : _vec{std::begin(other._vec) + from, std::begin(other._vec) + to} {};
  ~Vector() noexcept {};
  Vector(const Vector &other) noexcept : _vec{other._vec} {};
  Vector(Vector &&other) noexcept : _vec{std::move(other._vec)} {};
  Vector &operator=(const Vector &other) noexcept {
    if (other != *this)
      _vec = other._vec;
    return *this;
  }
  Vector &operator=(Vector &&other) noexcept {
    _vec = std::move(other._vec);
    return *this;
  }

  const std::vector<double> data() const noexcept { return _vec; }
  std::vector<double> data() noexcept { return _vec; }
  int cols() const noexcept { return _vec.size(); }
  double operator()(int i) const noexcept { return _vec[i]; }
  double operator()(int i) noexcept { return _vec[i]; }

  void append(const Vector &other) noexcept {
    _vec.reserve(_vec.size() + other._vec.size());
    _vec.insert(std::end(_vec), std::begin(other._vec), std::end(other._vec));
  }

  Vector slice(int from, int to) const noexcept {
    return Vector(*this, from, to);
  }

  double dot_product(const Vector &other) const noexcept {
    return std::inner_product(_vec.begin(), _vec.end(), other._vec.begin(),
                              0e0);
  }

#ifdef INCLUDE_EIGEN
  template<int N>
  Eigen::Matrix<double, N, 1> to_eigen() const noexcept {
    return Eigen::Matrix<double, N, 1>::Map(_vec.data(), N);
  }
  
  Eigen::VectorXcd to_eigen() const noexcept {
    return Eigen::VectorXcd::Map(_vec.data(), _vec.size());
  }
#endif

}; // Vector

} // dso

#endif
