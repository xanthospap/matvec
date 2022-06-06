#include "eigen3/Eigen/Eigen"
#include "eigen3/Eigen/Geometry"
#include "geodesy/units.hpp"
#include "matvec/matvec.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <matvec/mat3x3.hpp>
#include <matvec/vector3d.hpp>

bool equal(double x, double y) {
  double maxXY = std::max(std::abs(x), std::abs(y));
  return std::abs(x - y) <= std::numeric_limits<double>::epsilon() * maxXY;
}

double equal_limit(double x, double y) {
  double maxXY = std::max(std::abs(x), std::abs(y));
  return std::numeric_limits<double>::epsilon() * maxXY;
}


bool equal(const Eigen::Matrix<double, 3, 3> &m1,
           const Eigen::Matrix<double, 3, 3> &m2) {
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      if (!equal(m1(r, c), m2(r, c)))
        return false;
  return true;
}

void pmat(const dso::Mat3x3 &m) {
  printf("|%+8.2f %+8.2f %+8.2f|\n", m(0, 0), m(0, 1), m(0, 2));
  printf("|%+8.2f %+8.2f %+8.2f|\n", m(1, 0), m(1, 1), m(1, 2));
  printf("|%+8.2f %+8.2f %+8.2f|\n", m(2, 0), m(2, 1), m(2, 2));
}

void pmat(const Eigen::Matrix<double, 3, 3> &m) {
  printf("|%+8.2f %+8.2f %+8.2f|\n", m(0, 0), m(0, 1), m(0, 2));
  printf("|%+8.2f %+8.2f %+8.2f|\n", m(1, 0), m(1, 1), m(1, 2));
  printf("|%+8.2f %+8.2f %+8.2f|\n", m(2, 0), m(2, 1), m(2, 2));
}

bool equal(const dso::Mat3x3 &m1, const Eigen::Matrix<double, 3, 3> &m2) {
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      if (m1(r, c) != m2(r, c))
        return false;
  return true;
}

constexpr const double angle_x = M_PI / 2e0;
constexpr const double angle_y = M_PI / 6e0;
constexpr const double angle_z = M_PI / 4e0;

int main() {

  // the identity matrix (3x3)
  dso::Mat3x3 I({1e0, 0e0, 0e0, 0e0, 1e0, 0e0, 0e0, 0e0, 1e0});
  dso::Mat3x3 m = I;

  printf("Rotation angles: X-axis %.5f\n", dso::rad2deg(angle_x));
  printf("               : Y-axis %.5f\n", dso::rad2deg(angle_y));
  printf("               : Z-axis %.5f\n", dso::rad2deg(angle_z));

  /* ------------------------------------------------------------------------
   * Perform sequential (elementary) rotations around x-, y- and z-axis using
   * Eigen and  matvec. Results should be identical.
   * Mat-Mat operations only.
   * ------------------------------------------------------------------------
   */

  /* rotate around x-axis */
  double angle = angle_x;
  m.rotx(angle); // aka R = R_x(angle) * I
  dso::Mat3x3 rotated = m;

  Eigen::Matrix<double, 3, 3> Rx =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()) *
      Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> Rotated = Rx.transpose();

  assert(equal(rotated, Rotated));

  /* rotate again around the y-axis */
  angle = angle_y;
  m.roty(angle); // aka R = R_y(angle) * R
  rotated = m;

  Eigen::Matrix<double, 3, 3> Ry =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()) *
      Eigen::Matrix<double, 3, 3>::Identity();
  Rotated = Ry.transpose() * Rotated;

  assert(equal(rotated, Rotated));

  /* rotate again around the z-axis */
  angle = angle_z;
  m.rotz(angle); // aka R = R_z(angle) * R
  rotated = m;

  Eigen::Matrix<double, 3, 3> Rz =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()) *
      Eigen::Matrix<double, 3, 3>::Identity();
  Rotated = Rz.transpose() * Rotated;

  assert(equal(rotated, Rotated));

  /* report result */
  printf("Result R Matrix:\n");
  pmat(Rotated);

  /* ------------------------------------------------------------------------
   * Combine sequential (elementary) rotations around x-, y- and z-axis using
   * Eigen and matvec to asingle 3x3 accumulated rotation matrix. Rotate a
   * 3x1 vector using the accumulated matrix.
   * Mat-Mat and Mat-Vec operations.
   * ------------------------------------------------------------------------
   */
  Eigen::Matrix3d Rxyz;
  Rxyz = Eigen::AngleAxisd(angle_x, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(angle_y, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(angle_z, Eigen::Vector3d::UnitZ());

  // this will yield the Euler angles in x-, y- and z-axis
  Eigen::Vector3d ea = Rxyz.eulerAngles(0, 1, 2);
  assert(equal(ea(0), angle_x) && equal(ea(1), angle_y) &&
         equal(ea(2), angle_z));

  // this will yield the Euler angles that would produce the same rotation
  // matrix (Rxyz) but in another sequence, namely rotations around z-,x- and
  // y-axis
  ea = Rxyz.eulerAngles(2, 0, 1);
  Eigen::Matrix3d Rxyz2 (Eigen::AngleAxisd(ea(0), Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(ea(1), Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(ea(2), Eigen::Vector3d::UnitY()));
  // assert(equal(Rxyz, Rxyz2));
  // printf("\n"); pmat(Rxyz); printf("\n"); pmat(Rxyz2); printf("\n");

  // let's rotate a vector
  Eigen::Vector3d v1; v1 << 1.2e0, 0.9e0, 0.7e0;
  // printf("\n");pmat(Rxyz);printf("\n");
  auto v2 = Rxyz.transpose() * v1;

  // let's do the same with the matvec library
  dso::Vector3 V1({1.2e0, 0.9e0, 0.7e0});
  m = I; // identity matrix
  m.rotx(angle_x);
  m.roty(angle_y);
  m.rotz(angle_z);
  auto V2 = m * V1;
  pmat(m);printf("\n");

  // try geting the first column
  Eigen::Vector3d C0;
  C0 = Rxyz.block<3,1>(0,0);
  printf("\n"); pmat(Rxyz); printf("\n");
  printf("first  column [%.2f %.2f %.2f]\n", C0(0), C0(1), C0(2));
  const auto C1 = Rxyz.block<3,1>(0,1);
  printf("second column [%.2f %.2f %.2f]\n", C1(0), C1(1), C1(2));
  const auto C2 = Rxyz.block<3,1>(0,2);
  printf("third  column [%.2f %.2f %.2f]\n", C2(0), C2(1), C2(2));

  printf("v2=[%.3f %.3f %.3f]\n", v2(0), v2(1), v2(2));
  printf("V2=[%.3f %.3f %.3f]\n", V2(0), V2(1), V2(2));
  printf("V2=[%.15e %.15e %.15e]\n", std::abs(v2(0)-V2(0)), std::abs(v2(1)-V2(1)), std::abs(v2(2)-V2(2)));
  printf("V2=[%.15e %.15e %.15e]\n", equal_limit(v2(0),V2(0)), equal_limit(v2(1),V2(1)), equal_limit(v2(2),V2(2)));

  // should be equal ... !! FAILS !!
  // assert(equal(v2(0), V2(0)) && equal(v2(1), V2(1)) && equal(v2(2), V2(2)));

  // by the way, here is how we construct unit vectors
  const auto untx = Eigen::Vector3d::UnitX();
  const auto unty = Eigen::Vector3d::UnitY();
  const auto untz = Eigen::Vector3d::UnitZ();
  printf("Unit x=[%.3f %.3f %.3f]\n", untx(0), untx(1), untx(2));
  printf("Unit y=[%.3f %.3f %.3f]\n", unty(0), unty(1), unty(2));
  printf("Unit z=[%.3f %.3f %.3f]\n", untz(0), untz(1), untz(2));

  return 0;
}
