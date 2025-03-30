/*
 * MIT License
 *
 * Copyright (c) PhotonVision
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <constrained_pnp.h>
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <frc/geometry/Transform2d.h>
#include <frc/geometry/Translation2d.h>
#include <units/length.h>

/// @brief Get the positions of the corners of a tag relative to the tag origin, in homogeneous coordinates.
/// @return A 4x4 matrix where the first three rows are the x, y, and z coordinates of the corners, and the last row is all ones.
Eigen::Matrix4d tagCenterToCorners() {
  const units::meter_t width{6.0_in};
  const units::meter_t height{6.0_in};

  Eigen::Matrix<double, 4, 3> corners {};
  corners << 0, -width.value(), -height.value(), 
             0, width.value(), -height.value(), 
             0, width.value(), height.value(), 
             0, -width.value(), height.value();

  Eigen::Matrix4d ret {};
  ret.block(0, 0, 3, 4) = corners.transpose();
  ret.row(3) = Eigen::Matrix<double, 1, 4>::Ones();

  return ret;
}

/// @brief Get a list of test tags' corners in homogeneous coordinates. The locations of these tags is hard-coded, because I'm lazy
/// @return A 4xN matrix where the first three rows are the x, y, and z coordinates of the corners, and the last row is all ones.
Eigen::Matrix<double, 4, Eigen::Dynamic> getTestTags() {
  // change me to add more tags
  Eigen::Matrix<double, 4, Eigen::Dynamic> ret(4, 8);

  auto tagCorners = tagCenterToCorners();

  // Add all the corners of tag 1, located at (2, 0, 1) and rotated 180 degrees
  // about +Z
  auto tag1pose = frc::Pose3d{frc::Translation3d{2_m, 0_m, 1_m},
                              frc::Rotation3d{0_deg, 0_deg, 180_deg}}
                      .ToMatrix();

  ret.block(0, 0, 4, 4) = tag1pose * tagCorners;

  // Add all the corners of tag 2, located at (2, 1, 1) and rotated 180 degrees
  // about +Z
  auto tag2pose = frc::Pose3d{frc::Translation3d{2_m, 1_m, 1_m},
                              frc::Rotation3d{0_deg, 0_deg, 180_deg}}
                      .ToMatrix();

  ret.block(0, 4, 4, 4) = tag2pose * tagCorners;

  return ret;
}


/// @brief Project the corners of the tags into the camera frame.
/// @param K OpenCV camera calibration matrix
/// @param field2camera_wpi The location of the camera in the field frame. This is the "wpi" camera pose, with X forward, Y left, and Z up.
/// @param field2corners The locations of the corners of the tags in the field frame
/// @return Observed pixel locations
Eigen::Matrix<double, 2, Eigen::Dynamic>
projectPoints(Eigen::Matrix<double, 3, 3> K,
              Eigen::Matrix4d field2camera_wpi,
              Eigen::Matrix<double, 4, Eigen::Dynamic> field2corners) {
  // robot is ENU, cameras are SDE
  Eigen::Matrix4d camera2opencv{
      {0, 0, 1, 0},
      {-1, 0, 0, 0},
      {0, -1, 0, 0},
      {0, 0, 0, 1},
  };
  Eigen::Matrix4d field2camera = field2camera_wpi * camera2opencv;

  // transform the points to camera space
  auto camera2corners = field2camera.transpose() * field2corners; 

  // project the points. This is verbose but whatever
  auto pointsUnnormalized =
      K * camera2corners.block(0, 0, 3, camera2corners.cols());
  auto u =
      pointsUnnormalized.row(0).array() / pointsUnnormalized.row(2).array();
  auto v =
      pointsUnnormalized.row(0).array() / pointsUnnormalized.row(2).array();

  Eigen::Matrix<double, 2, Eigen::Dynamic> ret(2, camera2corners.cols());
  ret.row(0) = u;
  ret.row(1) = v;
  return ret;
}

TEST(PoseTest, Projection) {
  cpnp::ProblemParams params(4);

  // params.K << 599.375, 0., 479.5, 0., 599.16666667, 359.5, 0., 0., 1.;
  params.K << 100, 0., 0, 0., 100, 0, 0., 0., 1.;

  params.worldPoints = getTestTags();

  frc::Transform3d robot2camera {};
  params.imagePoints = projectPoints(params.K, robot2camera.ToMatrix(), params.worldPoints);

  std::cout << "world points:\n" << params.worldPoints << std::endl;
  std::cout << "image points:\n" << params.imagePoints << std::endl;
  std::cout << "K:\n" << params.K << std::endl;
}

TEST(PoseTest, Naive) {
  cpnp::ProblemParams params(4);

  params.worldPoints << 2.5, 2.5, 2.5, 2.5, 0 - 0.08255, 0 - 0.08255,
      0 + 0.08255, 0 + 0.08255, 0.5 - 0.08255, 0.5 + 0.08255, 0.5 + 0.08255,
      0.5 - 0.08255, 1, 1, 1, 1;

  std::cout << params.worldPoints << std::endl;

  /*
  world_points
  array([[-4.38785   , -0.14123904,  3.96938119],
         [-4.22275   , -0.14123904,  3.96938119],
         [-4.22275   , -0.3054346 ,  3.95212354],
         [-4.38785   , -0.3054346 ,  3.95212354],
         [-3.626993  , -0.17728208,  4.31230785],
         [-3.544443  , -0.19222765,  4.45450538],
         [-3.544443  , -0.35642321,  4.43724773],
         [-3.626993  , -0.34147765,  4.2950502 ],
         [-0.27559   , -1.38161781,  6.2647928 ],
         [-0.27559   , -1.36436016,  6.10059724],
         [-0.27559   , -1.52855573,  6.08333959],
         [-0.27559   , -1.54581338,  6.24753515],
         [-5.066157  , -0.19222765,  4.45450538],
         [-4.983607  , -0.17728208,  4.31230785],
         [-4.983607  , -0.34147765,  4.2950502 ],
         [-5.066157  , -0.35642321,  4.43724773],
         [-2.276856  , -2.19018622,  8.43562142],
         [-2.111756  , -2.19018622,  8.43562142],
         [-2.111756  , -2.32375492,  8.33857807],
         [-2.276856  , -2.32375492,  8.33857807],
         [-6.499606  , -2.19018622,  8.43562142],
         [-6.334506  , -2.19018622,  8.43562142],
         [-6.334506  , -2.32375492,  8.33857807],
         [-6.499606  , -2.32375492,  8.33857807],
         [-4.38785   , -1.03701724, 12.49214144],
         [-4.22275   , -1.03701724, 12.49214144],
         [-4.22275   , -1.2012128 , 12.47488379],
         [-4.38785   , -1.2012128 , 12.47488379]])
  */

  params.imagePoints << 333, 333, 267, 267, -17, -83, -83, -17;

  /*
  image_points
  array([[[401.35592651, 352.10610962]],
         [[434.62026978, 352.10610962]],
         [[434.35839844, 318.79934692]],
         [[400.89996338, 318.79934692]],
         [[546.88555908, 346.36123657]],
         [[558.41638184, 344.31311035]],
         [[558.81201172, 315.66061401]],
         [[547.23791504, 316.48504639]],
         [[903.06799316, 214.3102417 ]],
         [[916.68865967, 211.66645813]],
         [[918.17126465, 191.83242798]],
         [[904.45953369, 195.1048584 ]],
         [[294.80895996, 344.31311035]],
         [[301.80636597, 346.36123657]],
         [[300.8772583 , 316.48504639]],
         [[293.88314819, 315.66061401]],
         [[618.29797363, 191.56022644]],
         [[631.59661865, 191.56022644]],
         [[633.60644531, 178.44384766]],
         [[620.13201904, 178.44384766]],
         [[278.15863037, 191.56022644]],
         [[291.45727539, 191.56022644]],
         [[288.97250366, 178.44384766]],
         [[275.49810791, 178.44384766]],
         [[459.28125   , 310.90591431]],
         [[467.88793945, 310.90591431]],
         [[467.8704834 , 302.26342773]],
         [[459.25085449, 302.26342773]]])
  */

  params.K << 599.375, 0., 479.5, 0., 599.16666667, 359.5, 0., 0., 1.;

  /*
  K
  array([[599.375     ,   0.        , 479.5       ],
         [  0.        , 599.16666667, 359.5       ],
         [  0.        ,   0.        ,   1.        ]])
  */

  auto ret = cpnp::solve_polynomial(params);

  fmt::println("Polynomial method says robot is at:\n{}", ret.ToMatrix());

  ret = cpnp::solve_naive(params);

  fmt::println("Naive method says robot is at:\n{}", ret.ToMatrix());
}
