#pragma once

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Transform3d.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>

namespace cpnp {
    struct ProblemParams {
        // Homogonous world points, (x y z 1)^T
        Eigen::Matrix<double, 4, Eigen::Dynamic> worldPoints;
        // Image points, 
        Eigen::Matrix<double, 2, Eigen::Dynamic> imagePoints;
        Eigen::Matrix<double, 3, 3> K;
        Eigen::Matrix<double, 3, 3> K_inverse;

        explicit ProblemParams(int nLandmarks, const Eigen::Matrix<double, 3, 3>& K)
        : worldPoints(4, nLandmarks),
          imagePoints(2, nLandmarks),
          K(K),
          K_inverse(K.inverse())
    {}    };

    frc::Pose2d solve_naive(const ProblemParams& problem);

    frc::Pose2d solve_polynomial(const ProblemParams& problem);
}
