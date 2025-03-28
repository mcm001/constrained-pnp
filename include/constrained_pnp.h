#pragma once

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Transform3d.h>
#include <Eigen/Core>
#include <vector>

namespace cpnp {
    struct ProblemParams {
        // Homogonous world points, (x y z 1)^T
        Eigen::Matrix<double, 4, Eigen::Dynamic> worldPoints;
        // Image points, 
        Eigen::Matrix<double, 2, Eigen::Dynamic> imagePoints;
        Eigen::Matrix<double, 3, 3> K;
    };

    frc::Pose2d solve_naive(const ProblemParams& problem);
}
