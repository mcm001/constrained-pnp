#include <constrained_pnp.h>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <iostream>

// Returns the values of x which are roots of 
// 
//   y = a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0. 
std::array<std::optional<double>, 3> solve_cubic_roots(const Eigen::Matrix<double, 4, 1>& coeffs) {
  // // TODO: make sure this actually works, i was done with the rest and wanted to test so
  // // this is chatgpt
  std::array<std::optional<double>, 3> roots = { std::nullopt, std::nullopt, std::nullopt };
  const double tol = 1e-8;
  
  double a3 = coeffs(3);
  double a2 = coeffs(2);
  double a1 = coeffs(1);
  double a0 = coeffs(0);
  
  // Handle degenerate (quadratic or linear) cases.
  if (std::fabs(a3) < tol) {
      // Solve a2*x^2 + a1*x + a0 = 0.
      if (std::fabs(a2) < tol) {
          // Linear: a1*x + a0 = 0.
          if (std::fabs(a1) >= tol) {
              roots.at(0) = -a0 / a1;
          }
          return roots;
      } else {
          double discriminant = a1 * a1 - 4 * a2 * a0;
          if (discriminant >= 0) {
              double sqrt_disc = std::sqrt(discriminant);
              roots.at(0) = (-a1 + sqrt_disc) / (2 * a2);
              roots.at(1) = (-a1 - sqrt_disc) / (2 * a2);
          }
          return roots;
      }
  }
  
  // Normalize the cubic: x^3 + (a2/a3)*x^2 + (a1/a3)*x + (a0/a3) = 0.
  double b = a2 / a3;
  double c = a1 / a3;
  double d = a0 / a3;
  
  // Remove the quadratic term with the substitution: x = t - b/3.
  double offset = b / 3.0;
  double p = c - (b * b) / 3.0;
  double q = 2 * (b * b * b) / 27.0 - (b * c) / 3.0 + d;
  
  // Compute the discriminant for the depressed cubic: t^3 + p*t + q = 0.
  double discriminant = (q * q) / 4.0 + (p * p * p) / 27.0;
  
  if (discriminant > tol) {
      // One real root.
      double sqrt_disc = std::sqrt(discriminant);
      double u = std::cbrt(-q / 2.0 + sqrt_disc);
      double v = std::cbrt(-q / 2.0 - sqrt_disc);
      double t = u + v;
      roots.at(0) = t - offset;
  } else if (std::fabs(discriminant) <= tol) {
      // All roots real; at least two equal.
      double u = std::cbrt(-q / 2.0);
      roots.at(0) = 2 * u - offset;
      roots.at(1) = -u - offset;
  } else {
      // Three distinct real roots.
      double r = std::sqrt(-p * p * p / 27.0);
      double cos_phi = -q / (2 * r);
      // Clamp cos_phi to [-1, 1] to avoid numerical issues.
      if (cos_phi < -1) cos_phi = -1;
      if (cos_phi > 1)  cos_phi = 1;
      double phi = std::acos(cos_phi);
      double t1 = 2 * std::sqrt(-p / 3.0) * std::cos(phi / 3.0);
      double t2 = 2 * std::sqrt(-p / 3.0) * std::cos((phi + 2 * M_PI) / 3.0);
      double t3 = 2 * std::sqrt(-p / 3.0) * std::cos((phi + 4 * M_PI) / 3.0);
      roots.at(0) = t1 - offset;
      roots.at(1) = t2 - offset;
      roots.at(2) = t3 - offset;
  }
  
  return roots;
}

// Returns the value of x which minimizes 
// 
//   y = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0. 
// 
// Note we assume the polynomial has a finite value for the minimum.
double minimize_quartic(Eigen::Matrix<double, 5, 1> coeffs) {
  Eigen::Matrix<double, 4, 1> deriv;
  deriv(0) = coeffs(1);  // coefficient for x^3
  deriv(1) = 2 * coeffs(2);  // coefficient for x^2
  deriv(2) = 3 * coeffs(3);  // coefficient for x
  deriv(3) = 4 * coeffs(4);      // constant term
  auto critical_points = solve_cubic_roots(deriv);
  double min_x = 0;
  double min_y = INFINITY;
  for (const auto& opt_x : critical_points) {
    if (opt_x.has_value()) {
      double x = opt_x.value();
      double y = coeffs(4) * std::pow(x, 4) + 
                 coeffs(3) * std::pow(x, 3) + 
                 coeffs(2) * std::pow(x, 2) + 
                 coeffs(1) * x + 
                 coeffs(0);
      if (y < min_y) {
        min_x = x;
        min_y = y;
      }
    }
  }
  return min_x;
}

frc::Pose2d cpnp::solve_naive(const ProblemParams & params)
{
  using namespace cpnp;
  using namespace sleipnir;

  // Convert the points from the WPI coordinate system (NWU) to OpenCV's coordinate system 
  // (EDN).
  Eigen::Matrix<double, 4, 4> nwu_to_edn;
  nwu_to_edn <<
    0, -1,  0,  0,
    0,  0, -1,  0,
    1,  0,  0,  0,
    0,  0,  0,  1;
  auto world_points_opencv = nwu_to_edn * params.worldPoints;

  OptimizationProblem problem{};

  // robot pose
  auto robot_x = problem.DecisionVariable();
  auto robot_z = problem.DecisionVariable();
  auto robot_θ = problem.DecisionVariable();

  robot_x.SetValue(-1.5);
  robot_z.SetValue(-1);
  robot_θ.SetValue(0);

  // Generate r_t
  // rotation about +Y plus pose
  auto sinθ = sleipnir::sin(robot_θ);
  auto cosθ = sleipnir::cos(robot_θ);
  VariableMatrix R_T{
      {cosθ, 0, sinθ, robot_x},
      {0, 1, 0, 0},
      {-sinθ, 0, cosθ, robot_z},
  };

  Eigen::Matrix<double, 3, 3> K;
  K << params.f_x, 0, params.c_x, 0, params.f_y, params.c_y, 0, 0, 1;

  // TODO - can i just do this whole matrix at once, one col per observation?
  auto predicted_image_point = K * (R_T * world_points_opencv);

  auto u_pred = sleipnir::CwiseReduce(predicted_image_point.Row(0), predicted_image_point.Row(2), std::divides<>{});
  auto v_pred = sleipnir::CwiseReduce(predicted_image_point.Row(1), predicted_image_point.Row(2), std::divides<>{});

  Variable cost;
  for (int i = 0; i < u_pred.Cols(); ++i) {
    auto E_x = u_pred(0, i) - params.imagePoints(0, i);
    auto E_y = v_pred(0, i) - params.imagePoints(1, i);
    cost += E_x * E_x + E_y * E_y;
  }

  fmt::println("Initial cost: {}", cost.Value());
  fmt::println("Predicted corners:\n{}\n{}", u_pred.Value(), v_pred.Value());

  problem.Minimize(cost);

  auto status = problem.Solve({.diagnostics=false});

  fmt::println("Final cost: {}", cost.Value());

  frc::Pose3d pose{frc::Translation3d{units::meter_t{robot_x.Value()}, units::meter_t{0}, units::meter_t(robot_z.Value())}, 
                   frc::Rotation3d{units::radian_t{0}, units::radian_t{robot_θ.Value()}, units::radian_t{0}}};

  Eigen::Matrix3d transform;
  transform <<
    0, 0, 1,
    -1, 0, 0,
    0, -1, 0;
  frc::Rotation3d edn_to_nwu{transform};
  frc::Pose3d nwu_pose{pose.Translation().RotateBy(edn_to_nwu), 
                       -edn_to_nwu + pose.Rotation() + edn_to_nwu};

  frc::Pose3d inv_pose{-nwu_pose.Translation().RotateBy(-nwu_pose.Rotation()),
                       -nwu_pose.Rotation()};

  return inv_pose.ToPose2d();
}

frc::Pose2d cpnp::solve_polynomial(const ProblemParams& params) {
  using namespace cpnp;

  // The algorithm works as follows
  // 
  //   1. Formulate a cost function using a change of variables:
  // 
  //       tau = tan(theta / 2)
  //       x' = x * (1 + tau * tau)
  //       z' = z * (1 + tau * tau)
  // 
  //      Note this cost function is a fourth-order polynomial in terms of tau and second-
  //      order in terms of x' and y'.
  //   2. Solve for x' and y' in terms of tau and eliminate them so the cost function 
  //      becomes a fourth-order polynomial in terms of tau.
  //   3. Minimize the polynomial. This is possible by taking the derivative and finding
  //      the roots of the resulting third-degree polynomial. There is a closed form
  //      solution for finding the roots of cubic polynomials with Carbano's formula 
  //      (fancy quadratic formula).
  //   4. Undo the change of variables and solve for theta, x, and z in terms of tau.
  //   5. Undo the opencv transform and invert the transform.
  int N = params.imagePoints.cols();

  double f_x = params.f_x;
  double f_y = params.f_y;
  double c_x = params.c_x;
  double c_y = params.c_y;

  // Step 1
  // ok this looks unreadable but i swear it makes sense
  double a_400 = 0;
  double a_300 = 0;
  double a_200 = 0;
  double a_210 = 0;
  double a_201 = 0;
  double a_100 = 0;
  double a_110 = 0;
  double a_101 = 0;
  double a_020 = 0;
  double a_010 = 0;
  double a_011 = 0;
  double a_002 = 0;
  double a_001 = 0;
  double a_000 = 0;

  for (int i = 0; i < N; i++) {
    // Convert image points to normalized points
    double p_x = params.imagePoints(0, i);
    double p_y = params.imagePoints(1, i);
    double u = (p_x - c_x) / f_x;
    double v = (p_y - c_y) / f_y;
    
    // We do the opencv coordinate transform here to avoid an extra matrix multiply.
    double X = -params.worldPoints(1, i);
    double Y = -params.worldPoints(2, i);
    double Z = params.worldPoints(0, i);

    double Ax_tau = -Z * u + X;
    double Bx_tau = -2 * (X * u + Z);
    double Cx_tau = Z * u - X;

    double Ay_tau = -Z * v - Y;
    double By_tau = -2 * X * v;
    double Cy_tau = Z * v - Y;

    double Ax_xp = -1;
    double Ax_zp = u;

    double Ay_xp = 0;
    double Ay_zp = v;

    // Add the components from the x residual
    a_400 += Ax_tau * Ax_tau + Ay_tau * Ay_tau;
    a_300 += 2 * (Ax_tau * Bx_tau + Ay_tau * By_tau);
    a_200 += 2 * (Ax_tau * Cx_tau + Ay_tau * Cy_tau) + Bx_tau * Bx_tau + By_tau * By_tau;
    a_210 += 2 * (Ax_tau * Ax_xp + Ay_tau * Ay_xp);
    a_201 += 2 * (Ax_tau * Ax_zp + Ay_tau * Ay_zp);
    a_100 += 2 * (Bx_tau * Cx_tau + By_tau * Cy_tau);
    a_110 += 2 * (Bx_tau * Ax_xp + By_tau * Ay_xp);
    a_101 += 2 * (Bx_tau * Ax_zp + By_tau * Ay_zp);
    a_020 += Ax_xp * Ax_xp + Ay_xp * Ay_xp;
    a_010 += 2 * (Ax_xp * Cx_tau + Ay_xp * Cy_tau);
    a_011 += 2 * (Ax_xp * Ax_zp + Ay_xp * Ay_zp);
    a_002 += Ax_zp * Ax_zp + Ay_zp * Ay_zp;
    a_001 += 2 * (Ax_zp * Cx_tau + Ay_zp * Cy_tau);
    a_000 += Cx_tau * Cx_tau + Cy_tau * Cy_tau;
  }

  // Step 4. We want to find the optimal x' and z' value for each value of theta.
  // 
  // Taking the derivative of the cost function c(x, y, z) and setting it zero gives
  // 
  //   d/dy c(x,y,z) = a_210 * x² + a_110 * x + 2 * a_020 * y + a_010 + a_011 * z = 0
  //   d/dz c(x,y,z) = a_201 * x² + a_101 * x + 2 * a_002 * z + a_001 + a_011 * y = 0
  // 
  // Next we solve for y and z in terms of x in a linear system
  // 
  //   [2 * a_020    a_011  ][y] = [-(a_210 * x² + a_110 * x + a_010)]  
  //   [  a_011    2 * a_002][z]   [-(a_201 * x² + a_101 * x + a_001)]
  // 
  // This gives
  // 
  //   y = 1 / (4 * a_020 * a_002 - a_011 * a_011) * ((2 * a_002) * (-(a_210 * x² + a_110 * x + a_010)) + (-a_011) * (-(a_201 * x² + a_101 * x + a_001)))
  //   z = 1 / (4 * a_020 * a_002 - a_011 * a_011) * ((-a_011) * (-(a_210 * x² + a_110 * x + a_010)) + (2 * a_020) * (-(a_201 * x² + a_101 * x + a_001)))
  // 
  // Finally we can simplify this into constants
  // 
  //   y = A_y * x^2 + B_y * x + C_y
  //   z = A_z * x^2 + B_z * x + C_z
  double det = 4 * a_020 * a_002 - a_011 * a_011;

  double A_y = (-2 * a_002 * a_210 + a_011 * a_201) / det;
  double B_y = (-2 * a_002 * a_110 + a_011 * a_101) / det;
  double C_y = (-2 * a_002 * a_010 + a_011 * a_001) / det;

  double A_z = (a_011 * a_210 - 2 * a_020 * a_201) / det;
  double B_z = (a_011 * a_110 - 2 * a_020 * a_101) / det;
  double C_z = (a_011 * a_010 - 2 * a_020 * a_001) / det;

  // Substituting back in gives
  double b_4 = a_400 + a_210 * A_y + a_201 * A_z + a_020 * (A_y * A_y) + a_011 * (A_y * A_z) + a_002 * (A_z * A_z);
  double b_3 = a_300 + 
               a_210 * B_y + 
               a_201 * B_z + 
               a_110 * A_y + 
               a_101 * A_z + 
               a_011 * (A_y * B_z + B_y * A_z) + 
               a_002 * (A_z * B_z + B_z * A_z) +
               a_020 * (A_y * B_y + B_y * A_y);
  double b_2 = a_200 +
               a_210 * C_y +
               a_201 * C_z +
               a_110 * B_y +
               a_101 * B_z +
               a_020 * (B_y * B_y + A_y * C_y + C_y * A_y) +
               a_010 * A_y +
               a_011 * (A_y * C_z + B_y * B_z + C_y * A_z) +
               a_002 * (B_z * B_z + A_z * C_z + C_z * A_z) +
               a_001 * A_z;
  double b_1 = a_100 +
               a_020 * (B_y * C_y + C_y * B_y) +
               a_110 * C_y +
               a_101 * C_z +
               a_010 * B_y +
               a_011 * (B_y * C_z + C_y * B_z) +
               a_002 * (B_z * C_z + C_z * B_z) +
               a_001 * B_z;
  double b_0 = a_020 * (C_y * C_y) +
               a_010 * C_y +
               a_011 * (C_y * C_z) +
               a_002 * (C_z * C_z) +
               a_001 * C_z +
               a_000;
  
  const Eigen::Matrix<double, 5, 1> coeffs{b_0, b_1, b_2, b_3, b_4};

  // Step 5
  double tau = minimize_quartic(coeffs);

  // Step 6
  double x_prime = A_y * tau * tau + B_y * tau + C_y;
  double z_prime = A_z * tau * tau + B_z * tau + C_z;
  double x = x_prime / (1 + tau * tau);
  double z = z_prime / (1 + tau * tau);
  double theta = 2 * atan(tau);

  // Manually writing out the math instead of using wpilib geometry objects.
  double nwu_x = z;
  double nwu_y = -x;
  double nwu_theta = -theta;

  double ncos = cos(-nwu_theta);
  double nsin = sin(-nwu_theta);

  return frc::Pose2d{units::meter_t{-(ncos * nwu_x - nsin * nwu_y)}, 
                     units::meter_t{-(nsin * nwu_x + ncos * nwu_y)},
                     units::radian_t{-nwu_theta}};
}