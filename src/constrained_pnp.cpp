#include <constrained_pnp.h>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <iostream>

// Returns the values of x which are roots of 
// 
//   y = a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0. 
std::array<std::optional<double>, 3> solve_cubic_roots(Eigen::Matrix<double, 4, 1> coeffs) {
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

// // Finds the set of coefficients [a_0, a_1, a_2, a_3, a_4] which fit the provided samples.
Eigen::Matrix<double, 5, 1> fit_quartic(Eigen::Matrix<double, 5, 1> xs, 
                                        Eigen::Matrix<double, 5, 1> ys) {
  Eigen::Matrix<double, 5, 5> A;
  for (int i = 0; i < 5; ++i) {
    double x = xs(i);
    A(i, 0) = 1.0;
    A(i, 1) = x;
    A(i, 2) = x * x;
    A(i, 3) = x * x * x;
    A(i, 4) = x * x * x * x;
  }
  // TODO: use a linear solver or smth smarter here???
  Eigen::Matrix<double, 5, 1> coeffs = A.inverse() * ys;
  return coeffs; // coeffs = A⁻¹y.;
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

  robot_x.SetValue(0);
  robot_z.SetValue(0);
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

  // TODO - can i just do this whole matrix at once, one col per observation?
  auto predicted_image_point = params.K * (R_T * world_points_opencv);

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

  // fmt::println("Final x: {}, z: {}, theta: {}", robot_x.Value(), robot_z.Value(), robot_θ.Value());

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
  //   1. Convert image points to normalized image points.
  //   2. Convert the world points into the opencv coordinate system.
  //   3. Formulate a cost function using a change of variables:
  // 
  //       tau = tan(theta / 2)
  //       x' = x * (1 + tau * tau)
  //       z' = z * (1 + tau * tau)
  // 
  //      Note this cost function is a fourth-order polynomial in terms of tau and second-
  //      order in terms of x' and y'.
  //   4. Solve for x' and y' in terms of tau and eliminate them so the cost function 
  //      becomes a fourth-order polynomial in terms of tau.
  //   5. Minimize the polynomial. This is possible by taking the derivative and finding
  //      the roots of the resulting third-degree polynomial. There is a closed form
  //      solution for finding the roots of cubic polynomials with Carbano's formula 
  //      (fancy quadratic formula).
  //   6. Undo the change of variables and solve for theta, x, and z in terms of tau.
  //   7. Undo the opencv transform and invert the transform.
  assert(params.imagePoints.cols() == params.worldPoints.cols());

  int N = params.imagePoints.cols();
  
  // Step 1
  auto K_inverse = params.K.inverse();
  Eigen::Matrix<double, 3, Eigen::Dynamic> normalized_image_points{3, N};
  for (int i = 0; i < N; ++i) {
    Eigen::Matrix<double, 3, 1> homogenous_image_point;
    homogenous_image_point(0, 0) = params.imagePoints(0, i);
    homogenous_image_point(1, 0) = params.imagePoints(1, i);
    homogenous_image_point(2, 0) = 1;
    Eigen::Matrix<double, 3, 1> normalized_image_point = K_inverse * homogenous_image_point;
    normalized_image_points(0, i) = normalized_image_point(0, 0);
    normalized_image_points(1, i) = normalized_image_point(1, 0);
    normalized_image_points(2, i) = normalized_image_point(2, 0);
  }

  // Step 2
  Eigen::Matrix<double, 4, 4> nwu_to_edn;
  nwu_to_edn <<
    0, -1,  0,  0,
    0,  0, -1,  0,
    1,  0,  0,  0,
    0,  0,  0,  1;
  auto world_points_opencv = nwu_to_edn * params.worldPoints;

  // Step 3
  auto cost = [N, normalized_image_points, world_points_opencv](double x_prime, double z_prime, double tau) -> double {
    Eigen::Matrix<double, 3, 3> R_bar;
    R_bar <<
        (1 - tau * tau),     0,          (2 * tau),
          0,                     1 + tau * tau,                      0,
           -(2 * tau),       0,       (1 - tau * tau);
    
    Eigen::Matrix<double, 3, 1> T;
    T << x_prime, 0, z_prime;
    
    double total_cost = 0;
    for (int i = 0; i < N; ++i) {
      Eigen::Matrix<double, 3, 1> u = normalized_image_points.block(0, i, 3, 1);
      Eigen::Matrix<double, 3, 1> P = world_points_opencv.block(0, i, 3, 1);

      Eigen::Matrix<double, 3, 1> projected_point = R_bar * P + T;

      double residual_x = projected_point(2) * u(0) - projected_point(0);
      double residual_y = projected_point(2) * u(1) - projected_point(1);
      total_cost += residual_x * residual_x + residual_y * residual_y;
    }
    return total_cost;
  };

  // Step 4. We want to find the optimal x' and z' value for each value of theta. It turns
  // out they are quadratic functions of tau, so let's plot some points and fit a curve.
  // Lucky us!
  std::array<double, 3> tau_samples = {0, 1, 2};
  std::array<double, 3> x_prime_samples;
  std::array<double, 3> z_prime_samples;

  for (int i = 0; i < 3; ++i) {
    double tau = tau_samples.at(i);
    // This is a bivariate quadratic 
    // 
    //   Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    // 
    // let's plot some more points...
    double c0 = cost(0, 0, tau);
    double c1 = cost(1, 0, tau);
    double c2 = cost(2, 0, tau);
    double c3 = cost(0, 1, tau);
    double c4 = cost(0, 2, tau);
    double c5 = cost(1, 1, tau);

    // printf("(c0: %f), (c1: %f), (c2: %f), (c3: %f), (c4: %f), (c5: %f)\n", c0, c1, c2, c3, c4, c5);

    double F = c0;
    double C = (c4 - 2 * c3 + c0) / 2;
    double A = (c2 - 2 * c1 + c0) / 2;
    double E = c3 - C - F;
    double D = c1 - A - F;
    double B = c5 - A - C - D - E - F;

    double det = 4 * A * C - B * B;
    double x_prime = (-2 * C * D + B * E) / det;
    double z_prime = (B * D - 2 * A * E) / det;
    x_prime_samples.at(i) = x_prime;
    z_prime_samples.at(i) = z_prime;
    // printf("(x_prime sample: %f), (z_prime sample: %f)\n", x_prime, z_prime);
  }

  // Fit samples to a quadratic.
  double C_x_prime = x_prime_samples.at(0);
  double B_x_prime = (4 * (x_prime_samples.at(1) - C_x_prime) - (x_prime_samples.at(2) - C_x_prime)) / 2;
  double A_x_prime = x_prime_samples.at(1) - C_x_prime - B_x_prime;
  double C_z_prime = z_prime_samples.at(0);
  double B_z_prime = (4 * (z_prime_samples.at(1) - C_z_prime) - (z_prime_samples.at(2) - C_z_prime)) / 2;
  double A_z_prime = z_prime_samples.at(1) - C_z_prime - B_z_prime;

  // printf("(C_x_prime: %f), (B_x_prime: %f), (A_x_prime: %f)\n", C_x_prime, B_x_prime, A_x_prime);
  // printf("(C_z_prime: %f), (B_z_prime: %f), (A_z_prime: %f)\n", C_z_prime, B_z_prime, A_z_prime);

  auto tau_cost = [&](double tau) -> double {
    double x_prime = A_x_prime * tau * tau + B_x_prime * tau + C_x_prime;
    double z_prime = A_z_prime * tau * tau + B_z_prime * tau + C_z_prime;
    return cost(x_prime, z_prime, tau);
  };
  
  // Fit the cost function in terms of tau to a quartic polynomial.
  Eigen::Matrix<double, 5, 1> samples, costs;
  for (int i = 0; i < 5; ++i) {
    samples(i) = i;
    costs(i) = tau_cost(i);
  }

  // std::cout << "tau samples:\n" << costs << std::endl;
  
  Eigen::Matrix<double, 5, 1> coeffs = fit_quartic(samples, costs);

  // std::cout << "Coeffs:\n" << coeffs << std::endl;

  // Step 5
  double tau = minimize_quartic(coeffs);

  // printf("Final tau is: %f\n", tau);

  // Step 6
  double x_prime = A_x_prime * tau * tau + B_x_prime * tau + C_x_prime;
  double z_prime = A_z_prime * tau * tau + B_z_prime * tau + C_z_prime;
  double x = x_prime / (1 + tau * tau);
  double z = z_prime / (1 + tau * tau);
  double theta = 2 * atan(tau);
  frc::Pose3d pose{frc::Translation3d(units::meter_t{x}, 0_m, units::meter_t{z}), 
                   frc::Rotation3d(0_rad, units::radian_t{theta}, 0_rad)};
  
  // fmt::println("Final x: {}, z: {}, theta: {}", x, z, theta);

  // Step 7
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