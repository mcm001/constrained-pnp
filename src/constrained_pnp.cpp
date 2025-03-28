#include <constrained_pnp.h>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>

using namespace cpnp;
using namespace sleipnir;

frc::Pose2d solve_naive(const ProblemParams params)
{
  OptimizationProblem problem{};

  // robot pose
  auto robot_x = problem.DecisionVariable();
  auto robot_y = problem.DecisionVariable();
  auto robot_θ = problem.DecisionVariable();

  // Generate r_t
  // rotation about +Y plus pose
  auto sinθ = sleipnir::sin(robot_θ);
  auto cosθ = sleipnir::cos(robot_θ);
  VariableMatrix R_T{
      {cosθ, 0, sinθ, robot_x},
      {0, 1, robot_y},
      {-sinθ, 0, cosθ, 0},
  };

  // TODO - can i just do this whole matrix at once, one col per observation?
  auto predicted_image_point = params.K * R_T * params.worldPoints;

  auto u_pred = sleipnir::CwiseReduce(predicted_image_point.Row(0), predicted_image_point.Row(2), std::divides<>{});
  auto v_pred = sleipnir::CwiseReduce(predicted_image_point.Row(1), predicted_image_point.Row(2), std::divides<>{});

  Variable cost;
  for (const auto &u : u_pred)
    cost += u;
  for (const auto &v : v_pred)
    cost += v;

  problem.Minimize(cost);

  auto status = problem.Solve();

  return frc::Pose2d{units::meter_t{robot_x.Value()}, units::meter_t(robot_y.Value()), frc::Rotation2d(units::radian_t{robot_θ.Value()})};
}
