#include <constrained_pnp.h>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>

frc::Pose2d cpnp::solve_naive(const ProblemParams & params)
{
  using namespace cpnp;
  using namespace sleipnir;

  OptimizationProblem problem{};

  // robot pose
  auto robot_x = problem.DecisionVariable();
  auto robot_z = problem.DecisionVariable();
  auto robot_θ = problem.DecisionVariable();

  robot_x.SetValue(4);
  robot_z.SetValue(-1);

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
  auto predicted_image_point = params.K * (R_T * params.worldPoints);

  auto u_pred = sleipnir::CwiseReduce(predicted_image_point.Row(0), predicted_image_point.Row(2), std::divides<>{});
  auto v_pred = sleipnir::CwiseReduce(predicted_image_point.Row(1), predicted_image_point.Row(2), std::divides<>{});

  Variable cost;
  for (const auto &u : u_pred)
    cost += u;
  for (const auto &v : v_pred)
    cost += v;

  fmt::println("Initial cost: {}", cost.Value());
  fmt::println("Predicted corners:\n{}\n{}", u_pred.Value(), v_pred.Value());

  problem.Minimize(cost);

  auto status = problem.Solve({.diagnostics=false});

  fmt::println("Final cost: {}", cost.Value());

  // TODO lmao this is not x,y ???
  return frc::Pose2d{units::meter_t{robot_x.Value()}, units::meter_t(robot_z.Value()), frc::Rotation2d(units::radian_t{robot_θ.Value()})};
}
