# This file contains several implementations of solvers for a modified version of the 
# perspective-n-point problem, which we will denote the planar pnp problem, which 
# constrains the solution pose to lie in SE(2).
from enum import Enum, auto

def solve_planar_pnp_naive(world_points, image_points, K):
  return 1

def solve_planar_pnp_polynomial(world_points, image_points, K):
  return 1

class Strategy(Enum):
  NAIVE = auto()
  POLYNOMIAL = auto()

def solve_planar_pnp(strategy, world_points, image_points, K):
  if strategy == Strategy.NAIVE:
    return solve_planar_pnp_naive(world_points, image_points, K)
  elif strategy == Strategy.POLYNOMIAL:
    return solve_planar_pnp_naive(world_points, image_points, K)
  else:
    raise Exception("Attempted to call solve_planar_pnp with an undefined strategy.")