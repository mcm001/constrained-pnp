# This file contains several implementations of solvers for a modified version of the 
# perspective-n-point problem, which we will denote the planar pnp problem, which 
# constrains the solution pose to lie in SE(2).
from enum import Enum, auto
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as Rot
import cv2

def solve_planar_pnp_naive(world_points, image_points, K):
  N = len(world_points)

  def residuals(params, image_points, world_points, K):
    residuals = np.array([])

    for i in range(N):
      p = image_points[i,:].T
      P = np.array([world_points[i,:]]).T

      x, z, theta = params

      R = np.array([
        [ np.cos(theta),   0,   np.sin(theta)],
        [       0,         1,          0     ],
        [-np.sin(theta),   0,   np.cos(theta)],
      ])

      T = np.array([[x, 0, z]]).T

      # [R | T]
      R_T = np.hstack((R, T))

      # Convert world point to homogenous world point
      h_world_point = np.vstack((P, np.array([[1]])))

      # Pinhole camera model
      predicted_image_point = (K @ R_T @ h_world_point)

      u_pred = predicted_image_point[0, 0] / predicted_image_point[2, 0]
      v_pred = predicted_image_point[1, 0] / predicted_image_point[2, 0]
      
      # Residuals: differences between predicted and measured normalized image points.
      res_u = u_pred - p[0, 0]
      res_v = v_pred - p[1, 0]

      residuals = np.concat((residuals, np.array([res_u, res_v])))

    return residuals

  
  # Solve the least-squares problem to minimize the residuals
  init_params = np.array([0, 0, 0])
  result = scipy.optimize.least_squares(residuals, 
                                        init_params, 
                                        args=(image_points, world_points, K))
  x, z, theta = result.x

  R = Rot.from_euler('y', theta).as_matrix()
  T = np.array([[x, 0, z]]).T
  return R, T

def solve_planar_pnp_polynomial(world_points, 
                                image_points, 
                                K):
  raise Exception("Polynomial planar pnp not implemented")

def solve_planar_pnp_opencv(world_points, image_points, K):
  _, rvec, tvec = cv2.solvePnP(world_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
  R_cv, _ = cv2.Rodrigues(rvec)
  T_cv = tvec
  return R_cv, T_cv

class Strategy(Enum):
  NAIVE = auto()
  POLYNOMIAL = auto()
  OPENCV = auto()

def solve_planar_pnp(strategy, 
                     world_points, 
                     image_points, 
                     K):
  '''
  Solves the planar perspective-n-point problem which finds the pose of the camera given
  a set of world points and corresponding image points.
  
  The camera coordinate system is X right, Y down, and Z forwards. The solution is
  must have y = 0, rotation around x-axis = 0, and rotation around z-axis = 0.

  Args:
    strategy (Strategy)
    world_points (3xN np.array)
    image_points (2xN np.array)
    K (3x3 np.array)

  Returns:
    R, T (3x3 np.array, 3x1 np.array)
  '''
  # Size assertions for matrices
  for p in world_points:
    assert p.shape[0] == 3
  for p in image_points:
    assert p.shape[1] == 2
  assert K.shape == (3, 3)

  if strategy == Strategy.NAIVE:
    return solve_planar_pnp_naive(world_points, image_points, K)
  elif strategy == Strategy.POLYNOMIAL:
    return solve_planar_pnp_polynomial(world_points, image_points, K)
  elif strategy == Strategy.OPENCV:
    return solve_planar_pnp_opencv(world_points, image_points, K)
  else:
    raise Exception("Attempted to call solve_planar_pnp with an undefined strategy.")