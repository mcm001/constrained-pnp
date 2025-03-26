# This file contains several implementations of solvers for a modified version of the 
# perspective-n-point problem, which we will denote the planar pnp problem, which 
# constrains the solution pose to lie in SE(2).
from enum import Enum, auto
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as Rot
import cv2
import math

def solve_planar_pnp_naive(world_points, image_points, K):
  N = len(world_points)

  def residuals(params, image_points, world_points, K):
    residuals = np.array([])

    for i in range(N):
      p = image_points[i,:].T
      P = np.array([world_points[i,:]]).T

      x, z, theta = params

      R = Rot.from_euler('y', theta).as_matrix()
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
  N = len(image_points)

  # First we convert to normalized coordinates
  homogenous_points = []
  for r in image_points:
    homogenous_points.append(np.hstack((r[0], [1])))
  homogenous_points = np.array(homogenous_points).T
  normalized_points = ((np.linalg.inv(K) @ homogenous_points).T)[:,:2]

  # Next we introduce a change of variables to the cost function.
  # 
  #   τ  = tan(θ / 2)
  #   x' = x(1 + τ²)
  #   z' = z(1 + τ²)
  def cost(x_prime, z_prime, tau):
    R_bar = np.array([
       [1 - tau * tau,   0,      2 * tau   ],
       [      0,         1,         0      ],
       [   -2 * tau,     0,   1 - tau * tau]
    ])

    T = np.array([x_prime, 0, z_prime])

    cost = 0

    for i in range(N):
      P_i = np.array([world_points[i,:3]]).T
      # The cost function is least-squares.
      u_i = normalized_points[i, 0]
      v_i = normalized_points[i, 1]
      E_x = (R_bar[0] @ P_i + T[0]) - u_i * (R_bar[2] @ P_i + T[2])
      E_y = (R_bar[1] @ P_i + T[1]) - v_i * (R_bar[2] @ P_i + T[2])
      cost += E_x ** 2 + E_y ** 2
    
    return cost
  
  # Next we find the optimal values of x' and z' in terms of tau. We know they are 
  # quadratic functions of tau, but working them out explicitly sucks. Let's just plot 
  # some points and fit a curve.
  tau_values = np.array([0, 1, 2])
  optimal_x_primes = []
  optimal_z_primes = []
  for tau in tau_values:
    # We want to compute the optimal x prime and z prime values for a fixed value of tau.
    # We can do this by sampling six points and fitting a bivariate quadratic.
    a = cost(0, 0, tau) # F
    b = cost(1, 0, tau) # A + D + F
    c = cost(2, 0, tau) # 4A + 2D + F
    d = cost(0, 1, tau) # C + E + F
    e = cost(0, 2, tau) # 4C + 2E + F
    f = cost(1, 1, tau) # A + B + C + D + E + F

    F = a
    C = (e - 2 * d + a) / 2
    A = (c - 2 * b + a) / 2
    E = d - C - F
    D = b - A - F
    B = f - A - C - D - E - F
    # Now we have the bivariate quadratic
    # 
    #   Ax² + Bxz + Cz² + Dx + Ez + F
    # 
    # The optimality conditions at the minimum are
    # 
    #   2Ax + Bz + D = 0
    #   2Cz + Bx + E = 0
    # 
    #   [2A  B][x] = [-D]
    #   [B  2C][z]   [-E]
    det = 4 * A * C - B * B
    x = (-2 * C * D + B * E) / det
    z = (B * D - 2 * A * E) / det
    optimal_x_primes.append(x)
    optimal_z_primes.append(z)

  # Fit quadratic functions to x prime and z prime.
  coeffs_x = np.polyfit(tau_values, optimal_x_primes, 2)
  coeffs_z = np.polyfit(tau_values, optimal_z_primes, 2)

  # Now we have the cost as a fourth degree polynomial. Again the math sucks. Let's plot
  # some points and fit a curve.
  def tau_cost(tau): 
     return cost(np.polyval(coeffs_x, tau)[0], np.polyval(coeffs_z, tau)[0], tau)
  
  tau_samples = np.array([0, 1, 2, 3, 4])
  tau_costs = np.array([tau_cost(t) for t in tau_samples])
  coeffs_tau = np.polyfit(tau_samples, tau_costs, 4)

  # Now we can find minimum points of the polynomial.
  coeffs_tau_derivative = (np.array([4 * coeffs_tau[0], 3 * coeffs_tau[1], 2 * coeffs_tau[2], 1 * coeffs_tau[3]]).T)[0]

  roots = np.roots(coeffs_tau_derivative.T)
  real_valued = roots.real[abs(roots.imag)<1e-5]
  tau = None
  minimum = math.inf
  for root in real_valued:
    if np.polyval(coeffs_tau_derivative, root) < minimum:
      tau = root
      minimum = np.polyval(coeffs_tau_derivative, root)
    
  # Finally extract x, z, and theta from tau
  x = np.polyval(coeffs_x, tau)[0] / (1 + tau * tau)
  z = np.polyval(coeffs_z, tau)[0] / (1 + tau * tau)
  theta = 2 * np.atan(tau)

  R = Rot.from_euler('y', theta).as_matrix()
  T = np.array([[x, 0, z]]).T
  return R, T

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