from planar_pnp import Strategy, solve_planar_pnp
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Camera intrinsics
K = np.array([
    [600,  0,   300],
    [ 0,  600,  150],
    [ 0,   0,    1 ]
])

# World points
world_points = [
  np.array([[2.5, -0.08255, 0.5 - 0.08255]]).T,
  np.array([[2.5, -0.08255, 0.5 + 0.08255]]).T,
  np.array([[2.5, 0.08255, 0.5 + 0.08255]]).T,
  np.array([[2.5, 0.08255, 0.5 - 0.08255]]).T
]

# Image points
image_points = [
  np.array([[333, -17]]).T,
  np.array([[333, 83]]).T,
  np.array([[267, 83]]).T,
  np.array([[267, -17]]).T
]

R, T = solve_planar_pnp(Strategy.NAIVE, world_points, image_points, K)

print("R_x: ", round(Rot.from_matrix(R).as_euler('xyz')[0].item(), 3))
print("R_x: ", round(Rot.from_matrix(R).as_euler('xyz')[1].item(), 3))
print("R_x: ", round(Rot.from_matrix(R).as_euler('xyz')[2].item(), 3))
print("T_x: ", round(T[0, 0].item(), 3))
print("T_y: ", round(T[1, 0].item(), 3))
print("T_z: ", round(T[2, 0].item(), 3))