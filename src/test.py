from planar_pnp import Strategy, solve_planar_pnp
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import json

data = json.load(open('src/problem_setup.json'))

# Camera intrinsics
f_x = data["cameraCal"][0]
f_y = data["cameraCal"][1]
c_x = data["cameraCal"][2]
c_y = data["cameraCal"][3]
K = np.array([
    [f_x,  0,   c_x],
    [ 0,  f_y,  c_y],
    [ 0,   0,    1 ]
])

# World points
world_points = []
for (x, y, z) in zip(data["field2points"]["2DData"][0],
                     data["field2points"]["2DData"][1],
                     data["field2points"]["2DData"][2]):
  world_points.append(np.array([[x, y, z]]).T)

# Image points
image_points = []
for (u, v) in zip(data["point_observations"]["2DData"][0],
                     data["point_observations"]["2DData"][1]):
  image_points.append(np.array([[u, v]]).T)

# Robot to camera
robot_to_camera = np.array(data["groundTruthRobot2camera"]["2DData"])

# First we need to transform the world points into the camera coordinate system.
world_points_cam = []
for pt in world_points:
    # Convert point to homogeneous coordinates by appending a 1
    pt_homogeneous = np.vstack((pt, np.ones((1, 1))))
    # Apply the robot-to-camera transformation
    pt_cam_homogeneous = robot_to_camera @ pt_homogeneous
    # Convert back to 3D (drop the homogeneous coordinate)
    world_points_cam.append(pt_cam_homogeneous[:3])

R, T = solve_planar_pnp(Strategy.NAIVE, world_points_cam, image_points, K)

print("R_x: ", round(Rot.from_matrix(R).as_euler('xyz')[0].item(), 3))
print("R_x: ", round(Rot.from_matrix(R).as_euler('xyz')[1].item(), 3))
print("R_x: ", round(Rot.from_matrix(R).as_euler('xyz')[2].item(), 3))
print("T_x: ", round(T[0, 0].item(), 3))
print("T_y: ", round(T[1, 0].item(), 3))
print("T_z: ", round(T[2, 0].item(), 3))