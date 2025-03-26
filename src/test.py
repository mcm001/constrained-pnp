import numpy as np
from planar_pnp import solve_planar_pnp, Strategy
import json

data = json.load(open('src/problem_setup.json'))

# Camera instrinsics
f_x, f_y, c_x, c_y = data["cameraCal"]
K = np.array([
    [f_x, 0,   c_x],
    [0,   f_y, c_y],
    [0,   0,   1]
])

# Image points
image_points = (np.array(data["point_observations"]["2DData"]).T).reshape(-1, 1, 2)

# World points
world_points = np.array(data["field2points"]["2DData"])

print(f"image points shape {image_points.shape}")
print(f"world points shape {world_points.shape}")

# Eliminate robot to camera transform
robot_to_camera = np.array(data["groundTruthRobot2camera"]["2DData"])
world_nwu = (np.linalg.inv(robot_to_camera) @ world_points)[:3].T

# Convert from NWU to EDN
nwu_to_edn = np.array([[0, -1,  0],
                        [0,  0, -1],
                        [1,  0,  0]])
edn_to_nwu = nwu_to_edn.T
world_cv = (nwu_to_edn @ world_nwu.T).T

strategies = [Strategy.POLYNOMIAL, Strategy.NAIVE, Strategy.OPENCV]

for strategy in strategies:
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    R_res, T_res = solve_planar_pnp(strategy, world_cv, image_points, K)

    # Invert the transformation to get the camera pose in the OpenCV coordinate frame.
    R_cam_cv = R_res.T
    t_cam_cv = -R_cam_cv @ T_res

    R = edn_to_nwu.T @ R_cam_cv @ edn_to_nwu #what the fuck
    T = edn_to_nwu @ t_cam_cv

    # Prints
    print("Rotation Matrix (", str(strategy), "):\n", R)
    print("Translation Vector (", str(strategy), "):\n", T)
    print("")