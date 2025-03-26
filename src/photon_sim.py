from time import sleep
from typing import List
from ntcore import NetworkTableInstance
from photonlibpy import PhotonCamera
from photonlibpy.targeting import TargetCorner
from photonlibpy.simulation import PhotonCameraSim, VisionSystemSim, VisionTargetSim, SimCameraProperties
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Transform3d, Rotation3d, Translation3d
import numpy as np
import cv2
import numpy as np
from planar_pnp import solve_planar_pnp, Strategy

def solve(K: np.ndarray, image_points: np.ndarray, world_points: np.ndarray, robot_to_camera: np.ndarray):
    print(f"image points shape {image_points.shape}")
    print(f"world points shape {world_points.shape}")

    # Eliminate robot to camera transform
    world_nwu = (np.linalg.inv(robot_to_camera) @ world_points)[:3].T

    # Convert from NWU to EDN
    nwu_to_edn = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]])
    edn_to_nwu = nwu_to_edn.T
    world_cv = (nwu_to_edn @ world_nwu.T).T

    strategies = [Strategy.POLYNOMIAL, Strategy.NAIVE, Strategy.OPENCV]

    # TODO - undistort image points here, or consider converting to normalized image coordinates

    for strategy in strategies:
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

NetworkTableInstance.getDefault().stopServer()
NetworkTableInstance.getDefault().stopClient()
NetworkTableInstance.getDefault().startLocal()

cam = PhotonCamera("foo")
sim = PhotonCameraSim(cam)
sim.setMaxSightRange(999)
sim.setMinTargetAreaPercent(0.0001)

robotToCamera = Transform3d()

system = VisionSystemSim("simSystem")
system.addAprilTags(
    AprilTagFieldLayout.loadField(AprilTagField.kDefaultField),
)
system.addCamera(sim, robotToCamera)


robotPose = Pose3d(x=0, y=8, z=0.5, rotation=Rotation3d())
system.update(robotPose)

# And marshall into a sane format
result = cam.getLatestResult()
imagePoints: List[TargetCorner] = []  
worldPoints: List[Translation3d] = []
fiducialIDs = []
for target in result.getTargets():
    tgt: VisionTargetSim = next(
        (
            simTgt
            for simTgt in system.getVisionTargets("apriltag")
            if simTgt.fiducialId == target.fiducialId
        ),
        None,
    )
    if tgt is None:
        continue
    imagePoints += target.detectedCorners
    worldPoints += tgt.getFieldVertices()
    fiducialIDs.append((target.detectedCorners[0], target.fiducialId))

# And render imagePoints 
image = np.zeros((sim.prop.resHeight, sim.prop.resWidth, 3), dtype=np.uint8)

# Draw circles for each set of corners
for corner in imagePoints:
    x, y = int(corner.x), int(corner.y)
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circles with radius 5

# And draw the fiducial ID
for loc, id in fiducialIDs:
    cv2.putText(image, str(id), (int(loc.x), int(loc.y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)

# cv2.imshow("Detected Corners", image)
# cv2.waitKey(1)  # Update window with 1ms delay

print(f"Image points: {imagePoints}")
print(f"World points: {worldPoints}")

# convert arrays
imagePoints = np.array([(p.x, p.y) for p in imagePoints], dtype=np.float32).reshape(-1, 1, 2)
worldPoints = np.array([(p.x, p.y, p.z, 1) for p in worldPoints], dtype=np.float32).reshape(4, -1)

solve(
    sim.prop.getIntrinsics(),
    imagePoints, worldPoints, robotToCamera.toMatrix()
)

print(f"Expected robot to be at {robotPose}")
