from time import sleep
from typing import List
from ntcore import NetworkTableInstance
from photonlibpy import PhotonCamera
from photonlibpy.targeting import TargetCorner
from photonlibpy.simulation import PhotonCameraSim, VisionSystemSim, VisionTargetSim, SimCameraProperties
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Transform3d, Rotation3d
import numpy as np
import cv2

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


while True:

    robotPose = Pose3d(x=0, y=8, z=0.5, rotation=Rotation3d())
    system.update(robotPose)

    # And marshall into a sane format
    result = cam.getLatestResult()
    imagePoints: List[TargetCorner] = []  
    worldPoints = []
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
    
    cv2.imshow("Detected Corners", image)
    cv2.waitKey(1)  # Update window with 1ms delay

    print(f"Image points: {imagePoints}")
    print(f"World points: {worldPoints}")

    sleep(0.1)
