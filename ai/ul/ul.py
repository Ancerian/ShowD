import cv2
import numpy as np
from ultralytics import YOLO
import os

# det_model = YOLO('yolo11x.pt')
pose_model = YOLO('yolo11n-pose.pt')


stream = cv2.VideoCapture(0)
# stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Function to draw pose keypoints and skeleton (COCO 17-keypoint format)
def draw_pose(image, keypoints_xy, keypoints_conf, thickness=2):
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image
    
    # COCO 17-keypoint skeleton (edges between keypoints)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    for person_idx, (kpts, confs) in enumerate(zip(keypoints_xy, keypoints_conf)):
        #kpts = kpts.cpu().numpy()  # Shape: (17, 2) [x, y]
        #confs = kpts = kpts.cpu().numpy()  # Shape: (17,) [confidence]
        
        # Draw keypoints
        for i, (x, y) in enumerate(kpts):
            if confs[i] > 0.5:  # Draw keypoints with sufficient confidence
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Draw skeleton lines
        for (start, end) in skeleton:
            if confs[start] > 0.5 and confs[end] > 0.5:
                start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, start_pt, end_pt, (255, 0, 0), thickness)
    
    return image

if not stream.isOpened():
    print("Cannot open camera")
    exit()

while True:
    (ret, frame) = stream.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # cv2.imshow('Object detector', frame)

    # resize to 140x79
    # frame = cv2.resize(frame, (140, 79))

    pose_results = pose_model(
        frame,
        conf=0.25,
        iou=0.45,
        classes=[0],
        device='cpu',
        half=True,
        verbose=False
    )

    # Extract keypoints and confidence scores
    keypoints_xy = []
    keypoints_conf = []
    for result in pose_results:
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy  # Shape: (num_persons, 17, 2) [x, y]
            keypoints_conf = result.keypoints.conf  # Shape: (num_persons, 17) [conf]
    print(keypoints_xy)

    # Draw pose keypoints and skeleton
    vis_image = draw_pose(frame, keypoints_xy, keypoints_conf)

    cv2.imshow('Object detector', vis_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break