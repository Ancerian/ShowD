import cv2
import numpy as np
import os
import mediapipe as mp


stream = cv2.VideoCapture(0)
# stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not stream.isOpened():
    print("Cannot open camera")
    exit()

staticImageMode=False
modelComplexity=1
smoothLandmarks=True
enableSegmentation=False
smoothSegmentation=True
minDetectionConfidence=0.5
minTrackingConfidence=0.5
mpPose = mp.solutions.pose
pose = mpPose.Pose(staticImageMode, modelComplexity, smoothLandmarks, enableSegmentation, smoothSegmentation, minDetectionConfidence, minTrackingConfidence)
mpDraw = mp.solutions.drawing_utils

while True:
    (ret, frame) = stream.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index
    
    if results.pose_landmarks:
        print(len(results.pose_landmarks))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
    cv2.imshow('Object detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
