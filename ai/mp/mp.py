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

    # cv2.imshow('Object detector', frame)

    # resize to 140x79
    # frame = cv2.resize(frame, (140, 79))
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    print(results.pose_landmarks)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
    cv2.imshow('Object detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
