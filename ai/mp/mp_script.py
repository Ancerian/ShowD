import numpy as np
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

import mediapipe as mp
import cv2


staticImageMode=False
modelComplexity=1
smoothLandmarks=True
enableSegmentation=False
smoothSegmentation=True
minDetectionConfidence=0.5
minTrackingConfidence=0.5

mpPose = mp.solutions.pose
pose = mpPose.Pose(staticImageMode, modelComplexity, smoothLandmarks, enableSegmentation, smoothSegmentation, minDetectionConfidence, minTrackingConfidence)

#print(in_data)

output = []

origin = in_data.domain["image"].attributes["origin"]

for img in in_data:
    image = cv2.imread(origin + "/" + img["image"].value)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    out_img = []
    for landmark in results.pose_landmarks.landmark:
        out_img.append(landmark.x)
        out_img.append(landmark.y)
    #out_img.append(img["category"].value)
    output.append(out_img)

output_arr = np.array(output)

domain = Domain([
    ContinuousVariable("0 - nose x"),
    ContinuousVariable("0 - nose y"),
    ContinuousVariable("1 - left eye (inner) x"),
    ContinuousVariable("1 - left eye (inner) y"),
    ContinuousVariable("2 - left eye x"),
    ContinuousVariable("2 - left eye y"),
    ContinuousVariable("3 - left eye (outer) x"),
    ContinuousVariable("3 - left eye (outer) y"),
    ContinuousVariable("4 - right eye (inner) x"),
    ContinuousVariable("4 - right eye (inner) y"),
    ContinuousVariable("5 - right eye x"),
    ContinuousVariable("5 - right eye y"),
    ContinuousVariable("6 - right eye (outer) x"),
    ContinuousVariable("6 - right eye (outer) y"),
    ContinuousVariable("7 - left ear x"),
    ContinuousVariable("7 - left ear y"),
    ContinuousVariable("8 - right ear x"),
    ContinuousVariable("8 - right ear y"),
    ContinuousVariable("9 - mouth (left) x"),
    ContinuousVariable("9 - mouth (left) y"),
    ContinuousVariable("10 - mouth (right) x"),
    ContinuousVariable("10 - mouth (right) y"),
    ContinuousVariable("11 - left shoulder x"),
    ContinuousVariable("11 - left shoulder y"),
    ContinuousVariable("12 - right shoulder x"),
    ContinuousVariable("12 - right shoulder y"),
    ContinuousVariable("13 - left elbow x"),
    ContinuousVariable("13 - left elbow y"),
    ContinuousVariable("14 - right elbow x"),
    ContinuousVariable("14 - right elbow y"),
    ContinuousVariable("15 - left wrist x"),
    ContinuousVariable("15 - left wrist y"),
    ContinuousVariable("16 - right wrist x"),
    ContinuousVariable("16 - right wrist y"),
    ContinuousVariable("17 - left pinky x"),
    ContinuousVariable("17 - left pinky y"),
    ContinuousVariable("18 - right pinky x"),
    ContinuousVariable("18 - right pinky y"),
    ContinuousVariable("19 - left index x"),
    ContinuousVariable("19 - left index y"),
    ContinuousVariable("20 - right index x"),
    ContinuousVariable("20 - right index y"),
    ContinuousVariable("21 - left thumb x"),
    ContinuousVariable("21 - left thumb y"),
    ContinuousVariable("22 - right thumb x"),
    ContinuousVariable("22 - right thumb y"),
    ContinuousVariable("23 - left hip x"),
    ContinuousVariable("23 - left hip y"),
    ContinuousVariable("24 - right hip x"),
    ContinuousVariable("24 - right hip y"),
    ContinuousVariable("25 - left knee x"),
    ContinuousVariable("25 - left knee y"),
    ContinuousVariable("26 - right knee x"),
    ContinuousVariable("26 - right knee y"),
    ContinuousVariable("27 - left ankle x"),
    ContinuousVariable("27 - left ankle y"),
    ContinuousVariable("28 - right ankle x"),
    ContinuousVariable("28 - right ankle y"),
    ContinuousVariable("29 - left heel x"),
    ContinuousVariable("29 - left heel y"),
    ContinuousVariable("30 - right heel x"),
    ContinuousVariable("30 - right heel y"),
    ContinuousVariable("31 - left foot index x"),
    ContinuousVariable("31 - left foot index y"),
    ContinuousVariable("32 - right foot index x"),
    ContinuousVariable("32 - right foot index y"),
], class_vars=in_data.domain.class_vars, metas=in_data.domain.metas)

out_data = Table.from_numpy(domain, output_arr, Y=in_data.Y, metas=in_data.metas)
