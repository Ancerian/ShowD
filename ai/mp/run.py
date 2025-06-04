import cv2
import numpy as np
import os
import mediapipe as mp
import pickle
import time

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


# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkcls')
model = pickle.load(open(model_path, 'rb'))
classes = [cv.values for cv in model.domain.class_vars][0]
print(classes)

while True:
    (ret, frame) = stream.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    features = []
    if results.pose_landmarks:

        for landmark in results.pose_landmarks.landmark:
            features.append(landmark.x)
            features.append(landmark.y)
        
        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Make a prediction
        # features = np.array(features)

        sel_features = [
            features[22],
            features[23],
            features[24],
            features[25],
            features[26],
            features[27],
            features[28],
            features[29],
            features[30],
            features[31],
            features[32],
            features[33],

            features[46],
            features[47],
            features[48],
            features[49],
        ]

        out_data = np.array(sel_features).reshape(1, -1)
        
        out_data[:,2] = out_data[:,2] - out_data[:,0]
        out_data[:,4] = out_data[:,4] - out_data[:,0]
        out_data[:,6] = out_data[:,6] - out_data[:,0]
        out_data[:,8] = out_data[:,8] - out_data[:,0]
        out_data[:,10] = out_data[:,10] - out_data[:,0]
        out_data[:,12] = out_data[:,12] - out_data[:,0]
        out_data[:,14] = out_data[:,14] - out_data[:,0]

        out_data[:,1] = out_data[:,1] - out_data[:,0]
        out_data[:,3] = out_data[:,3] - out_data[:,0]
        out_data[:,5] = out_data[:,5] - out_data[:,0]
        out_data[:,7] = out_data[:,7] - out_data[:,0]
        out_data[:,9] = out_data[:,9] - out_data[:,0]
        out_data[:,11] = out_data[:,11] - out_data[:,0]
        out_data[:,13] = out_data[:,13] - out_data[:,0]
        out_data[:,15] = out_data[:,15] - out_data[:,0]

        out_data[:,0] = out_data[:,0] - out_data[:,0]
        out_data[:,1] = out_data[:,1] - out_data[:,1]

        prediction = model.predict(out_data[:,2:])
        # (array([2.]), array([[0.30498589, 0.19397866, 0.50103545]]))
        # print(prediction)
        # print the most likely class
        # print(f"Predicted class probabilities: {prediction}")
        print(f"Predicted class: {classes[int(prediction[0][0])]}")
        # print(prediction)
        time.sleep(0.1)

    cv2.imshow('Object detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
