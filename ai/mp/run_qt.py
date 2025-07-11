import sys
import cv2
import numpy as np
import os
import mediapipe as mp
import pickle

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# --- Ваша логика инициализации модели и функций ---
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

model_path = os.path.join(os.path.dirname(__file__), 'model.pkcls')
model = pickle.load(open(model_path, 'rb'))
classes = [cv.values for cv in model.domain.class_vars][0]

def recognize_action(imgRGB):
    results = pose.process(imgRGB)
    features = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            features.append(landmark.x)
            features.append(landmark.y)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(imgRGB, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        sel_features = [
            features[22], features[23], features[24], features[25],
            features[26], features[27], features[28], features[29],
            features[30], features[31], features[32], features[33],
            features[46], features[47], features[48], features[49],
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
        return prediction, imgRGB
    return None, imgRGB

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Action Recognition PyQt5")
        self.resize(900, 700)
        self.setMinimumSize(400, 300)
        self.setMaximumSize(1200, 900)

        # Белый фон
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("white"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Видео-рамка
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.Box)
        self.image_frame.setLineWidth(2)
        self.image_frame.setStyleSheet("background-color: #fafafa; border-radius: 10px;")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_layout)

        # Текст класса
        self.text_label = QLabel("Predicted class: ...")
        self.text_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color: #222; margin-top: 20px; margin-bottom: 10px;")

        # Главный layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        main_layout.addWidget(self.image_frame, stretch=5)
        main_layout.addWidget(self.text_label, stretch=1)
        self.setLayout(main_layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction, imgRGB_draw = recognize_action(imgRGB.copy())
        h, w, ch = imgRGB_draw.shape
        bytes_per_line = ch * w
        qt_img = QImage(imgRGB_draw.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        if prediction is not None:
            self.text_label.setText(f"Predicted class: {classes[int(prediction[0][0])]}")
        else:
            self.text_label.setText("Predicted class: ...")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())