import cv2
import time
import numpy as np
import collections

import requests

from mp.run import recognize_action, classes

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
stream.set(3, 640)
stream.set(4, 480)

def send_command(command):
    r = requests.get(f"http://127.0.0.1:5000/{command}")
    if r.status_code == 200:
        print(f"Command {command} sent successfully.")
    else:
        print(f"Failed to send command {command}. Status code: {r.status_code}")

predictions_buffer = []
frames_to_accumulate = 10

while True:
    (grabbed, frame) = stream.read()

    frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = recognize_action(frame_rgb)
    if r:
        prediction, frame_rgb = r

        predicted = classes[int(prediction[0][0])]
        predictions_buffer.append(predicted)

        if len(predictions_buffer) == frames_to_accumulate:
            most_common = collections.Counter(predictions_buffer).most_common(1)[0][0]
            send_command(most_common)
            predictions_buffer = []

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.05)

cv2.waitKey(0)
cv2.destroyAllWindows()
