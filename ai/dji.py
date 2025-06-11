import cv2
import time
import numpy as np
import collections

from djitellopy import Tello
from mp.run import recognize_action, classes

tello = Tello()
tello.connect(False)

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
stream.set(3, 640)
stream.set(4, 480)

height = 100

def send_command(command):
    global height
    if command == "up":
        if height < 100:
            tello.move_up(20)
            height += 50
        else:
            print("Maximum height reached.")
    elif command == "down":
        if height > 10:
            tello.move_down(20)
            height -= 50
        else:
            print("Already at minimum height.")
    elif command == "forward":
        tello.move_forward(100)
    elif command == "backward":
        tello.move_backward(100)
    elif command == "left":
        tello.move_left(50)
    elif command == "right":
        tello.move_right(50)

predictions_buffer = []
frames_to_accumulate = 10

tello.takeoff()

try:

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

        time.sleep(0.1)
        
except KeyboardInterrupt:
    tello.land()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
