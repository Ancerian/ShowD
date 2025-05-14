import cv2
import time
import numpy as np

import requests

from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path="tflite-model/tflite_learn_4.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
stream.set(3, 640)
stream.set(4, 480)

def send_command(command):
    r = requests.get(f"http://localhost:5000/{command}")
    if r.status_code == 200:
        print(f"Command {command} sent successfully.")
    else:
        print(f"Failed to send command {command}. Status code: {r.status_code}")


while True:
    (grabbed, frame) = stream.read()
    cv2.imshow('Object detector', frame)

    frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_rgb, (96, 96))
    frame_resized = np.expand_dims(frame_resized, axis=-1)
    input_data = np.expand_dims(frame_resized, axis=0)

    # Convert to int8 using quantization parameters
    input_data = (np.round(input_data.astype(np.int32)) - 127)
    input_data = np.clip(input_data, -128, 127).astype(np.int8)

    interpreter.set_tensor(0, input_data)
    interpreter.invoke()

    scores = interpreter.get_tensor(180).astype(np.int32).flatten()
    scores_float = (scores + 128) * 0.003921568859368563

    prob = scores_float / sum(scores_float)
    print(sum(prob))
    prob_names = zip([
        "down",
        "left",
        "nothing",
        "right",
        "stop",
        "up"
    ], prob)

    max_class = max(prob_names, key=lambda x: x[1])
    print(f"Max class: {max_class[0]} with probability {max_class[1]}")

    if max_class[1] > 0.5:
        if max_class[0] == "up":
            send_command("up")
        elif max_class[0] == "down":
            send_command("down")
        elif max_class[0] == "left":
            send_command("left")
        elif max_class[0] == "right":
            send_command("right")
        elif max_class[0] == "stop":
            send_command("stop")

    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(1)

cv2.waitKey(0)
cv2.destroyAllWindows()
