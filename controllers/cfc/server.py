from flask import Flask
import queue

CONTROL_QUEUE = queue.Queue()
app = Flask(__name__)


@app.route("/up")
def drone_up():
    CONTROL_QUEUE.put("UP")
    return "UP OK"

@app.route("/down")
def drone_down():
    CONTROL_QUEUE.put("DOWN")
    return "DOWN OK"

@app.route("/forward")
def drone_forward():
    CONTROL_QUEUE.put("FORWARD")
    return "FORWARD OK"

@app.route("/backward")
def drone_backward():
    CONTROL_QUEUE.put("BACKWARD")
    return "BACKWARD OK"

@app.route("/left")
def drone_left():
    CONTROL_QUEUE.put("LEFT")
    return "LEFT OK"

@app.route("/right")
def drone_right():
    CONTROL_QUEUE.put("RIGHT")
    return "RIGHT OK"


def start_server():
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


if __name__ == "__main__":
    start_server()
