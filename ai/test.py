from djitellopy import Tello

if __name__ == "__main__":
    tello = Tello()
    tello.connect(False)
    tello.takeoff()
    tello.move_up(50)
    tello.move_forward(100)
    tello.rotate_clockwise(90)
    tello.move_forward(100)
    tello.land()
