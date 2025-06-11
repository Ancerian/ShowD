from djitellopy import Tello

tello = Tello()
tello.connect(False)

tello.takeoff()
tello.land()