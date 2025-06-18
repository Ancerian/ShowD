from scan import search_tello

from djitellopy import TelloSwarm

swarm = TelloSwarm.fromIps(search_tello())

swarm.connect(False)
swarm.takeoff()

swarm.land()
swarm.end()
