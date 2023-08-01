import asyncio
from mavsdk import System
import rtsp
import numpy as np
import time
import argparse
import threading

from mavsdk.offboard import OffboardError
from mavsdk.offboard import VelocityBodyYawspeed
import time 
eps = 0.000001
import signal
    

async def get_drone_hight(drone,area_covered_by_detected_body):
    
    print("drone", drone, area_covered_by_detected_body)
    if drone is not None:
        print("Fetching amsl altitude at home location....")
        async for terrain_info in drone.telemetry.home():
            absolute_altitude = terrain_info.absolute_altitude_m
            return absolute_altitude
    else:
        print(area_covered_by_detected_body, 1/area_covered_by_detected_body)
        return 1/area_covered_by_detected_body 

async def land_drone(drone):#,status_text_task):
    
    await drone.action.land()
    #status_text_task.cancel()

async def init_drone(port="ttyUSB0",baud =57600, fly_meters = 0, speed = 1, fly_mode = 'local', port_abs_path = False):
 
    drone = System()

    print("trying to connect...")
    if port_abs_path:
        print("connecting via full path to {} ..".format(port))
        await drone.connect(system_address="{}".format(port))
    else:
        print("connecting via port and baud to serial:///dev/{}:{}".format(port, baud))
        await drone.connect(system_address="serial:///dev/{}:{}".format(port,baud))

    #status_text_task = asyncio.ensure_future(print_status_text())

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global/local position estimate...")
    if fly_mode == 'global':
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position state is good enough for flying.")
                break
        print("Fetching amsl altitude at home location....")
        async for terrain_info in drone.telemetry.home():
            absolute_altitude = terrain_info.absolute_altitude_m
            break   
    elif fly_mode == 'local':
        print("checking local")
        async for health in drone.telemetry.health():
            if health.is_local_position_ok:
                print("-- Local position state is good enough for flying.")
                break
    else:
        print("Exisiting. No such a fly_mode {}".format(fly_mode))
        exit(9)
    

    print("-- Arming")
    if fly_meters != 0: #drone is already armed:
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(fly_meters)
        print("-- Taking off")
        await drone.action.takeoff()
        await asyncio.sleep(10)
    

    print("Press any key to start...")
    input()
    ##############################3
    
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
    
    return drone

async def get_drone_flight_mode(drone):
    if drone is None:
        return "OFFBOARD"
    async for flight_mode in drone.telemetry.flight_mode():
            if  str(flight_mode) != 'OFFBOARD':
                print(f"Exisiting: Flight mode is: {flight_mode}")
            break
    return flight_mode

async def move_drone_by_velocity(drone,forward_speed, right_speed, down_speed, yaw_speed, K, yaw_K):
    if drone is not None:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(forward_speed*K, right_speed*K, down_speed*K, yaw_speed*yaw_K))
    else:
        pass

async def move_drone_by_meters(drone,y_meters, x_meters, z_meters, speed):


    largest = max(abs(y_meters), abs(x_meters), abs(z_meters))
  
    if drone is not None:
        #print("in")
        if largest > 0:
            y_speed = (y_meters/largest)*speed
            x_speed = (x_meters/largest)*speed
            z_speed = (z_meters/largest)*speed
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(y_speed, x_speed, z_speed, 0.0))
            
            await asyncio.sleep(largest/speed - eps)
            
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    else:
        await asyncio.sleep(largest/speed - eps)

    

async def print_status_text():
    global drone
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        print("Error")
async def stop_velocity_commands(drone):
    await drone.offboard.stop()

def move_wrapper(drone,move):
    loop.run_until_complete(move_drone(drone,move[0], move[1], move[2], args.speed))    
    loop.run_until_complete(land_drone(drone))
  

