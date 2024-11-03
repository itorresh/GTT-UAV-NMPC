from pymavlink import mavutil
import numpy as np
import math
import time
from Algorithms import Bat_Algorithm
from Functions import NMPC
import csv


def normalize_angle_positive(res):
    while res > 2 * np.pi:
        res -= 2 * np.pi
    while res < 0 * np.pi:
        res += 2 * np.pi
    return res


def normalize_angle_pi(res):
    while res > np.pi:
        res -= 2 * np.pi
    while res < -np.pi:
        res += 2 * np.pi
    return res


def to_quaternion(yawd, rolld, pitchd):
    cy = math.cos(yawd * 0.5)
    sy = math.sin(yawd * 0.5)
    cr = math.cos(rolld * 0.5)
    sr = math.sin(rolld * 0.5)
    cp = math.cos(pitchd * 0.5)
    sp = math.sin(pitchd * 0.5)
    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return [w, x, y, z]


def get_angles_from_uav(vehicle):
    msg = (vehicle.recv_match(type='ATTITUDE', blocking=True))
    rollx = msg.roll
    pitchx = msg.pitch
    yawx = -1 * msg.yaw + math.pi / 2
    # Convert yaw to the range [-pi, pi]
    yawx = (yawx + math.pi) % (2 * math.pi) - math.pi
    return [rollx, pitchx, yawx]


def get_quaternion_from_uav(vehicle):
    msg = (vehicle.recv_match(type='ATTITUDE_QUATERNION', blocking=True))
    quaternion = [msg.q1, msg.q2, msg.q3, msg.q4]
    return quaternion


def get_position_from_uav(vehicle):
    msg = vehicle.recv_match(type='LOCAL_POSITION_NED', blocking=True)
    y = msg.x
    x = msg.y
    z = -1 * msg.z
    vy = msg.vx
    vx = msg.vy
    vz = msg.vz

    return [x, y, z, vx, vy, vz]


def get_flight_mode(master):
    msg = master.recv_match(type='HEARTBEAT', blocking=True)
    flight_mode = mavutil.mode_string_v10(msg)
    return flight_mode


def truck_positions_from_csv(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row if it exists
        for row in csv_reader:
            x, y, z = map(float, row)  # Assuming the CSV has three colu$
            yield x, y, z


# Do the first GPU  compile
print("Compiling Numba...")
# (self, radio, theta, z, chi, V, rref,  thetaref, zref, Vref, uref, phi, Ts, dx, dy, du, dr)
fn = NMPC(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

print("Numba Ready, waiting for heartbeat...")

# Mavlink Connection
#the_connection = mavutil.mavlink_connection('udpin:localhost:14550')
the_connection = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600) # Jetson Nano
the_connection.wait_heartbeat()
print("Heartbeat ok")

# System parameters
posGT = [150, -150, 30]
R_ref = 150
theta_ref = np.pi / 2
V_UAV = 10
hRef = 50
u_ref = -V_UAV / R_ref
umin = -np.pi / 3
umax = np.pi / 3
Ts = 1
# Ts = 1
VUAVSet = V_UAV


csv_file = 'tramo.csv'

positions_generator = truck_positions_from_csv(csv_file)

# Options
loggingData = 1
v_flag = 1
if loggingData == 1:
    file_path = "Test_results.csv"
    new_data = ["R0", "Theta0", "UAVx", "UAVy", "UAVz", "h", "yaw", "phi", "u", "u1", "u2", "V_UAV",
                "posGT[0]", "posGT[1]", "posGT[2]", "optimization_time"]

    with open(file_path, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(new_data)

# Flight Mode Flag
print("Waiting for Guided mode...")
flag = 0
movingTarget = 'True' #movingTarget = 'False' for stationary


# NMPC Weighting Factors tuning
Qr = 0.0037
Qtheta = 54
Qz = 0.006
Qchi = 1
Qv = 0.1
R = 1
Rchi = 1
N = 10

start_time = time.time()
while True:
    if flag == 0:
        if get_flight_mode(the_connection) != 'GUIDED':
            continue
        else:
            flag = 1
    if movingTarget == 'True':
        try:
            posGT = next(positions_generator)

        except StopIteration:
            break

    # Get position and velocity
    [UAVx, UAVy, UAVz, vUAVx, vUAVy, vUAVz] = get_position_from_uav(the_connection)
    VUAVread = np.sqrt(vUAVx ** 2 + vUAVy ** 2 + vUAVz ** 2)

    # Get angles
    [roll, pitch, yaw] = get_angles_from_uav(the_connection)
    yaw = normalize_angle_pi(yaw)
    phi = normalize_angle_pi(np.arctan2((UAVy - posGT[1]), (UAVx - posGT[0])))

    # Get states
    R0 = np.sqrt((posGT[0] - UAVx) ** 2 + (posGT[1] - UAVy) ** 2)
    Theta0 = normalize_angle_pi(np.pi - phi + yaw)

    h = np.sqrt((UAVz - posGT[2]) ** 2)

    x_kk = np.array([[R0 - R_ref], [Theta0 - theta_ref], [h - hRef], [pitch - 0], [VUAVread - VUAVSet]])

    K = np.array([
        [-0.0209, -1.1537, -0.0000, -0.0000, -0.0065],
        [-0.0000, -0.0000, -0.0326, -1.2467, 0.0000],
        [0.0001, 0.0001, -0.0000, -0.0000, -0.2702]
    ])

    pop_init = u = np.matmul(K, x_kk)

    controller = 'NL'
    #u_ref = -VUAVread/R_ref
    start_time_optimization = time.time()
    fn = NMPC(R0, Theta0, h, pitch, VUAVread, R_ref, theta_ref, hRef, V_UAV, u_ref, Ts, phi, yaw, Qr, Qtheta, Qz, Qchi, Qv, R,
              Rchi)
    optimiser = Bat_Algorithm(fn, Population_Size=800, Num_Movements=30, dimension=N, pop_init=pop_init)
    optimiser.Run(fn)
    elapsed_time_optimization = start_time_optimization - time.time()

    u = optimiser.Best_Position[0, 0]
    u1 = optimiser.Best_Position[0, 1]
    u2 = optimiser.Best_Position[0, 2]
    costo = optimiser.Best_Fitness


    the_connection.mav.send(mavutil.mavlink.MAVLink_set_attitude_target_message(
        0,
        the_connection.target_system,
        the_connection.target_component,
        0b01100111,
        to_quaternion(-(u + u_ref), 0, u1),  # to_quaternion(-(u+u_ref), 0, np.deg2rad(hCommand)),
        0, 0, 0, 0) #0, 0, 0, u2)
    )
    Ckf = np.array([[np.cos(phi), np.sin(phi), 0, 1], [np.sin(phi) / R0, -np.cos(phi) / R0, 1, 0]])


    if loggingData == 1:
        new_data = [R0, Theta0, UAVx, UAVy, UAVz, h, yaw, phi, u, u1, u2, VUAVread, posGT[0], posGT[1],
                    posGT[2], elapsed_time_optimization]
        with open(file_path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(new_data)

    elapsed_time = time.time() - start_time


    if elapsed_time > 20:
        if v_flag == 1:
            the_connection.mav.command_long_send(
                the_connection.target_system,
                the_connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                0,  # Confirmation
                1,  # Speed type
                10,  # Speed in m/s
                -1,  # Throttle
                0, 0, 0, 0  # Unused parameters
            )
            v_flag = 0



    print(
        f"{controller} Radio: {R0}, Theta: {Theta0}, posUAV: {UAVx}, {UAVy},{UAVz}, h: {h}, yaw: {yaw}, phi: {phi}, u: {u}, u1 {u1},u2 {u2}, applied: {-(u + u_ref)}, elapsed time: {elapsed_time}")
