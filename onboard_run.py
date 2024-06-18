from dynamics import QuadrotorEnv
from agent import SAC
from pyquaternion import Quaternion
import numpy as np
import argparse
import numpy as np
#from sensor_msgs.msg import Image
import time
import sys
import math
# from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
from pymavlink import mavutil
from array import array
import time
from pymavlink.quaternion import QuaternionBase
from scipy.spatial.transform import Rotation
from gym import spaces
import motioncapture
import csv
from compensator import compensator



# define quat multiply, wxyz

def quart_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    # in rad
    return roll, pitch, yaw



def rpy_to_quat(r,p,y):
    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler('xyz', [r,p,y], degrees=False)
    rot_quat = rot.as_quat() # x,y,z,w
    quat = [rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]]  # w,x,y,z
    return quat

def request_message_interval(connection, message_id, frequency_hz):
    """
    Request MAVLink message in a desired frequency,
    documentation for SET_MESSAGE_INTERVAL:
        https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL

    Args:
        message_id (int): MAVLink message ID
        frequency_hz (float): Desired frequency in Hz
    """
    connection.mav.command_long_send(
        connection.target_system, connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        message_id, # The MAVLink message ID
        1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
        0, 0, 0, 0, # Unused parameters
        0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
    )





def set_target_attitude_throttle_easy(connection, boot_time, action):
    """ Sets the target attitude while in depth-hold mode.

    'roll', 'pitch', and 'yaw' are angles in degrees.

    """

    quat, thrust_k = yaw3force_to_quat_thrust(action[0],action[1],action[2], 0)

    connection.mav.set_attitude_target_send(
        int(1e3 * (time.time() - boot_time)), # ms since boot
        connection.target_system, connection.target_component,
        # allow throttle to be controlled by depth_hold mode
        0b00000111, #use throttle and rpy(quat form) only
        # -> attitude quaternion (w, x, y, z | zero-rotation is 1, 0, 0, 0)
        quat,0,0,0, # quat, roll rate, pitch rate, yaw rate
        thrust_k, 0 #  thrust/z_rate/z_ratio, thrust_3d_ned
    )


    # 0.5 throttle means hover, this implies that thrust 0.5 == gravity == mass * 9.8 == 2.1 * 9.8 N
    # saturation 0.05 - 0.95
    # ratio->thrust      thrust == 2.1*9.8/(0.5-0.05) * (ratio - 0.05)
    # thrust->ratio      ratio == (thrust/(2.1*9.8/(0.5-0.05))) + 0.05
    # more details in website: https://docs.google.com/spreadsheets/d/1_75aZqiT_K1CdduhUe4-DjRgx3Alun4p8V2pt6vM5P8/edit#gid=0
    # more details in website: https://docs.google.com/spreadsheets/d/1dymF2TC28N2Mu_rkw31S5rMkdSGRO-Go9Q7r3SkbNuA/edit#gid=0




def yaw3force_to_quat_thrust(fx, fy, fz, yaw):
    
    f_total = np.sqrt(fx**2 + fy**2 + fz**2)

    roll = -np.arcsin(fy/-f_total) * 2.5  # phi
    
    pitch = np.arctan(fx/fz) * 2.5 # theta   y, x == y/x

    yaw = yaw
    
    while roll > np.pi  or roll < -np.pi:
        if roll > np.pi :
            roll -= np.pi *2
        elif roll < -np.pi:
            roll += np.pi *2

    while pitch > np.pi  or pitch < -np.pi:
        if pitch > np.pi :
            pitch -= np.pi *2
        elif pitch < -np.pi:
            pitch += np.pi *2

    # print('control_rpy',roll, pitch, yaw)

    quat = rpy_to_quat(roll, pitch, yaw)

    # param set guid_options 0
    thrust_k = (f_total/(2*9.81/(0.50-0.0))) + 0.0  # relative throttle

    # param set guid_options 8
    # thrust_k = (f_total/(2*9.81/(0.215-0.0))) + 0.0  # absolute throttle
    
    # print('thrust k:', thrust_k)

    return quat, thrust_k



# Configure AHRS2 message to be sent at xxx Hz


def test(args):

    sys.path.append('/home/tx2/RL_ws/src/RL_gazebo_drone/scripts/checkpoints/')
    env = QuadrotorEnv(args)
    env_nom = QuadrotorEnv(args, mass=1.667, wind=None)
    action_space_tf = spaces.Box(low=np.array([-0.3, -0.3, -25.0]), high=np.array([0.3, 0.3, 0.0]), shape=(3,))
    action_space_tr = spaces.Box(low=np.array([-0.5, -0.5, -25.0]), high=np.array([0.5, 0.5, 0.0]), shape=(3,))

    agent_tf = SAC(19, action_space_tf, args)
    agent_tf.load_model('checkpoints/takeoff_NED_10m_50hz_ready_19_02')

    agent_tr = SAC(19, action_space_tr, args)
    agent_tr.load_model('checkpoints/dynamic_chasing_NED_10m_50hz_circle_s19_ready')
    print("loaded")
    # env.desired_hover_height = -0.5


    state = np.zeros(19,) #env.reset()
    uni_prev_buffer = np.zeros((2, 4))
    action = np.zeros(3,)
    done = False


    comp = compensator(state[:6], args, args.Ts) # compensator

    # Start a connection listening on a UDP port
    connection = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

    # Wait for the first heartbeat to set the system and component ID of remote system for the link
    connection.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))

    #message request definitions

    request_message_interval(connection, 31, 200)
    #attitude_quat msg id = 31
    request_message_interval(connection, 32, 200)
    #local position NED msg id = 32

    ned_num = 0
    quat_num = 0
    boot_time = time.time()



    time.sleep(1)

    target_mode = 4
    connection.set_mode(target_mode)
    connection.arducopter_arm()
    # wait until arming confirmed (can manually check with master.motors_armed())
    print("Waiting for the vehicle to arm")
    connection.motors_armed_wait()
    print('Armed!')
    r_31_record = [0]
    p_31_record = [0]
    y_31_record = [0]
    roll_speed_record = [0]
    pitch_speed_record = [0]
    yaw_speed_record = [0]
    x_record = [0]
    y_record = [0]
    z_record = [0]
    mc = motioncapture.connect('vicon', '192.17.163.14')

    logname = 'panpanpan.csv'
    fields = ['time', 'x', 'y', 'z', 'fx', 'fy', 'fz', 'sig_x', 'sig_y', 'sig_z', 'sig_vx', 'sig_vy', 'sig_vz']
    csvlog = open('csvlog/' + logname, 'a')
    while not done:
        
        msg = connection.recv_match()
        if not msg:
            continue
        
        if msg.get_type() == 'LOCAL_POSITION_NED':
            # print("mavutil.mavlink.LOCAL_POSITION_NED",32)
            # print("\n\n*****Got message: %s*****" % msg.get_type())
            # print("Message: %s" % msg)
            ned_num += 1
            # append ned position into enu records
            x_record[0] = msg.x
            y_record[0] = msg.y
            z_record[0] = msg.z + 0.3 * 0 # quadrotor height is 0.3m
            # append ned velocity into enu records
        if x_record[0] != 0:
            break
        

    vx_record = [0]
    vy_record = [0]
    vz_record = [0]

    
    time_start = time.time()
    time_last = time_start

    time_record = []
    flag = 0

    while not done:
        #target_mode = 4
        #connection.set_mode(target_mode)
        
        msg = connection.recv_match()
        if not msg:
            continue
        if msg.get_type() == 'ATTITUDE_QUATERNION':
            # print("mavutil.mavlink.MAVLINK_MSG_QUAT",31)
            quat_num += 1
            # append ned velocity into enu record
            quat_ned_collect = np.array([msg.q1, msg.q2, msg.q3, msg.q4])
            

            r_31, p_31, y_31 = quart_to_rpy(msg.q2, msg.q3, msg.q4, msg.q1)
            # print("\n\n*****Got message: %s*****" % msg.get_type())
            # print("real roll, pitch, yaw", r_31,p_31,y_31)
            # print("Message: %s" % msg)
            quat_num += 1
            r_31_record.append(r_31)
            p_31_record.append(p_31)
            y_31_record.append(y_31)

            # quat_enu_collect = quat_ned2enu(quat_ned_collect)

            # # quat_enu_collect = quat_ned_collect
            # quat_enu_collect = np.array([1,0,0,0])
            
            # print('quat_ned_collect')
            # print(quat_ned_collect)
            # print('quat_enu_collect')
            # print(quat_enu_collect)

        if msg.get_type() == 'LOCAL_POSITION_NED':
            # print("mavutil.mavlink.LOCAL_POSITION_NED",32)
            # print("\n\n*****Got message: %s*****" % msg.get_type())
            # print("Message: %s" % msg)
            ned_num += 1
            # append ned position into enu records
            x_record.append(msg.x - x_record[0])
            y_record.append(msg.y - y_record[0])
            z_record.append(msg.z - z_record[0])
            # append ned velocity into enu records
            vx_record.append(msg.vx)
            vy_record.append(msg.vy)
            vz_record.append(msg.vz)

            time_record.append(time.time() - time_start)
        
        #frequency control
        time_now = time.time()

        if time_now - time_last < 0.0075:
            continue
        time_last = time_now

	    # target_mode = 4
    	# connection.set_mode(target_mode)

        
        writer = csv.DictWriter(csvlog, fieldnames=fields)
        writer.writeheader()


        state[0:10] = [x_record[-1],y_record[-1],z_record[-1],vx_record[-1],vy_record[-1],vz_record[-1],
                       quat_ned_collect[0],quat_ned_collect[1], quat_ned_collect[2], quat_ned_collect[3]]
        state[-1] = env.desired_hover_height - state[2] # relative height
        f = env_nom.get_f(state[:6])
        g = env_nom.get_g(state[:6])
        sigma_hat = comp.get_estimation(state[:6], action, f, g)
        print('sigma: ', sigma_hat)
        
        
        # mc.waitForNextFrame()
        # for name, obj in mc.rigidBodies.items():
        #     if name == 'uni':
        #         uni_state = obj.position[:2].copy()
        #         uni_state[1] = -uni_state[1] # change y direction

        uni_state = env.get_unicycle_state(env.steps).copy() # to be edited
        uni_prev_buffer = np.roll(uni_prev_buffer, shift=1, axis=1)
        uni_prev_buffer[:, 0] = uni_state[:2]
        rel_pos_prev = (uni_prev_buffer - state[:2].reshape(-1, 1)).flatten('F')

        # print('cur z:', z_record[-1])

        if time.time() - time_start > 40: 
            # env.desired_hover_height = -0.5
            state[10:18] = rel_pos_prev
            action = agent_tr.select_action(state, eval=True)
        else: # takeoff
            state[10:18] = np.zeros(8, )
            action = agent_tf.select_action(state, eval=True)
        if time.time() - time_start > 20:
            env.desired_hover_height = -0.5
        if time.time() - time_start > 30:
            env.desired_hover_height = -0.3

        # print(action)
        # print('xyz', x_record[-1], y_record[-1], z_record[-1])
        flag += 1
        env.steps += 1 

        # set_target_attitude_throttle(connection, boot_time, [action[1], action[0], action[2]])
        set_target_attitude_throttle_easy(connection, boot_time, action)
        time_now = time.time()

        rowdata ={'time': time_now, 'x': x_record[-1], 'y': y_record[-1], 'z': z_record[-1], 'fx': action[0], 'fy': action[1], 'fz': action[2], 'sig_x': sigma_hat[0], 'sig_y': sigma_hat[1], 'sig_z': sigma_hat[2], 'sig_vx': sigma_hat[3], 'sig_vy': sigma_hat[4], 'sig_vz': sigma_hat[5]}
        writer.writerow(rowdata)

        time_last = time_now

        if time.time() - time_start > 20:
            csvlog.close()
            done = True

    time.sleep(2)


    connection.mav.command_long_send(
            connection.target_system, connection.target_component,
            21, 0, #mode land, empty
            0,  0, 0, 0, 0, 0,  0, # arbot-alt(0), landmode(0), empty, yaw, lat, lon, alt
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--num_episodes', type=int, nargs='?', default=800, help='total number of episode')
    parser.add_argument('--updates_per_step', type=int, nargs='?', default=1, help='total number of updates per step')
    parser.add_argument('--batch_size', type=int, nargs='?', default=256, help='batch size (default: 256)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--hidden_size', type=int, nargs='?', default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('seed', type=int, nargs='?', default=12345, help='random seed')
    parser.add_argument('--gamma', type=float, nargs='?', default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, nargs='?', default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, nargs='?', default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--target_update_interval', type=int, nargs='?', default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, nargs='?', default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--lr', type=float, nargs='?', default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--policy', default="Gaussian", type=str,  nargs='?', help='Policy Type: Gaussian | Deterministic')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
    
    parser.add_argument('--env_name', type=str, nargs='?', default='Quadrotor', help='env name')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--control_mode', default='takeoff', type=str, help='')
    parser.add_argument('--load_model', default=False, type=bool, help='load trained model for train function')
    parser.add_argument('--load_model_path', default='/home/tx2/RL_ws/src/RL_gazebo_drone/scripts/checkpoints/good_med_takeoff_15', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--save_model_path', default='checkpoints', type=str, help='path to save model')
    parser.add_argument('--mode', default='test', type=str, help='train or evaluate')

    # compensator parameters
    parser.add_argument('--Ts', type=float, nargs='?', default=0.01, help='sampling time')
    parser.add_argument('--obs_dim', type=int, nargs='?', default=6, help='observation dimension')
    parser.add_argument('--act_dim', type=int, nargs='?', default=3, help='action dimension')
    parser.add_argument('--wc', type=float, nargs='?', default=100, help='cut-off frequency')
    parser.add_argument('--a_param', type=float, nargs='?', default=-10, help='a parameter for adaptive control')
    

    args = parser.parse_args()
    test(args)
