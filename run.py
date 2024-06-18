from dynamics import QuadrotorEnv
from utils import render, render_video
from agent import SAC
from pyquaternion import Quaternion
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from gym import spaces
from compensator import compensator
from cbf import robust_safe_filter

## test the trained model
def test(args):

    env_norm = QuadrotorEnv(args, mass=None, wind=None) # nominal environment
    env = QuadrotorEnv(args, mass=None, wind=None) # perturbed environment
    env.max_steps= 2000
    cwd = os.getcwd()
    env.desired_hover_height = -1.0

    action_space = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))
    agent = SAC(19, action_space, args)
    # path_dl = os.path.join(cwd, 'checkpoints/dynamic_chasing_NED_10m_50hz_circle_s19_ready')
    path_dl = os.path.join(cwd, 'checkpoints/sac_checkpoint_Quadrotor_episode3400_mode_dynamic_chasing')
    agent.load_model(path_dl)

    obs = env.reset()
    rel_pos_prev = np.zeros((2, 4))
    # obs[:3] = [0, 0, 0] # uncomment if need to take off from the [0, 0, 0]
    obs_list = []
    action_list = []
    angles = []
    uni_states = []
    done = False

    while not done:
        s = obs[:6].copy()
        s[2] = -s[2]
        s[5] = -s[5]
        obs_list.append(s)
        uni_state = env.get_unicycle_state(env.steps, args.traj)
        uni_states.append(uni_state)
        action = agent.select_action(obs, eval=True)


        next_state, q, _, rel_pos_prev = env.move(obs[:6], action) # move the quadrotor
        # target moving streategy
        # if env.steps <= 500:
            # rel_pos_prev = np.zeros((2, 4)) - obs[:2].reshape(-1, 1)
        # rel_pos_prev = np.zeros((2, 4)) - obs[:2].reshape(-1, 1)
        if env.steps > 500:
            env.desired_hover_height = -0.2
        # rel_pos_prev = np.zeros((2, 4)) - obs[:2].reshape(-1, 1)
                
        rel_height = env.desired_hover_height - next_state[2]
        obs = np.concatenate([next_state, q, rel_pos_prev.flatten('F'), [rel_height]])
        a = action.copy()
        a[2] = -a[2]
        action_list.append(a)
        q = np.array(obs[6:10])
        quaternion = Quaternion(q[0], q[1], q[2], q[3])
        yaw, pitch, roll  = quaternion.yaw_pitch_roll
        angles.append([roll, pitch, yaw])

        if env.steps > env.max_steps:
            done = True

    action_list = np.array(action_list)
    obs_list = np.array(obs_list)


    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # plt.subplots_adjust(wspace=0.1)
    
    # fig = plt.figure()
    # plt.plot(obs_list[:, 3], label='vx')
    # plt.plot(obs_list[:, 4], label='vy')
    # plt.plot(obs_list[:, 5], label='vz')
    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # plt.plot(action_list[:, 0], label='fx')
    # plt.plot(action_list[:, 1], label='fy')
    # plt.plot(action_list[:, 2], label='fz')
    # plt.legend()
    # plt.show()

    angles = np.array(angles)
    uni_states = np.array(uni_states)
    render(obs_list, angles, uni_states, action_list, enable_cone=False)

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
    parser.add_argument('--lam_a', type=float, nargs='?', default=10.0, metavar='G', help='action temporal penalty coefficient (set to 0 to disable smoothness penalty)')
    parser.add_argument('--policy', default="Gaussian", type=str,  nargs='?', help='Policy Type: Gaussian | Deterministic')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random action_list (default: 10000)')
    
    parser.add_argument('--env_name', type=str, nargs='?', default='Quadrotor', help='env name')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--control_mode', default='tracking', type=str, help='')
    parser.add_argument('--load_model', default=False, type=bool, help='load trained model for train function')

    parser.add_argument('--load_model_path', default='checkpoints/tracking_NED_15m_50hz_01', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--traj', default='figure8', type=str, help='set desired trajectory shape')
    
    # parser.add_argument('--load_model_path', default='checkpoints/sac_checkpoint_Quadrotor_episode2000_mode_tracking', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--save_model_path', default='checkpoints', type=str, help='path to save model')
    parser.add_argument('--mode', default='test', type=str, help='train or evaluate')

    # compensator parameters
    parser.add_argument('--Ts', type=float, nargs='?', default=0.02, help='sampling time')
    parser.add_argument('--obs_dim', type=int, nargs='?', default=6, help='observation dimension')
    parser.add_argument('--act_dim', type=int, nargs='?', default=3, help='action dimension')
    parser.add_argument('--wc', type=float, nargs='?', default=50, help='cut-off frequency')
    parser.add_argument('--a_param', type=float, nargs='?', default=-10, help='a parameter for adaptive control')
    

    args = parser.parse_args()
    test(args)