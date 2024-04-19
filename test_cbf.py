from dynamics import QuadrotorEnv, render, render_video
from agent import SAC
from pyquaternion import Quaternion
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from gym import spaces
from compensator import compensator
from cbf import safe_filter, robust_safe_filter

def test(args):
    env_nom = QuadrotorEnv(args)
    env = QuadrotorEnv(args, mass=2.0, wind=0.5)
    env.control_mode = 'dynamic_landing'
    env.max_steps= 1500
    cwd = os.getcwd()

    action_space_dl = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))

    agent_dl = SAC(18, action_space_dl, args)

    path_dl = os.path.join(cwd, 'checkpoints/dynamic_landing_NED_10m_50hz_ready')

    agent_dl.load_model(path_dl)

    comp_on = False
    cbf_on = False
    esti_on = cbf_on
    in_safe_set = False
    done = False
    obs = env.reset()
    obs = obs[:18]
    state = obs[:6]
    rel_pos_prev = np.zeros((2, 4))
    # rel_pos_fur = np.zeros((2, 4))
    comp = compensator(state, args, env_nom.dt) # compensator
    cbf = robust_safe_filter(args, env_nom.dt, env_nom.mass) # robust control barrier function
    
    # create lists to store data
    obss = []
    actions = []
    angles = []
    uni_states = []
    track_error = []
    rs = []
    last_uni_vel = ...
    sigma_hat = np.zeros_like(state)
    while not done:
        s = obs[:3].copy()
        s[2] = -s[2]
        obss.append(s)
        uni_state = env.get_unicycle_state(env.steps)
        uni_states.append(uni_state)

        if last_uni_vel is ...:
            last_uni_vel = uni_state[2:4]

        track_error.append(np.sqrt((state[0] - uni_state[0])**2 + (state[1] - uni_state[1])**2))
        rs.append(cbf.get_r(state[:6]))

        state_dl = obs
        state_dl[10:] = rel_pos_prev.flatten('F')
        
        action = agent_dl.select_action(state_dl, eval=True)
        state = obs[:6]

        if cbf.get_r(state[:6]) - np.linalg.norm([state[0] - uni_state[0], state[1] - uni_state[1]]) > 0.015 and in_safe_set is False and cbf_on is True:
            print('cbf activated at: ', env.steps)
            in_safe_set = True

        # robust filter
        if comp_on is True and in_safe_set is False:
            f = env_nom.get_f(state)
            g = env_nom.get_g(state)
            action, sigma_hat = comp.get_safe_control(state, action, f, g)

        # robust & safe filter
        if in_safe_set is True and cbf_on is True:
            action = cbf.get_safe_control(state, np.concatenate([uni_state, (uni_state[2:4] - last_uni_vel)/env.dt]), sigma_hat, action)

        if esti_on is True:
            f = env_nom.get_f(state)
            g = env_nom.get_g(state)
            sigma_hat = comp.get_estimation(state, action, f, g)

        last_uni_vel = uni_state[2:4]

        next_state, q, rel_pos_fur, rel_pos_prev = env.move(obs[:6], action)
        obs = np.concatenate([next_state, q, rel_pos_prev.flatten('F')])
        a = action.copy()
        a[2] = -a[2]
        actions.append(a)
        q = np.array(obs[6:10])
        quaternion = Quaternion(q[0], q[1], q[2], q[3])
        yaw, pitch, roll  = quaternion.yaw_pitch_roll
        angles.append([roll, pitch, yaw])
        # obs = next_state

        if env.steps > env.max_steps:# or obs[2] >= -0.3:
            done = True

    track_error = np.array(track_error)
    rs = np.array(rs)
    fig = plt.figure()
    plt.plot(track_error, label='distance Xq Xu', color='darkviolet')
    plt.plot(rs, label='r', color='darkorange')
    plt.legend()
    plt.show()


    actions = np.array(actions)
    fig = plt.figure()
    plt.plot(actions[:, 0], label='ux', color='darkviolet')
    plt.plot(actions[:, 1], label='uy', color='darkorange')
    plt.plot(actions[:, 2], label='uz', color='dodgerblue')
    plt.legend()
    plt.show()
        
    

    obss = np.array(obss)

    fig = plt.figure()
    plt.plot(obss[:, 0], label='x', color='darkviolet')
    plt.plot(obss[:, 1], label='y', color='darkorange')
    plt.plot(obss[:, 2], label='z', color='dodgerblue')
    plt.legend()
    plt.show()
    angles = np.array(angles)
    uni_states = np.array(uni_states)
    render_video(obss, angles, uni_states, actions)

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
                    help='Steps sampling random actions (default: 10000)')
    
    parser.add_argument('--env_name', type=str, nargs='?', default='Quadrotor', help='env name')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--control_mode', default='dynamic_landing', type=str, help='')
    parser.add_argument('--load_model', default=False, type=bool, help='load trained model for train function')

    parser.add_argument('--load_model_path', default='checkpoints/tracking_NED_15m_50hz_01', type=str, help='path to trained model (caution: do not use it for model saving)')
    
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