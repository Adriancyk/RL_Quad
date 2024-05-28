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

def test(args):

    env_norm = QuadrotorEnv(args, mass=None, wind=None) # nominal environment
    env = QuadrotorEnv(args, mass=None, wind=None) # perturbed environment
    env.max_steps= 2000
    cwd = os.getcwd()
    env.desired_hover_height = -1.0

    action_space = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))
    agent = SAC(19, action_space, args)
    path_dl = os.path.join(cwd, 'checkpoints/dynamic_chasing_NED_10m_50hz_circle_s19_ready')
    # path_dl = os.path.join(cwd, 'checkpoints/sac_checkpoint_Quadrotor_episode1550_mode_tracking')
    agent.load_model(path_dl)

    obs = env.reset()
    rel_pos_prev = np.zeros((2, 4))
    obs[:3] = [0, 0, 0] # uncomment if need to take off from the [0, 0, 0]
    obs_list = []
    action_list = []
    angles = []
    uni_states = []
    rel_dist = []
    rs = []
    last_uni_vel = ...
    start_landing_step = ...
    sigma_hat = np.zeros_like(obs[:6])

    comp = compensator(obs[:6], args, env_norm.dt) # compensator
    rcbf = robust_safe_filter(args, env_norm.dt, env_norm.mass) # robust control barrier function

    comp_on = False
    cbf_on = False
    in_safe_set = False
    done = False

    while not done:
        s = obs[:3].copy()
        s[2] = -s[2]
        obs_list.append(s)
        uni_state = env.get_unicycle_state(env.steps)
        uni_states.append(uni_state)

        if last_uni_vel is ...:
            last_uni_vel = uni_state[2:4]

        rel_dist.append(np.sqrt((obs[0] - uni_state[0])**2 + (obs[1] - uni_state[1])**2)) # distance between quadrotor and unicycle
        safe_radius = rcbf.get_safe_radius(obs[:6])
        rs.append(safe_radius)

        action = agent.select_action(obs, eval=True)

        # decide to activate the safe filter or not
        if safe_radius - np.linalg.norm([obs[0] - uni_state[0], obs[1] - uni_state[1]]) > 0.01 and in_safe_set is False:
            if cbf_on is True:
                print('RCBF activated at: ', env.steps)
            start_landing_step = env.steps
            # env.desired_hover_height = -0.3
            in_safe_set = True

        # disturbance estimator
        f = env_norm.get_f(obs[:6])
        g = env_norm.get_g(obs[:6])
        _, action_comp, sigma_hat = comp.get_safe_control(obs[:6], action, f, g)

        # robust filter
        if comp_on is True and in_safe_set is False:
            action = action_comp + action

        # robust & safe filter
        if in_safe_set is True and cbf_on is True:
            action = rcbf.get_safe_control(obs[:6], np.concatenate([uni_state, (uni_state[2:4] - last_uni_vel)/env.dt]), sigma_hat, action)

        last_uni_vel = uni_state[2:4]

        action[0:2] = np.clip(action[0:2], -1, 1)  # Clip the first two actions to the range [-1, 1]
        action[2] = np.clip(action[2], -25, 0)  # Clip the third action to the range [-25, 0]

        next_state, q, _, rel_pos_prev = env.move(obs[:6], action) # move the quadrotor
        env.last_action = action
        # target moving streategy
        if env.steps <= 500:
            rel_pos_prev = np.zeros((2, 4)) - obs[:2].reshape(-1, 1)

        if env.steps > 500 and env.steps % 30 == 0 and in_safe_set is True: # gradually lower the height
            env.desired_hover_height += 0.1 if env.desired_hover_height < -0.3 else 0.0
        

                
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

    rel_dist = np.array(rel_dist)
    rs = np.array(rs)
    action_list = np.array(action_list)
    obs_list = np.array(obs_list)
    time_steps = np.arange(len(rel_dist))


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.1)
    
    # ============================================
    axs[0].plot(time_steps, rel_dist, label='Relative Distance', color='darkviolet')
    axs[0].plot(time_steps, rs, label='Safe Set Radius', color='darkorange')
    axs[0].text(0.20, 0.03, 'Takeoff and Tracking Mode', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[0].text(0.65, 0.03, 'Landing Mode', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    min_y = min(min(rel_dist), min(rs))
    max_y = max(max(rel_dist), max(rs))
    axs[0].axvline(x=start_landing_step, ymin=min_y, ymax=max_y, color='teal', alpha=0.5, linestyle='--')
    axs[0].fill_between(time_steps, min_y, max_y, where=time_steps<start_landing_step, color='gold', alpha=0.2)
    axs[0].fill_between(time_steps, min_y, max_y, where=time_steps>=start_landing_step, color='mediumseagreen', alpha=0.2)
    axs[0].title.set_text('Relative Distance')
    axs[0].legend(loc = 'upper left', ncol = 2, frameon=False)
    # ============================================
    axs[1].plot(action_list[:, 0], label='ux', color='darkviolet')
    axs[1].plot(action_list[:, 1], label='uy', color='darkorange')
    axs[1].plot(action_list[:, 2], label='uz', color='dodgerblue')
    axs[1].title.set_text('Action')
    axs[1].legend(loc = 'upper center', ncol = 3, frameon=False)
    # ============================================
    axs[2].plot(obs_list[:, 0], label='x', color='darkviolet')
    axs[2].plot(obs_list[:, 1], label='y', color='darkorange')
    axs[2].plot(obs_list[:, 2], label='z', color='dodgerblue')
    axs[2].text(0.20, 0.03, 'Takeoff and Tracking Mode', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    axs[2].text(0.65, 0.03, 'Landing Mode', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    min_y = obs_list.min()
    max_y = obs_list.max()
    axs[2].axvline(x=start_landing_step, ymin=min_y, ymax=max_y, color='teal', alpha=0.5, linestyle='--')
    axs[2].fill_between(time_steps, min_y, max_y, where=time_steps<start_landing_step, color='gold', alpha=0.2)
    axs[2].fill_between(time_steps, min_y, max_y, where=time_steps>=start_landing_step, color='mediumseagreen', alpha=0.2)
    axs[2].legend(loc = 'upper right', ncol = 3, frameon=False)

    plt.show()

    # angles = np.array(angles)
    # uni_states = np.array(uni_states)
    # render_video(obs_list, angles, uni_states, action_list)

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
    parser.add_argument('--traj', default=None, type=bool, help='set desired trajectory shape')
    
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