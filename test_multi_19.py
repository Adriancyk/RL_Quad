from dynamics import QuadrotorEnv, render
from agent import SAC
from pyquaternion import Quaternion
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from compensator import compensator
from gym import spaces

def test(args):
    
    env_norm = QuadrotorEnv(args)
    env = QuadrotorEnv(args, mass=2.0, wind=None)
    env.max_steps= 6000
    cwd = os.getcwd()
    env.desired_hover_height = -1.0

    action_space = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))
    # action_space_tr = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))
    # action_space_dl = spaces.Box(low=np.array([-1.0, -1.0, -25.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,))

    agent_tf = SAC(19, action_space, args)
    agent_tr = SAC(19, action_space, args)
    agent_dl = SAC(19, action_space, args)

    path_tf = os.path.join(cwd, 'checkpoints/takeoff_NED_10m_50hz_ready_19_02')
    path_tr = os.path.join(cwd, 'checkpoints/tracking_NED_10m_50hz_ready_19_02')
    path_dl = os.path.join(cwd, 'checkpoints/sac_checkpoint_Quadrotor_episode3000_mode_dynamic_chasing')

    agent_tf.load_model(path_tf)
    agent_tr.load_model(path_tr)
    agent_dl.load_model(path_dl)


    obs = env.reset()
    rel_pos_prev = np.zeros((2, 4))
    obs[:3] = [0, 0, 0]
    done = False
    obs_list = []
    action_list = []
    angles = []
    uni_states = []

    # comp = compensator(obs[:6], args, env.dt)
    # comp_on = False
    # env.control_mode = 'takeoff'

    while not done:
        s = obs[:3].copy()
        s[2] = -s[2]
        obs_list.append(s)
        uni_states.append(env.get_unicycle_state(env.steps))

        state_tf = obs.copy()
        state_tf[10:18] = np.zeros(8)
        action = agent_tf.select_action(state_tf, eval=True)
        if env.steps > 200 and env.steps <= 1000:
            action = agent_tr.select_action(obs, eval=True)
        elif env.steps > 1000:
            # if env.steps % 500 == 0:
            #     env.desired_hover_height = np.random.uniform(0.5, 1.5)
            #     print(env.steps)
            #     print('new desired height:', env.desired_hover_height)

            if env.steps % 500 == 0:
                env.desired_hover_height = -0.2
            if env.steps % 1000 == 0:
                env.desired_hover_height = -1.0
            # env.desired_hover_height = -0.5
            # if env.steps > 1500:
            #     env.desired_hover_height = -1.0
            action = agent_dl.select_action(obs, eval=True)
        
        # if comp_on is True:
        #     f = env_norm.get_f(obs[:6])
        #     g = env_norm.get_g(obs[:6])
        #     action = comp.get_safe_control(obs[:6], action, f, g)

        next_state, q, _, rel_pos_prev = env.move(obs[:6], action)
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
    fig = plt.figure()
    plt.plot(action_list[:, 0], label='ux', color='darkviolet')
    plt.plot(action_list[:, 1], label='uy', color='darkorange')
    plt.plot(action_list[:, 2], label='uz', color='dodgerblue')
    plt.legend()
    plt.show()
        
    

    obs_list = np.array(obs_list)

    fig = plt.figure()
    plt.plot(obs_list[:, 0], label='x', color='darkviolet')
    plt.plot(obs_list[:, 1], label='y', color='darkorange')
    plt.plot(obs_list[:, 2], label='z', color='dodgerblue')
    plt.legend()
    plt.show()
    angles = np.array(angles)
    uni_states = np.array(uni_states)
    render(obs_list, angles, uni_states, action_list)

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
    
    # parser.add_argument('--load_model_path', default='checkpoints/sac_checkpoint_Quadrotor_episode2000_mode_tracking', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--save_model_path', default='checkpoints', type=str, help='path to save model')
    parser.add_argument('--mode', default='test', type=str, help='train or evaluate')
    

    args = parser.parse_args()
    test(args)