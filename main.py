import torch
import numpy as np
import argparse

from dynamics import QuadrotorEnv, render1, render2
from agent import SAC
from utils import prYellow
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
from pyquaternion import Quaternion
import os, sys

def train(agent, env, args):
    env.control_mode = args.control_mode
    if args.load_model is True:
        cwd = os.getcwd()
        model_path = os.path.join(cwd, args.load_model_path)
        agent.load_model(model_path)

    total_numsteps = 0
    updates = 0

    memory = ReplayMemory(args.replay_size)

    for i_episode in range(1, args.num_episodes + 1):
        episode_reward = 0
        episode_steps = 0
        done = False
        env.reset()
        obs = env.observation

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(obs)  # Sample action from policy

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            total_numsteps += 1
            episode_reward += reward
            episode_steps += 1

            mask = 1 if episode_steps == env.max_steps else float(not done)

            memory.push(obs, action, reward, next_obs, mask)

            obs = next_obs
            if done:
                if 'out_of_bound' in info and info['out_of_bound']:
                    prYellow('Episode {} - step {} - eps_rew {} - Info: out of bound'.format(i_episode, episode_steps, episode_reward))
                    
                elif 'reach_max_steps' in info and info['reach_max_steps']:
                    prYellow('Episode {} - step {} - eps_rew {} - Info: reach max steps'.format(i_episode, episode_steps, episode_reward))

        if i_episode > 0 and i_episode % 50 == 0:
            agent.save_model(args.env_name, suffix = 'episode{}_mode_{}'.format(i_episode, args.control_mode))
            
def test(agent, env, args):
    env.control_mode = args.control_mode
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(args.load_model_path, evaluate=True)

    state = env.reset()
    total_reward = 0
    done = False
    states = []
    angles = []
    uni_states = []
    actions = []
    while not done:
        s = state[:3].copy()
        s[2] = -s[2]
        states.append(s)
        uni_states.append(state[10:])
        action = agent.select_action(state, eval=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        a = action.copy()
        actions.append(a)
        q = np.array(state[6:10])
        quaternion = Quaternion(q[0], q[1], q[2], q[3])
        yaw, pitch, roll  = quaternion.yaw_pitch_roll
        angles.append([roll, pitch, yaw])

        total_reward += reward
        state = next_state
    actions = np.array(actions)
    fig = plt.figure()
    plt.plot(actions[:, 0], label='u1')
    plt.plot(actions[:, 1], label='u2')
    plt.plot(actions[:, 2], label='u3')
    plt.show()
        
    

    states = np.array(states)

    fig = plt.figure()
    plt.plot(states[:, 0], label='x')
    plt.plot(states[:, 1], label='y')
    plt.plot(states[:, 2], label='z')
    plt.show()
    angles = np.array(angles)
    uni_states = np.array(uni_states)
    render2(states, angles, uni_states)








if __name__ == "__main__":
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
    parser.add_argument('--control_mode', default='tracking', type=str, help='')
    parser.add_argument('--load_model', default=False, type=bool, help='load trained model')
    parser.add_argument('--load_model_path', default='checkpoints/takeoff_NED_25m_50hz_01', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--save_model_path', default='checkpoints', type=str, help='path to save model')
    parser.add_argument('--mode', default='test', type=str, help='train or evaluate')


    args = parser.parse_args()

    
    env = QuadrotorEnv()
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    if args.seed > 0:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.mode == 'train':
        train(agent, env, args)
    else:
        test(agent, env, args)