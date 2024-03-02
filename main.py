import torch
import numpy as np
import argparse

from dynamics import QuadrotorEnv
from RL_agent import SAC
from utils import prYellow
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
import os, sys

def train(agent, env, args):

    if hasattr(args, 'load_agent'):
        agent.load_weights(args.resume)

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

            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            total_numsteps += 1
            episode_reward += reward
            episode_steps += 1

            done = 0 if episode_steps + 1 == env.max_steps else float(done) # 0 if episode_steps + 1 == env.max_steps else 1
            memory.push(obs, action, reward, next_obs, done)

            obs = next_obs
            if done:
                if 'out_of_bound' in info and info['out_of_bound']:
                    prYellow('Episode {} - step {} - eps_rew {} - Info: out of bound'.format(i_episode, episode_steps, episode_reward))
                    
                elif 'reach_max_steps' in info and info['reach_max_steps']:
                    prYellow('Episode {} - step {} - eps_rew {} - Info: reach max steps'.format(i_episode, episode_steps, episode_reward))

        if i_episode > 0 and i_episode % 50 == 0:
            agent.save_model(args.env_name)
            
    







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--num_episodes', type=int, nargs='?', default=400, help='total number of episode')
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
    parser.add_argument('--env_name', type=str, nargs='?', default='Quadrotor', help='env name')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--control_mode', default='hover', type=str, help='')

    args = parser.parse_args()

    # if args.resume == 'default':
    #     args.resume = os.getcwd() + '/output/{}-run0'.format(args.env_name)
    # elif args.resume.isnumeric():
    #     args.resume = os.getcwd() + '/output/{}-run{}'.format(args.env_name, args.resume)
    #     args.load_agent = True

    
    env = QuadrotorEnv()
    agent = SAC(env.observation_space.shape[0], env.action_space, env, args)

    if args.seed > 0:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    train(agent, env, args)