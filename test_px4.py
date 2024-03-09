from dynamics import QuadrotorEnv, render1
from agent import SAC
from pyquaternion import Quaternion
import numpy as np
import argparse

def test(args):
    env = QuadrotorEnv()
    env.control_mode = 'takeoff'
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(args.load_model_path, evaluate=True)

    state = env.reset()
    done = False
    states = []
    angles = []
    while not done:
        states.append(state[:3])
        state[:2] = [0, 0]
        # state[10:] = [0, 0, 0, 0]
        # state = [0, 0, state[2], 0, 0, state[5], state[6], state[7], state[8], state[9], 0,0, 0, 0]
        state[6:10] = [1, 0, 0, 0]
        action = agent.select_action(state, eval=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        q = np.array(state[6:10])
        quaternion = Quaternion(q[0], q[1], q[2], q[3])
        yaw, pitch, roll  = quaternion.yaw_pitch_roll
        angles.append([roll, pitch, yaw])
        state = next_state
    states = np.array(states)
    angles = np.array(angles)
    render1(states, angles)

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
    parser.add_argument('--load_model_path', default='checkpoints/sac_Quadrotor_takeoff_1m_06_50hz', type=str, help='path to trained model (caution: do not use it for model saving)')
    parser.add_argument('--save_model_path', default='checkpoints', type=str, help='path to save model')
    parser.add_argument('--mode', default='test', type=str, help='train or evaluate')
    

    args = parser.parse_args()
    test(args)