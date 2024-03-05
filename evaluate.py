import torch
import numpy as np
import argparse

from dynamics import QuadrotorEnv
from RL_agent import SAC
from utils import prYellow
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from replay_memory import ReplayMemory
import os, sys

# import numpy as np
# from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import time
from scipy.spatial.transform import Rotation as R

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

############# axes compute from rpy
def generate_axes(r, p, y):

    r = R.from_euler('xyz', [r,p,y], degrees=False)

    v = r.as_matrix()

    return v

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
    parser.add_argument('--policy', default="Gaussian", type=str,  nargs='?', help='Policy Type: Gaussian | Deterministic')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
    
    parser.add_argument('--env_name', type=str, nargs='?', default='Quadrotor', help='env name')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--control_mode', default='hover', type=str, help='')

    args = parser.parse_args()
    cwd = os.getcwd()
    model_path = os.path.join(cwd, 'checkpoints/sac_takeoff_Quadrotor_1m')

    env = QuadrotorEnv()
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(model_path, evaluate=True)

    state = env.reset()
    done = False
    total_reward = 0
    states = []
    angles = []
    while not done:
        states.append(state)
        action = agent.select_action(state, eval=True)
        # action[0] = 0.0
        # action[1] = 0.0
        # action[2] = 9.81*2+1.0
        # env.desired_yaw = 0.1
        q = np.array(state[6:10])
        quaternion = Quaternion(q[0], q[1], q[2], q[3])
        yaw, pitch, roll  = quaternion.yaw_pitch_roll
        angles.append([roll, pitch, yaw])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    states = np.array(states)
    angles = np.array(angles)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(states[:, 0], states[:, 1], states[:, 2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()


    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    roll = angles[:, 0]
    pitch = angles[:, 1]
    yaw = angles[:, 2]

    for i in range(len(states)):
        ax.plot(x[:i], y[:i], z[:i], 'o', markersize=2, color='black', alpha=0.5)

        ### xaxis = red, yaxis = blue, zaxis = green
        
        v = generate_axes(roll[i], pitch[i], yaw[i])
        
        
        a = Arrow3D([x[i], x[i]+v[0, 0]], [y[i], y[i]+v[1, 0]], [z[i], z[i]+v[2, 0]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
        
        a = Arrow3D([x[i], x[i]+v[0, 1]], [y[i], y[i]+v[1, 1]], [z[i], z[i]+v[2, 1]], mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
        ax.add_artist(a)
        
        a = Arrow3D([x[i], x[i]+v[0, 2]], [y[i], y[i]+v[1, 2]], [z[i], z[i]+v[2, 2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="g")
        ax.add_artist(a)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim3d(-10,10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(0,10)
        
        plt.title('Quadrotor trajectory and orientation in 3D')
        plt.draw()
        plt.show(block=False)

        plt.pause(0.01)
        plt.cla()
