from dis import dis
from typing import List
import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class QuadrotorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(QuadrotorEnv, self).__init__()

        self.dynamics_mode = 'Quadrotor'
        self.get_f, self.get_g = self.get_dynamics()
        self.mass = 0.027
        self.g = 9.81
        self.z_ground = 0.0
        self.dt = 0.01
        self.max_episode_steps = 400
        self.uni_circle_radius = 1.0
        self.uni_vel = 0.1
        self.reward_exp = True

        self.action_low = np.array([0.0, 0.0, 0.0]) # fx fy fz
        self.action_high = np.array([0.5, 0.5, 1.0]) # fx fy fz
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(3,)) # fx fy fz

        # Initialize Env
        self.state = np.zeros((6,)) # x y z dx dy dz
        self.observaton = np.zeros((6+4+4,)) # Quad: x y z dx dy dz + q0 q1 q2 q3 + Uni: x y dx dy
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0
        self.uni_state = np.zeros((4,)) # x y dx dy
        self.uni_state[0] = self.uni_circle_radius # initial x at (1, 0)
        self.episode_step = 0
        self.desired_yaw = 0.0
        self.desired_hover_height = 0.5


        self.reset()

    def step(self, action, use_reward=True):
        state, reward, done, info = self._step(action, use_reward)
        return state, reward, done, info

    def _step(self, action, use_reward=True):
        # x y z dx dy dz without no attitude
        self.desired_attitude(action)
        self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state, self.quaternion) @ action) + self.state
        self.observation = np.concatenate([self.state, self.quaternion, self.uni_state])
        self.episode_step += 1
        reward = 0.0
        
        info = dict()
        if use_reward:
            reward = self.get_reward(self.state, action, self.uni_state)
        if self.get_done():
            info['out_of_bound'] = True
            reward += -100
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps
            info['reach_max_steps'] = True

        return self.state, reward, self.observation, done, info
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        # roll pitch yaw to quaternion ([123] sequence)
        # reference paper link: https://www.semanticscholar.org/paper/Representing-Attitude-%3A-Euler-Angles-%2C-Unit-%2C-and-Diebel/5c0edc899359a69c3769da238491f93e7a2f6d6d
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        q0 = cy * cr * cp + sy * sr * sp
        q1 = cy * cp * sr - cr * sy * sp
        q2 = cy * cr * sp + sy * sr * cp
        q3 = cr * cp * sy - cy * sr * sp
        return np.array([q0, q1, q2, q3])

    def desired_attitude(self, action):
        # compute desired state from action and tranfer to quaternion
        # R ([123] sequence) from body to world frame pitch: theta; roll: phi; yaw: psi
        # [[cos(pitch)cos(yaw), sin(roll)sin(pitch)cos(yaw)-cos(roll)sin(yaw), cos(roll)sin(pitch)cos(yaw)+sin(roll)sin(yaw)];
        # [cos(pitch)sin(yaw), sin(roll)sin(pitch)sin(yaw)+cos(roll)cos(yaw), cos(roll)sin(pitch)sin(yaw)-sin(roll)cos(yaw)];
        # [-sin(pitch), cos(pitch)sin(roll), cos(roll)cos(pitch)]]

        f_total = np.linalg.norm(action) # in body frame
        f_x = action[0] # in world frame
        f_y = action[1] # in world frame
        f_z = action[2] # in world frame

        roll = -np.arcsin(f_y/f_total) # phi
        pitch = np.arctan2(f_x/f_z) # theta
        yaw = self.desired_yaw
        self.quaternion = self.euler_to_quaternion(roll, pitch, yaw)
        return 

    def get_reward(self, state, action, uni_state):
        
        reward = 0.0
        if state[2] < self.z_ground:
            reward += -100
            return reward
        
        reward += -2*((state[0] - uni_state[0])**2 + (state[1] - uni_state[1])**2) # x and y position difference
        reward += -0.1*(state[3]**2 + state[4]**2) # x and y velocity
        reward += self.desired_hover_height - state[2] # z position difference

        if self.reward_exp:
            reward = np.exp(reward)
        return reward

    def get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            f_x[0] = state[3]
            f_x[1] = state[4]
            f_x[2] = state[5]
            f_x[5] = -self.g
            return f_x

        def get_g(state, quaternion):
            q0 = quaternion[0]
            q1 = quaternion[1]
            q2 = quaternion[2]
            q3 = quaternion[3]
            g_x = np.zeros((state.shape[0], 3))  # 6x3
            g_x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
            return g_x

        return get_f, get_g
    
    def get_unicycle_state(self):
        ang_vel = self.uni_vel / self.uni_circle_radius
        theta = ang_vel * self.episode_step * self.dt / np.pi * 180
        self.uni_state[0] = self.uni_circle_radius * np.cos(theta) # x
        self.uni_state[1] = self.uni_circle_radius * np.sin(theta) # y
        self.uni_state[2] = -self.uni_vel * np.sin(theta) # dx
        self.uni_state[3] = self.uni_vel * np.cos(theta) # dy

    def get_done(self):

        mask = np.array([1, 1, 1, 0, 0, 0])
        out_of_bound = np.logical_or(self.state < self.bounded_state_space.low, self.state > self.bounded_state_space.high)
        out_of_bound = np.any(out_of_bound * mask)
        if out_of_bound:
            return True
        return False
    

    def reset(self):
        self.episode_step = 0
        self.state = np.zeros((6,)) # x y z dx dy dz
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0
        self.uni_state = np.zeros((4,)) # x y dx dy
        self.uni_state[0] = self.uni_circle_radius # initial x at (1, 0)
        self.episode_step = 0
        self.desired_yaw = 0.0
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)


def uni_animation():
    
    env = QuadrotorEnv()
    state_list= []
    env.dt = 0.01
    for env.episode_step in range(150):
        state = env.uni_state.copy()
        state_list.append(state)
        env.get_unicycle_state()

    state_list = np.array(state_list)

    fig, ax = plt.subplots()
    line, = ax.plot(state_list[0, 0], state_list[0, 1])
    quiver_x = ax.quiver(state_list[0, 0], state_list[0, 1], state_list[0, 2], 0, color='pink', scale=env.uni_vel/0.3)
    quiver_y = ax.quiver(state_list[0, 0], state_list[0, 1], 0, state_list[0, 3], color='b', scale=env.uni_vel/0.3)
    quiver_total = ax.quiver(state_list[0, 0], state_list[0, 1], state_list[0, 2], state_list[0, 3], color='g', scale=env.uni_vel/0.3)

    def update(frame):
        line.set_data(state_list[:frame, 0], state_list[:frame, 1])
        quiver_x.set_UVC(state_list[frame, 2], 0)
        quiver_y.set_UVC(0, state_list[frame, 3])
        quiver_total.set_UVC(state_list[frame, 2], state_list[frame, 3])
        quiver_x.set_offsets(state_list[frame, :2])
        quiver_y.set_offsets(state_list[frame, :2])
        quiver_total.set_offsets(state_list[frame, :2])
        return line, quiver_x, quiver_y, quiver_total,

    ani = FuncAnimation(fig, update, frames=range(len(state_list)), blit=True, repeat=False)

    ax.set_xlim([min(state_list[:, 0]-0.5), max(state_list[:, 0]+0.5)])
    ax.set_ylim([min(state_list[:, 1]-0.5), max(state_list[:, 1]+0.5)])
    ax.set_aspect('equal')
    plt.show()



if __name__ == "__main__":


    uni_animation()