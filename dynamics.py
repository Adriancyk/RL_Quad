from dis import dis
from typing import List
import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
from pyquaternion import Quaternion
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from utils import generate_axes, Arrow3D
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class QuadrotorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(QuadrotorEnv, self).__init__()
        # Using North-East-Down (NED) coordinate system
        self.control_mode = 'takeoff'
        self.get_f, self.get_g = self.get_dynamics()
        self.mass = 2.0
        self.g = 9.81
        self.z_ground = 0.0
        self.dt = 0.02 # 50Hz
        self.max_steps = 2000
        self.uni_circle_radius = 3.0 # m
        self.uni_vel = 0.05 # m/s
        self.reward_exp = True

        self.action_low = np.array([-0.3, -0.3, -25.0]) # fx fy fz
        self.action_high = np.array([0.3, 0.3, 0.0]) # fx fy fz
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(3,)) # fx fy fz

        self.observation_low = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10, 0, 0, 0, 0, -10, -10, -10, -10]) # x y z dx dy dz q0 q1 q2 q3 x y dx dy
        self.observation_high = np.array([10.0, 10.0, 2.0, 10.0, 10.0, 10.0, 1, 1, 1, 1, 10, 10, 10, 10])
        self.observation_space = spaces.Box(low=self.observation_low, high=self.observation_high, shape=(14,)) # x y z dx dy dz

        self.bounded_state_space = spaces.Box(low=self.observation_low[:6], high=self.observation_high[:6], shape=(6,)) # x y z dx dy dz

        # Initialize Env
        self.state = np.zeros((6,)) # x y z dx dy dz
        self.state[:2] += np.random.uniform(-1.0, 1.0, size=(2,))
        self.state[2] += np.random.uniform(0.05, 0.35)
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0
        self.uni_state = np.zeros((4,)) # x y dx dy
        self.uni_state[0] = self.uni_circle_radius # initial x at (1, 0)
        self.steps = 0
        self.desired_yaw = 0.0
        self.desired_hover_height = -2.5

        self.observation = np.concatenate([self.state, self.quaternion, self.uni_state]) # update observation ---> # Quad: x y z dx dy dz + q0 q1 q2 q3 + Uni: x y dx dy

        self.reset()

    def step(self, action, use_reward=True):
        # mix the states + q + uni_states to observation
        state, reward, done, info = self._step(action, use_reward) # t+1
        uni_state = self.get_unicycle_state() # t+1
        self.observation = np.concatenate([state, self.quaternion, uni_state]) # t+1
        
        return self.observation, reward, done, info

    def _step(self, action, use_reward=True):
        # x y z dx dy dz without no attitude
        
        self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action) + self.state # t+1
        self.desired_attitude(action) # t+1
        self.steps += 1 # t+1
        done = False
        reward = 0.0
        info = dict()
        if use_reward:
            reward = self.get_reward(self.state, action, self.uni_state)
        if self.get_out():
            info['out_of_bound'] = True
            reward = 0
            done = True
        else:
            done = self.steps >= self.max_steps
            info['reach_max_steps'] = True

        return self.state, reward, done, info
    
    # def euler_to_quaternion(self, roll, pitch, yaw):
    #     # roll pitch yaw to quaternion ([123] sequence)
    #     # reference paper (section 5.6) link: https://www.semanticscholar.org/paper/Representing-Attitude-%3A-Euler-Angles-%2C-Unit-%2C-and-Diebel/5c0edc899359a69c3769da238491f93e7a2f6d6d
    #     cy = np.cos(yaw * 0.5)
    #     sy = np.sin(yaw * 0.5)
    #     cr = np.cos(roll * 0.5)
    #     sr = np.sin(roll * 0.5)
    #     cp = np.cos(pitch * 0.5)
    #     sp = np.sin(pitch * 0.5)

    #     q0 = cy * cr * cp + sy * sr * sp
    #     q1 = cy * cp * sr - cr * sy * sp
    #     q2 = cy * cr * sp + sy * sr * cp
    #     q3 = cr * cp * sy - cy * sr * sp
    #     return np.array([q0, q1, q2, q3])

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
        pitch = np.arctan2(f_x, f_z) # theta   y, x == y/x
        yaw = self.desired_yaw
        q = Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(axis=[0, 1, 0], angle=pitch) * Quaternion(axis=[1, 0, 0], angle=roll)
        self.quaternion = np.array([q[0], q[1], q[2], q[3]])
        return self.quaternion, roll, pitch, yaw

    def get_reward(self, state, action, uni_state):
        
        reward = 0.0
        if state[2] > self.z_ground:
            reward += -100
        
        
        if self.control_mode == 'takeoff':
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
            reward += -2*((state[0] - 0)**2 + (state[1] - 0)**2)
        elif self.control_mode == 'tracking':
            reward += -5*((state[0] - uni_state[0])**2 + (state[1] - uni_state[1])**2) # x and y position difference
            reward += -0.05*((state[3] - uni_state[2])**2 + (state[4] - uni_state[3])**2) # x and y velocity difference
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
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
            f_x[5] = self.g * self.mass
            return f_x

        def get_g(state):
            g_x = np.zeros((state.shape[0], 3))  # 6x3
            g_x[3:, :] = np.eye(3)
            # g_x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
            #             [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            #             [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            #             [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
            return g_x

        return get_f, get_g
    
    def get_out(self):

        mask = np.array([1, 1, 1, 0, 0, 0])
        out_of_bound = np.logical_or(self.state < self.bounded_state_space.low, self.state > self.bounded_state_space.high)
        out_of_bound = np.any(out_of_bound * mask)
        if out_of_bound:
            return True
        return False
    
    def get_unicycle_state(self):
        ang_vel = self.uni_vel / self.uni_circle_radius
        theta = ang_vel * self.steps * self.dt / np.pi * 180
        self.uni_state[0] = self.uni_circle_radius * np.cos(theta) # x
        self.uni_state[1] = self.uni_circle_radius * np.sin(theta) # y
        self.uni_state[2] = -self.uni_vel * np.sin(theta) # dx
        self.uni_state[3] = self.uni_vel * np.cos(theta) # dy

        return self.uni_state
    

    def reset(self):
        self.steps = 0
        self.state = np.zeros((6,)) # x y z dx dy dz
        self.state[:2] += np.random.uniform(-1.0, 1.0, size=(2,))
        self.state[2] -= np.random.uniform(0.05, 0.35)
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0
        self.uni_state = np.zeros((4,)) # x y dx dy
        self.uni_state[0] = self.uni_circle_radius # initial x at (1, 0)
        self.desired_yaw = 0.0
        self.observation = np.concatenate([self.state, self.quaternion, self.uni_state])
        return self.observation
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)


def uni_animation():
    
    env = QuadrotorEnv()
    state_list= []
    env.dt = 0.01
    for env.steps in range(150):
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

def render1(states, angles):


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

def render2(quad_state, quad_angles, uni_states):

    x = quad_state[:, 0]
    y = quad_state[:, 1]
    z = quad_state[:, 2]

    roll = quad_angles[:, 0]
    pitch = quad_angles[:, 1]
    yaw = quad_angles[:, 2]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(quad_state)):
        ax.plot(x[:i], y[:i], z[:i], 'o', markersize=2, color='black', alpha=0.5)
        ax.plot(uni_states[:i, 0], uni_states[:i, 1], 0, 'o', markersize=3, color='blue', alpha=0.5)

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


if __name__ == "__main__":


    uni_animation()