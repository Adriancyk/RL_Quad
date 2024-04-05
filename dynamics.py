from dis import dis
from typing import List
import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
from pyquaternion import Quaternion
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import generate_axes, Arrow3D
import os
import matplotlib.animation as animation

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class QuadrotorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, args):

        super(QuadrotorEnv, self).__init__()
        # Using North-East-Down (NED) coordinate system
        if args is None:
            self.control_mode = 'tracking'
        else:
            self.control_mode = args.control_mode

        self.args = args
        self.get_f, self.get_g = self.get_dynamics()
        self.iter = 0 # for buffer saving
        self.mass = 2.0
        self.g = 9.81
        self.z_ground = 0.0
        self.dt = 0.02 # 50Hz
        self.max_steps = 2000
        self.reward_exp = True
        self.steps = 0
        self.init_uni_angle = np.random.uniform(0, 2*np.pi, size=(1,)).item()
        self.desired_yaw = 0.0 # quadrotor yaw angle
        self.uni_circle_radius = 1.5 # m
        self.desired_hover_height = ... # quadrotor hover height
        self.init_quad_height = ...


        # quadrotor
        self.action_low = np.array([-1.0, -1.0, -25.0]) # fx fy fz
        self.action_high = np.array([1.0, 1.0, 0.0]) # fx fy fz
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(3,)) # fx fy fz

        # self.observation_low = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10, 0, 0, 0, 0, -10, -10, -10, -10]) # x y z dx dy dz q0 q1 q2 q3 x y dx dy
        # self.observation_high = np.array([10.0, 10.0, 2.0, 10.0, 10.0, 10.0, 1, 1, 1, 1, 10, 10, 10, 10])
        # self.observation_space = spaces.Box(low=self.observation_low, high=self.observation_high, shape=(14,)) # x y z dx dy dz

        self.bounded_state_space = spaces.Box(low=np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10]), high=np.array([10.0, 10.0, 2.0, 10.0, 10.0, 10.0]), shape=(6,)) # x y z dx dy dz
        
        if self.control_mode == 'takeoff': # takeoff vertically
            self.init_quad_height = 0.0
            self.desired_hover_height = -1.0
        elif self.control_mode == 'tracking': # tracking a unicycle on the ground
            self.init_quad_height = -1.0
            self.desired_hover_height = -1.0
        elif self.control_mode == 'landing': # landing vertically to the ground
            self.init_quad_height = -1.0
            self.desired_hover_height = 0.0
        elif self.control_mode == 'dynamic_landing': # landing dynamically on a moving unicycle
            self.init_quad_height = -1.0
            self.desired_hover_height = -0.3
        
        # set initial state
        self.state = np.zeros((6,)) # x y z dx dy dz

        if self.control_mode == 'takeoff':
            self.state[:2] = [0.0, 0.0] # add noise to x y
            self.state[2] = 0.0 # init z
        elif self.control_mode == 'tracking':
            self.state[:2] = [0.0, 0.0]
            self.state[2] = self.init_quad_height
        elif self.control_mode == 'landing':
            self.state[:2] = [0.0, 0.0]
            self.state[2] = self.init_quad_height
        elif self.control_mode == 'dynamic_landing':
            theta = np.random.uniform(0, 2*np.pi)
            self.state[:2] = [self.uni_circle_radius * np.cos(theta), self.uni_circle_radius * np.sin(theta)]
            self.state[2] = self.init_quad_height

        self.state[:2] += np.random.uniform(-0.2, 0.2, size=(2,)) # add noise to x y
        self.state[2] += np.random.uniform(-0.15, 0.15) # add noise to z
        self.state[3:6] += np.random.uniform(-0.2, 0.2, size=(3,)) # add noise to velocity
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0

        # unicycle
        self.uni_vel = 0.5 # m/s
        self.buffer_steps = 4
        self.uni_vel += np.random.uniform(-0.1, 0.1)
        self.uni_circle_radius += np.random.uniform(-0.1, 0.1)
        self.uni_state = np.zeros((4,)) # x y dx dy the center of the circle is (0, 0) for now
        self.uni_future_pos = np.zeros((2, self.buffer_steps)) # x, y * steps
        self.uni_prev_buffer = np.zeros((2, self.buffer_steps))
        self.relative_pos_fur = np.zeros((2, self.buffer_steps)) # x, y * fur steps
        self.relative_pos_prev = np.zeros((2, self.buffer_steps)) # x, y * prev steps

        # observation
        self.observation = np.concatenate([self.state, self.quaternion, self.relative_pos_prev.flatten('F')]) # update observation ---> # Quad: x y z dx dy dz + q0 q1 q2 q3 + (relative_pos_fur x y * buffer_steps) 
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.observation),))

        self.reset()

    def step(self, action, use_reward=True):
        # mix the states + q + uni_states to observation
        state, reward, done, info = self._step(action, use_reward) # t+1
        uni_state = self.get_unicycle_state(self.steps) # t+1
        # self.uni_future_pos, _ = self.compute_uni_future_traj(self.buffer_steps) # t+1
        
        # Shift the elements in the buffer to the right
        self.uni_prev_buffer = np.roll(self.uni_prev_buffer, shift=1, axis=1)
        # Insert the new state at the first position
        self.uni_prev_buffer[:, 0] = uni_state[:2]

        # self.relative_pos_prev = self.uni_prev_buffer - state[:2].reshape(-1, 1) + np.random.uniform(-0.02, 0.02, size=(2,4)) * 0 if self.args.mode == 'test' else 1 # t+1
        self.relative_pos_prev = self.uni_prev_buffer - state[:2].reshape(-1, 1) + np.random.uniform(-0.02, 0.02, size=(2, self.buffer_steps)) * (0 if self.args.mode == 'test' else 1) # t+1
        self.observation = np.concatenate([state, self.quaternion, self.relative_pos_prev.flatten('F')]) # t+1
        
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
            reward = self.get_reward(self.state, action, self.uni_state, self.relative_pos_prev)
        if self.get_out():
            info['out_of_bound'] = True
            reward = 0
            done = True
        else:
            done = self.steps >= self.max_steps
            info['reach_max_steps'] = True

        return self.state, reward, done, info
    
    def move(self, state, action):
        state = self.dt * (self.get_f(state) + self.get_g(state) @ action) + state
        uni_state = self.get_unicycle_state(self.steps)
        self.uni_prev_buffer = np.roll(self.uni_prev_buffer, shift=1, axis=1)
        self.uni_prev_buffer[:, 0] = uni_state[:2]
        self.desired_attitude(action) # t+1
        self.uni_future_pos, _ = self.compute_uni_future_traj(self.buffer_steps) # t+1
        rel_pos_fur = self.uni_future_pos - state[:2].reshape(-1, 1)
        rel_pos_prev = self.uni_prev_buffer - state[:2].reshape(-1, 1)
        
        # state = np.concatenate([state, self.quaternion, rel_pos.flatten('F')])
        self.steps += 1
        return state, self.quaternion, rel_pos_fur, rel_pos_prev
    
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

        roll = -np.arcsin(f_y/(-f_total)) # phi
        pitch = np.arctan(f_x/f_z) # theta   y, x == y/x
        yaw = self.desired_yaw
        q = Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(axis=[0, 1, 0], angle=pitch) * Quaternion(axis=[1, 0, 0], angle=roll)
        self.quaternion = np.array([q[0], q[1], q[2], q[3]])
        return self.quaternion, roll, pitch, yaw

    def get_reward(self, state, action, uni_state, relative_pos):
        
        reward = 0.0
        if state[2] > self.z_ground:
            reward += -100
        
        
        if self.control_mode == 'takeoff':
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
            reward += -2*((state[0] - 0)**2 + (state[1] - 0)**2)

        elif self.control_mode == 'tracking':
            distance = np.linalg.norm(relative_pos[:, 0], axis=-1) # absolute distance
            reward += -1.2*np.sum(distance**2)
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
        elif self.control_mode == 'landing':
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
        elif self.control_mode == 'dynamic_landing':
            reward += -1.0*(self.desired_hover_height - state[2])**2
            distance = np.linalg.norm(relative_pos[:, 0], axis=-1)
            reward += -1.2*np.sum(distance**2)
        
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
        # check if the quadrotor is out of boundary
        mask = np.array([1, 1, 1, 0, 0, 0])
        out_of_bound = np.logical_or(self.state < self.bounded_state_space.low, self.state > self.bounded_state_space.high)
        out_of_bound = np.any(out_of_bound * mask)
        if out_of_bound:
            return True
        return False
    
    def get_unicycle_state(self, steps=0):
        uni_state = np.zeros((4,))
        ang_vel = self.uni_vel / self.uni_circle_radius
        theta = ang_vel * steps * self.dt + self.init_uni_angle
        uni_state[0] = self.uni_circle_radius * np.cos(theta) # x
        uni_state[1] = self.uni_circle_radius * np.sin(theta) # y
        uni_state[2] = -self.uni_vel * np.sin(theta) # dx
        uni_state[3] = self.uni_vel * np.cos(theta) # dy

        return uni_state
    
    def compute_uni_future_traj(self, future_steps, dt=None):
        uni_future_pos = []
        uni_furture_vel = []
        for i in range(future_steps):
            cur = self.get_unicycle_state(steps=self.steps + i) 
            uni_future_pos.append(cur[:2])
            uni_furture_vel.append(cur[2:])
        return np.array(uni_future_pos).T, np.array(uni_furture_vel).T

    def reset(self):
        # reset the environment
        self.steps = 0
        # quadrotor
        self.state = np.zeros((6,)) # x y z dx dy dz
        if self.control_mode == 'takeoff':
            self.state[:2] = [0.0, 0.0] # add noise to x y
            self.state[2] = 0.0 # init z
        elif self.control_mode == 'tracking':
            self.state[:2] = [0.0, 0.0]
            self.state[2] = self.init_quad_height
        elif self.control_mode == 'landing':
            self.state[:2] = [0.0, 0.0]
            self.state[2] = self.init_quad_height
        elif self.control_mode == 'dynamic_landing':
            theta = np.random.uniform(0, 2*np.pi)
            self.state[:2] = [self.uni_circle_radius * np.cos(theta), self.uni_circle_radius * np.sin(theta)]
            self.state[2] = self.init_quad_height

        self.state[:2] += np.random.uniform(-1.0, 1.0, size=(2,)) # add noise to x y
        self.state[2] += np.random.uniform(-0.15, 0.15) # add noise to z
        self.state[3:6] += np.random.uniform(-0.2, 0.2, size=(3,)) # add noise to velocity

        # quaternion
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0

        # unicycle
        self.uni_vel = 0.5 # m/s
        self.buffer_steps = 4
        self.uni_vel += np.random.uniform(-0.1, 0.1)
        self.uni_circle_radius += np.random.uniform(-0.1, 0.1)
        self.uni_state = np.zeros((4,)) # x y dx dy the center of the circle is (0, 0) for now
        self.uni_future_pos = np.zeros((2, self.buffer_steps)) # x, y * steps
        self.uni_prev_buffer = np.zeros((2, self.buffer_steps))
        self.relative_pos_fur = np.zeros((2, self.buffer_steps)) # x, y * fur steps
        self.relative_pos_prev = np.zeros((2, self.buffer_steps)) # x, y * prev steps

        # observation
        self.observation = np.concatenate([self.state, self.quaternion, self.relative_pos_prev.flatten('F')]) # update observation ---> # Quad: x y z dx dy dz + q0 q1 q2 q3 + (relative_pos_fur x y * buffer_steps) 
        return self.observation
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)


def uni_animation():
    
    env = QuadrotorEnv(args=None)
    state_list= []
    env.dt = 0.02
    for env.steps in range(env.max_steps):
        pos, vel = env.compute_uni_future_traj(1)
        state = np.concatenate([pos, vel], axis=0).flatten()
        state_list.append(state)
        
    state_list = np.array(state_list)
    print(state_list.shape)

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

    ani = animation.FuncAnimation(fig, update, frames=range(len(state_list)), blit=True, repeat=False)

    ax.set_xlim([min(state_list[:, 0]-0.5), max(state_list[:, 0]+0.5)])
    ax.set_ylim([min(state_list[:, 1]-0.5), max(state_list[:, 1]+0.5)])
    ax.set_aspect('equal')
    plt.show()


def render(quad_state, quad_angles, uni_states, actions):

    x = quad_state[:, 0]
    y = quad_state[:, 1]
    z = quad_state[:, 2]

    roll = quad_angles[:, 0]
    pitch = quad_angles[:, 1]
    yaw = quad_angles[:, 2]
    fx = actions[:, 0]*2
    fy = actions[:, 1]*2
    fz = actions[:, 2]/25

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(quad_state)):
        
        # unicycle
        ax.plot(uni_states[:i, 0], uni_states[:i, 1], 0, 'o', markersize=2, color='dodgerblue', alpha=0.5)

        a = Arrow3D([uni_states[i, 0], uni_states[i, 0] + uni_states[i, 2]/0.3], [uni_states[i, 1], uni_states[i, 1]], [0, 0], mutation_scale=14, lw=1, arrowstyle="->", color="darkorange")
        ax.add_artist(a)
        
        a = Arrow3D([uni_states[i, 0], uni_states[i, 0]], [uni_states[i, 1], uni_states[i, 1] + uni_states[i, 3]/0.3], [0, 0], mutation_scale=14, lw=1, arrowstyle="->", color="fuchsia")
        ax.add_artist(a)
        
        ### x-axis = darkorange, y-axis = fuchsia, z-axis = lightseagreen
        
        # quadrotor
        ax.plot(x[:i], y[:i], z[:i], 'o', markersize=2, color='darkviolet', alpha=0.5)

        a = Arrow3D([x[i], x[i]+fx[i]], [y[i], y[i]], [z[i], z[i]], mutation_scale=14, lw=1, arrowstyle="->", color="darkorange")
        ax.add_artist(a)
        
        a = Arrow3D([x[i], x[i]], [y[i], y[i] + fy[i]], [z[i], z[i]], mutation_scale=14, lw=1, arrowstyle="->", color="fuchsia")
        ax.add_artist(a)
        
        a = Arrow3D([x[i], x[i]], [y[i], y[i]], [z[i], z[i] + fz[i]], mutation_scale=14, lw=1, arrowstyle="->", color="lightseagreen")
        ax.add_artist(a)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_xlim3d(-10,10)
        # ax.set_ylim3d(-10,10)
        # ax.set_zlim3d(0,10)
        ax.set_xlim3d(-4,4)
        ax.set_ylim3d(-4,4)
        ax.set_zlim3d(0,4)
        
        plt.title('Quadrotor trajectory and orientation in 3D')
        plt.draw()
        plt.show(block=False)

        plt.pause(0.01)
        plt.cla()
def video_maker(quad):
    pass


if __name__ == "__main__":


    uni_animation()

