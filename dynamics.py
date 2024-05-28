import numpy as np
import gym
from gym import spaces
from pyquaternion import Quaternion
import torch
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class QuadrotorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, args, mass=None, wind=None):

        super(QuadrotorEnv, self).__init__()
        # Using North-East-Down (NED) coordinate system
        if args is None:
            self.control_mode = 'tracking'
        else:
            self.control_mode = args.control_mode
        self.wind = wind
        self.args = args
        self.get_f, self.get_g = self.get_dynamics()
        self.mass = 2.0
        if mass is not None:
            self.mass = mass
        self.g = 9.81
        self.z_ground = 0.0
        self.dt = 0.02 # 50Hz
        self.max_steps = 2000
        self.reward_exp = True
        self.steps = 0
        self.init_uni_angle = np.random.uniform(0, 2*np.pi, size=(1,)).item()# * 0 if self.args.mode == 'test' else 1
        self.desired_yaw = 0.0 # quadrotor yaw angle
        self.uni_circle_radius = 1.5 # m
        self.desired_hover_height = ... # quadrotor hover height
        self.init_quad_height = ...
        self.last_action = np.zeros((3,))


        # quadrotor
        self.action_low = np.array([-1.0, -1.0, -25.0]) # fx fy fz
        self.action_high = np.array([1.0, 1.0, 0.0]) # fx fy fz
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(3,)) # fx fy fz

        self.bounded_state_space = spaces.Box(low=np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10]), high=np.array([10.0, 10.0, 2.0, 10.0, 10.0, 10.0]), shape=(6,)) # x y z dx dy dz

        # set initial height and desired height for control modes
        if self.control_mode == 'takeoff': # takeoff vertically
            self.init_quad_height = 0.0
            self.desired_hover_height = -1.0
        elif self.control_mode == 'tracking': # tracking a unicycle on the ground
            self.init_quad_height = -1.0
            self.desired_hover_height = -1.0
        elif self.control_mode == 'landing': # landing vertically to the ground
            self.init_quad_height = -1.0
            self.desired_hover_height = 0.0
        elif self.control_mode == 'dynamic_soft_landing': # landing dynamically on a moving unicycle at a certain height
            self.init_quad_height = -1.0
            self.desired_hover_height = -0.3
        elif self.control_mode == 'dynamic_chasing': # chasing a moving unicycle at a random height
            self.init_quad_height = -1.0
            self.desired_hover_height = -np.random.uniform(0.0, 1.5)
        
        # set initial state
        self.state = np.zeros((6,)) # quad: x y z dx dy dz
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3

        # unicycle
        self.uni_vel = 0.5 # m/s
        self.buffer_steps = 4
        self.uni_state = np.zeros((4,)) # x y dx dy the center of the circle is (0, 0) for now
        # self.uni_future_pos = np.zeros((2, self.buffer_steps)) # x, y * steps
        self.uni_prev_buffer = np.zeros((2, self.buffer_steps))
        # self.relative_pos_fur = np.zeros((2, self.buffer_steps)) # x, y * fur steps
        self.relative_pos_prev = np.zeros((2, self.buffer_steps)) # x, y * prev steps
        self.relative_height = np.array([self.desired_hover_height - self.state[2]])
        # observation
        self.observation = np.concatenate([self.state, self.quaternion, self.relative_pos_prev.flatten('F'), self.relative_height]) # update observation ---> # Quad: x y z dx dy dz + q0 q1 q2 q3 + (relative_pos_fur x y * buffer_steps) + relative_height in z axis

        self.reset()

    def step(self, action, use_reward=True):
        # mix the states + q + uni_states to observation
        state, reward, done, info = self._step(action, use_reward) # t+1
        self.last_action = action
        uni_state = self.get_unicycle_state(self.steps, shape=self.args.traj) # t+1
        # self.uni_future_pos, _ = self.compute_uni_future_traj(self.buffer_steps) # t+1
        
        # Shift the elements in the buffer to the right
        self.uni_prev_buffer = np.roll(self.uni_prev_buffer, shift=1, axis=1)
        # Insert the new state at the first position
        self.uni_prev_buffer[:, 0] = uni_state[:2]

        # self.relative_pos_prev = self.uni_prev_buffer - state[:2].reshape(-1, 1) + np.random.uniform(-0.02, 0.02, size=(2,4)) * 0 if self.args.mode == 'test' else 1 # t+1
        self.relative_pos_prev = self.uni_prev_buffer - state[:2].reshape(-1, 1) + np.random.uniform(-0.02, 0.02, size=(2, self.buffer_steps)) * (0 if self.args.mode == 'test' else 1) # t+1

        if self.control_mode == 'takeoff' or self.control_mode == 'landing':
            self.relative_pos_prev = np.zeros_like(self.relative_pos_prev)

        self.relative_height = np.array([self.desired_hover_height - state[2]])

        self.observation = np.concatenate([state, self.quaternion, self.relative_pos_prev.flatten('F'), self.relative_height]) # t+1
        
        return self.observation, reward, done, info

    def _step(self, action, use_reward=True):
        # x y z dx dy dz without no attitude
        # self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state) @ self.last_action) + self.state # t+1
        # self.desired_attitude(self.last_action) # t+1
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
        uni_state = self.get_unicycle_state(self.steps, shape=self.args.traj) # t+1
        self.uni_prev_buffer = np.roll(self.uni_prev_buffer, shift=1, axis=1)
        self.uni_prev_buffer[:, 0] = uni_state[:2]

        self.desired_attitude(self.last_action) # t+1
        # self.uni_future_pos, _ = self.compute_uni_future_traj(self.buffer_steps) # t+1
        # rel_pos_fur = self.uni_future_pos - state[:2].reshape(-1, 1)
        rel_pos_fur = []
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

        if f_total == 0 or f_z == 0:
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
            q = Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(axis=[0, 1, 0], angle=pitch) * Quaternion(axis=[1, 0, 0], angle=roll)
            self.quaternion = np.array([q[0], q[1], q[2], q[3]])
            return self.quaternion, roll, pitch, yaw

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
            distance = np.linalg.norm(relative_pos[:, 0], axis=-1) # distance
            reward += -1.2*np.sum(distance**2)
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
        elif self.control_mode == 'landing': # land vertically
            reward += -2*(self.desired_hover_height - state[2])**2 # z position difference
            reward += -(np.sqrt(state[3]**2 + state[4]**2 + state[5]**2))/np.exp(abs(3*(self.desired_hover_height - state[2])))
        elif self.control_mode == 'dynamic_soft_landing':
            reward += -2.0*(self.desired_hover_height - state[2])**2
            distance = np.linalg.norm(relative_pos[:, 0], axis=-1)
            reward += -1.2*np.sum(distance**2)
            reward += -(np.sqrt(0*state[3]**2 + 0*state[4]**2 + state[5]**2))/np.exp(abs(3*(self.desired_hover_height - state[2])))
            # reward += -2*(np.abs(action[2]) - 22)**2 if np.abs(action[2]) > 22.00 else 0      
        elif self.control_mode == 'dynamic_chasing':
            distance = np.linalg.norm(relative_pos[:, 0], axis=-1)
            reward += -1.2*np.sum(distance**2)
            reward += -2*(self.desired_hover_height - state[2])**2
        
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
            f_x[3] += np.random.uniform(-self.wind, self.wind) if self.wind is not None else 0
            f_x[4] += np.random.uniform(-self.wind, self.wind) if self.wind is not None else 0
            f_x[5] = self.g*self.mass + (np.random.uniform(-self.wind, self.wind) if self.wind is not None else 0) # add wind
            return f_x

        def get_g(state):
            g_x = np.zeros((state.shape[0], 3))  # 6x3
            g_x[3:, :] = np.eye(3)
            return g_x

        return get_f, get_g
    
    def get_out(self):
        # check if the quadrotor is out of boundary
        mask = np.array([1, 1, 1, 0, 0, 0]) # 1 means check, 0 means not check
        out_of_bound = np.logical_or(self.state < self.bounded_state_space.low, self.state > self.bounded_state_space.high)
        out_of_bound = np.any(out_of_bound * mask)
        if out_of_bound:
            return True
        return False
    
    def get_unicycle_state(self, steps=0, shape=None, scale=2.0):
        # default is circle
        uni_state = np.zeros((4,))
        ang_vel = self.uni_vel / self.uni_circle_radius
        theta = ang_vel * steps * self.dt + self.init_uni_angle
        uni_state[0] = self.uni_circle_radius * np.cos(theta) # x
        uni_state[1] = self.uni_circle_radius * np.sin(theta) # y
        uni_state[2] = -self.uni_vel * np.sin(theta) # dx
        uni_state[3] = self.uni_vel * np.cos(theta) # dy

        if shape is not None and shape=='figure8': # figure 8
            uni_state = np.zeros((4,))
            ang_vel = self.uni_vel / self.uni_circle_radius
            theta = ang_vel * steps * self.dt + self.init_uni_angle
            uni_state[0] = scale*np.sin(theta) # x
            uni_state[1] = scale*np.sin(theta)*np.cos(theta) # y
            uni_state[2] = scale*ang_vel*np.cos(theta) # dx
            uni_state[3] = scale*ang_vel*(np.cos(theta)**2 - np.sin(theta)**2) # dy
        
        if shape is not None and shape=='triangle':
            uni_state = np.zeros((4,))
            ang_vel = self.uni_vel / self.uni_circle_radius
            theta = ang_vel * steps * self.dt + self.init_uni_angle

        
        return uni_state
    
    # def compute_uni_future_traj(self, future_steps, dt=None):
    #     uni_future_pos = []
    #     uni_furture_vel = []
    #     for i in range(future_steps):
    #         cur = self.get_unicycle_state(steps=self.steps + i) 
    #         uni_future_pos.append(cur[:2])
    #         uni_furture_vel.append(cur[2:])
    #     return np.array(uni_future_pos).T, np.array(uni_furture_vel).T

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
        elif self.control_mode == 'dynamic_soft_landing':
            theta = np.random.uniform(0, 2*np.pi) # * 0 if self.args.mode == 'test' else 1
            self.state[:2] = [self.uni_circle_radius * np.cos(theta), self.uni_circle_radius * np.sin(theta)]
            self.state[2] = self.init_quad_height
        elif self.control_mode == 'dynamic_chasing':
            theta = np.random.uniform(0, 2*np.pi)
            self.state[2] = self.init_quad_height
            self.desired_hover_height = -np.random.uniform(0.0, 1.3)
            self.state[:2] = [self.uni_circle_radius * np.cos(theta), self.uni_circle_radius * np.sin(theta)]

        self.state[:2] += np.random.uniform(-1.0, 1.0, size=(2,)) * 0 if self.args.mode == 'test' else 1 # add noise to x y
        self.state[2] += np.random.uniform(-0.15, 0.15)  * 0 if self.args.mode == 'test' else 1 # add noise to z
        self.state[3:6] += np.random.uniform(-0.2, 0.2, size=(3,)) * 0 if self.args.mode == 'test' else 1 # add noise to velocity

        # quaternion
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0

        # unicycle
        self.uni_vel = 0.5 # m/s
        self.uni_circle_radius = 1.5 # m
        self.buffer_steps = 4
        self.init_uni_angle = np.random.uniform(0, 2*np.pi, size=(1,)).item()
        self.uni_vel += np.random.uniform(-0.2, 0.2) * 0 if self.args.mode == 'test' else 1
        self.uni_circle_radius += np.random.uniform(-0.2, 0.2) * 0 if self.args.mode == 'test' else 1
        self.uni_state = np.zeros((4,)) # x y dx dy the center of the circle is (0, 0) for now
        # self.uni_future_pos = np.zeros((2, self.buffer_steps)) # x, y * steps
        self.uni_prev_buffer = np.zeros((2, self.buffer_steps))
        # self.relative_pos_fur = np.zeros((2, self.buffer_steps)) # x, y * fur steps
        self.relative_pos_prev = np.zeros((2, self.buffer_steps)) # x, y * prev steps
        self.relative_height = np.array([self.desired_hover_height - self.state[2]])
        # observation
        self.observation = np.concatenate([self.state, self.quaternion, self.relative_pos_prev.flatten('F'), self.relative_height]) # update observation ---> # Quad: x y z dx dy dz + q0 q1 q2 q3 + (relative_pos_fur x y * buffer_steps) 
        return self.observation
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

if __name__=="__main__":
    uni_state = np.zeros((4,))
    uni_vel = 0.5
    uni_circle_radius = 1.5
    size = 2
    dt = 0.02
    uni_list = []
    init_uni_angle = 0

    for steps in range(1000):
        uni_state = np.zeros((4,))
        ang_vel = uni_vel / uni_circle_radius
        theta = ang_vel * steps * dt + init_uni_angle
        uni_state[0] = size*np.sin(theta) # x
        uni_state[1] = size*np.sin(theta)*np.cos(theta) # y
        uni_state[2] = size*ang_vel*np.cos(theta) # dx
        uni_state[3] = size*ang_vel*(np.cos(theta)**2 - np.sin(theta)**2) # dy
        uni_list.append(uni_state)

    uni_list = np.array(uni_list)
    fig = plt.figure()
    plt.plot(uni_list[:, 0], uni_list[:, 1])
    plt.scatter(uni_list[-1, 0], uni_list[-1, 1], c='r')
    plt.show()
    
