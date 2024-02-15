from dis import dis
import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
import torch
import os
# from PD import PD_controller
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class QuadrotorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(QuadrotorEnv, self).__init__()

        self.dynamics_mode = 'Quadrotor'
        self.get_f, self.get_g = self._get_dynamics()
        self.mass = 0.027
        self.Ixx = 1.4e-5
        self.Iyy = 1.4e-5
        self.Izz = 2.17e-5
        self.g = 9.81
        self.L = 0.046
        self.d = self.L/np.sqrt(2)
        self.x_threshold = 2.0
        self.z_threshold = 2.5
        self.z_ground = 0.0
        self.dt = 0.01
        self.max_episode_steps = 400
        self.uni_circle_radius = 1.0
        self.uni_vel = 0.5
        self.reward_exp = True

        self.action_low = np.array([0.0, 0.0, 0.0]) # fx fy fz yaw
        self.action_high = np.array([1.0, 1.0, 1.0]) # fx fy fz yaw
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(3,)) # fx fy fz yaw

        # Initialize Env
        self.state = np.zeros((6,)) # x y z dx dy dz
        self.quaternion = np.zeros((4,)) # q0 q1 q2 q3
        self.quaternion[0] = 1.0
        self.state_uni = np.zeros((4,)) # x y dx dy
        self.state_uni[0] = self.uni_circle_radius # initial x at (1, 0)
        self.episode_step = 0

        self.reset()




    def step(self, action, use_reward=True):
        # action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self._step(action, use_reward)
        return state, reward, done, info

    def _step(self, action, use_reward=True):
        # x y z dx dy dz
        # self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action) + self.state  # save for later
        self.state ;
        self.episode_step += 1
        reward = 0.0
        

        info = dict()
        if use_reward:
            reward = self.get_reward(self.state, action)
        if self.get_done():
            info['out_of_bound'] = True
            reward += -100
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps
            info['reach_max_steps'] = True

        return self.state, reward, done, info

    def get_reward(self, state, action):

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
        theta = ang_vel * self.episode_step * self.dt

    def get_done(self):

        mask = np.array([1, 1, 1, 0, 0, 0])
        out_of_bound = np.logical_or(self.state < self.bounded_state_space.low, self.state > self.bounded_state_space.high)
        out_of_bound = np.any(out_of_bound * mask)
        if out_of_bound:
            return True
        return False

    def reset(self):
        self.episode_step = 0
        self.state = np.zeros((6, ))
        self.obs = np.zeros((12,))

        return self.state, self.obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    


if __name__ == "__main__":

    

    env = QuadrotorEnv()
