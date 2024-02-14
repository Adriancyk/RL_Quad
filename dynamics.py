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
        self.state_err_weight = np.diag([16, 16, 0, 7, 7, 0]) # 16 7
        self.act_err_weight = np.diag([0.01, 0.01])
        self.circle_bound_radius = 0.9
        # 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
        # self.reward_goal = 100
        self.reward_exp = True
        low_ext = np.array([
                -self.x_threshold, 
                self.z_ground, 
                -self.theta_threshold_radians, 
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max
            ])

        high_ext = np.array([
                self.x_threshold, 
                self.z_threshold, 
                self.theta_threshold_radians, 
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ])
        self.action_low = np.array([0.0, 0.0, 0.0, -np.pi/4]) # fx fy fz yaw
        self.action_high = np.array([1.0, 1.0, 1.0, np.pi/4]) # fx fy fz yaw
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(4,)) # fx fy fz yaw
        self.observation_space = spaces.Box(low=low_ext, high=high_ext, shape=(12,)) # quad: x y z dx dy dz   Uni: x y z dx dy dz

        # Initialize Env
        self.state = np.zeros((6,))
        self.obs = np.zeros((12,))
        self.uncertainty = np.zeros((6,))
        self.episode_step = 0

        self.reset()




    def step(self, action, use_reward=True):
        # action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self._step(action, use_reward)
        return state, reward, done, info

    def _step(self, action, use_reward=True):
        # x z theta dx dz dtheta
        # Start with the prior for continuous time system x' = f(x) + g(x)u
        # Disturbed continuous time system x' = f(x) + g(x)(u(x) + dm(x)) + d(x)

        self.uncertainty_d = np.zeros(self.state.shape)
        self.uncertainty_dm = np.zeros(action.shape)

        self.uncertainty_d[3] = self.drag_factor * self.state[0] * self.state[0] # mimic drag force disturbance in x direction
        self.uncertainty_d[4] = self.drag_factor * self.state[1] * self.state[1] # mimic drag force disturbance in z direction

        self.uncertainty_dm[0] = self.frac_factor * action[0] # mimic rotor fraction force disturbance in T1 direction
        self.uncertainty_dm[1] = self.frac_factor * action[1] # mimic rotor fraction force disturbance in T2 direction
        self.wind_disturbance.append(self.uncertainty_d)
        self.friction_disturbance.append(self.uncertainty_dm)
        
        self.uncertainty = self.get_g(self.state) @ (self.uncertainty_dm) + self.uncertainty_d
        self.overall_disturbance.append(self.uncertainty)

        self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state) @ (action + self.uncertainty_dm)) + self.state
        self.state = self.dt * self.uncertainty_d + self.state
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

        idx = min(self.episode_step, self.max_episode_steps-1)
        act_err = action - self.u_ref
        ref_traj = np.array([self.x_ref_traj[idx], self.z_ref_traj[idx], self.theta_ref_traj[idx], self.dx_ref_traj[idx], self.dz_ref_traj[idx], self.dtheta_ref_traj[idx]])
        state_err = state - ref_traj
        dist = np.sum(state_err@self.state_err_weight@state_err)
        dist += np.sum(act_err@self.act_err_weight@act_err)
        reward = -dist
        if self.reward_exp:
            reward = np.exp(reward)
        return reward

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
        self.state_nom = np.zeros((6, ))
        self.uncertainty = np.zeros((6,))
        self.obs = np.zeros((12,))

        return self.state, self.obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def _get_dynamics(self):
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
            f_x[4] = -self.g
            return f_x

        def get_g(state):
            g_x = np.zeros((state.shape[0], 2))
            theta = state[2]
            g_x = np.array([[0, 0], [0, 0], [0, 0],
                        [-np.sin(theta)/self.mass, -np.sin(theta)/self.mass],
                        [np.cos(theta)/self.mass, np.cos(theta)/self.mass],
                        [       -self.d/self.Iyy,         self.d/self.Iyy]])
            return g_x

        return get_f, get_g

    
    def get_obs(self, states, step):
        self.ref = np.array([self.x_ref_traj[step], self.z_ref_traj[step], self.theta_ref_traj[step], self.dx_ref_traj[step], self.dz_ref_traj[step], self.dtheta_ref_traj[step]])
        self.obs = np.zeros((12,))
        self.obs[0:6] = states
        self.obs[6:12] = self.ref
        return self.obs
    
    def get_unicycle_state(self):
        pass


if __name__ == "__main__":

    

    env = QuadrotorEnv()
