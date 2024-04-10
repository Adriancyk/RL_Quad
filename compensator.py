import numpy as np
import scipy.linalg as la
import argparse

class LTISystem(object): 
    # for safe control rendering
    def __init__(self,A,B,C,D=0.0):
        """Initialize the system."""
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        m = self.A.shape[0]
        self.x = np.zeros((m,))

    def get_next_state(self, u, dt=0.02):
        print(self.A@self.x)
        print(self.B@u)
        state_change = self.A@self.x + self.B@u
        self.x += dt*state_change
        output = self.C.dot(self.x) + self.D*u
        return output

class compensator():
    def __init__(self, x, args, Ts):
        self.Ts = Ts
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.x = x
        self.x_hat = self.x
        self.x_tilde = np.zeros_like(self.x)
        self.wc = args.wc
        self.As = args.a_param * np.eye(self.obs_dim)
        mat_expm = la.expm(self.As*self.Ts)
        Phi = la.inv(self.As) @ (mat_expm - np.eye(self.obs_dim))
        self.adapt_gain = -la.inv(Phi) @ mat_expm

        self.lpf = LTISystem(A=-self.wc*np.eye(self.act_dim), B=np.eye(self.act_dim), C=self.wc*np.eye(self.act_dim))
        

    def state_predictor(self, f, g, g_prep, u, sigma_hat_m, sigma_hat_um):

        x_hat_dot = f + g@(u + sigma_hat_m) + np.matmul(g_prep, sigma_hat_um) + np.matmul(self.As, self.x_tilde)
        x_hat = self.x_hat + x_hat_dot * self.Ts

        return x_hat

    def adaptive_law(self, x_tilde, gg):
        sigma_hat = self.adapt_gain@x_tilde
        temp = la.inv(gg)@sigma_hat
        sigma_hat_m = temp[:self.act_dim]
        sigma_hat_um = temp[self.act_dim:]

        return sigma_hat_m, sigma_hat_um, sigma_hat 

    def control_law(self, sigma_hat_m):
        u_l1 = -self.lpf.get_next_state(sigma_hat_m, self.Ts)
        return u_l1

    def get_safe_control(self, x, u_bl, f, g):

        g_perp = la.null_space(g.T)
        gg = np.concatenate((g, g_perp), axis=1)
        self.x_tilde = self.x_hat - x

        sigma_hat_m, sigma_hat_um, sigma_hat = self.adaptive_law(self.x_tilde, gg)
        u_l1 = self.control_law(sigma_hat_m)
        u = u_bl + u_l1
        u[0] = np.clip(u[0], -1.0, 1.0)
        u[1] = np.clip(u[1], -1.0, 1.0)
        u[2] = np.clip(u[2], -35.0, 0.0)

        self.x_hat = self.state_predictor(f, g, g_perp, u, sigma_hat_m, sigma_hat_um)

        return u