import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import mosek





class safe_filter():
    def __init__(self, args, dt, mass=2.0, alpha=0.5):
        self.alpha = alpha
        self.d = 1.0 # m
        self.theta = 5/180*np.pi
        self.tan_theta = np.tan(self.theta)
        self.dt = dt
        self.g = 9.81
        self.mass = mass
    
    def get_r(self, x_quad):
        zq = -x_quad[2]
        return (zq + self.d)*self.tan_theta
    
    def get_safe_control(self, x_quad, x_uni, u_rl):
        xq = x_quad[0]
        yq = x_quad[1]
        zq = x_quad[2]
        dxq = x_quad[3]
        dyq = x_quad[4]
        dzq = x_quad[5]

        xu = x_uni[0]
        yu = x_uni[1]
        dxu = x_uni[2]
        dyu = x_uni[3]
        ddxu = x_uni[4]
        ddyu = x_uni[5]

        ddphi0 = ...
        dphi0 = 2*(self.tan_theta**2)*(zq - self.d)*dzq - 2*(xq - xu)*(dxq - dxu) - 2*(yq - yu)*(dyq - dyu)
        phi0 = self.tan_theta**2*(zq - self.d)**2 - (xq - xu)**2 - (yq - yu)**2

        h = matrix(0.0, (7, 1))
        h[0, 0] = 2*(self.tan_theta**2)*((zq - self.d)*self.mass*self.g + dzq**2) - 2*((dxq - dxu)**2 + (dyq - dyu)**2 + (xq - xu)*(-ddxu) + (yq - yu)*(-ddyu)) + 2*self.alpha*dphi0 + self.alpha**2*phi0
        h[1, 0] = 1.0
        h[2, 0] = 1.0
        h[3, 0] = 1.0
        h[4, 0] = 1.0
        h[5, 0] = 0.0
        h[6, 0] = 25.0


        # minimize 0.5*(u-u_rl)^T*P*(u-u_rl) => 0.5*u^T*P*u - u^T*P*u_rl
        P = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        q = matrix(-P@u_rl)

        # Gx <= h
        G = matrix(0.0, (7, u_rl.shape[0]))
        G[0, 0] = -(-2*(xq - xu))
        G[0, 1] = -(-2*(yq - yu))
        G[0, 2] = -(2*(self.tan_theta**2)*(zq - self.d))
        G[1, 0] = 1.0
        G[2, 0] = -1.0
        G[3, 1] = 1.0
        G[4, 1] = -1.0
        G[5, 2] = 1.0
        G[6, 2] = -1.0

        # h = h

        sol = solvers.qp(P, q, G, h, options={'show_progress':False})
        u_safe = np.array(sol['x']).squeeze()
        # print('u_safe:', u_safe)
        # print('u_rl:', u_rl)
        # print(u_safe - u_rl)
        return u_safe


class robust_safe_filter():
    def __init__(self, args, dt, mass=2.0, alpha=1.0):
        self.alpha = alpha
        self.d = 1.0 # m
        self.theta = 5/180*np.pi
        self.tan_theta = np.tan(self.theta)
        self.dt = dt
        self.g = 9.81
        self.mass = mass
        self.bound_param = 0.005
    
    def get_r(self, x_quad):
        zq = -x_quad[2]
        return (zq + self.d)*self.tan_theta
    
    def get_safe_control(self, x_quad_nom, x_uni, sigma_hat, u_rl):
        xq = x_quad_nom[0]
        yq = x_quad_nom[1]
        zq = x_quad_nom[2]
        dxq = x_quad_nom[3]
        dyq = x_quad_nom[4]
        dzq = x_quad_nom[5]

        xu = x_uni[0]
        yu = x_uni[1]
        dxu = x_uni[2]
        dyu = x_uni[3]
        ddxu = x_uni[4]
        ddyu = x_uni[5]

        disturb_x = sigma_hat[3]
        disturb_y = sigma_hat[4]
        disturb_z = sigma_hat[5]

        ddphi0 = ...
        dphi0 = 2*(self.tan_theta**2)*(zq - self.d)*dzq - 2*(xq - xu)*(dxq - dxu) - 2*(yq - yu)*(dyq - dyu)
        phi0 = self.tan_theta**2*(zq - self.d)**2 - (xq - xu)**2 - (yq - yu)**2

        h = matrix(0.0, (7, 1))
        h[0, 0] = 2*(self.tan_theta**2)*((zq - self.d)*(self.mass*self.g + disturb_z) + dzq**2) - 2*((dxq - dxu)**2 + (dyq - dyu)**2 + (xq - xu)*(disturb_x - ddxu) + (yq - yu)*(disturb_y - ddyu)) + 2*dphi0 + phi0 - self.bound_param
        h[1, 0] = 1.0
        h[2, 0] = 1.0
        h[3, 0] = 1.0
        h[4, 0] = 1.0
        h[5, 0] = 0.0
        h[6, 0] = 25.0


        # minimize 0.5*(u-u_rl)^T*P*(u-u_rl) => 0.5*u^T*P*u - u^T*P*u_rl
        P = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        q = matrix(-P@u_rl)

        # Gx <= h
        G = matrix(0.0, (7, u_rl.shape[0]))
        G[0, 0] = -(-2*(xq - xu))
        G[0, 1] = -(-2*(yq - yu))
        G[0, 2] = -(2*(self.tan_theta**2)*(zq - self.d))
        G[1, 0] = 1.0
        G[2, 0] = -1.0
        G[3, 1] = 1.0
        G[4, 1] = -1.0
        G[5, 2] = 1.0
        G[6, 2] = -1.0

        solvers.options['MOSEK'] = {mosek.iparam.log: 0}
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)
        u_safe = np.array(sol['x']).squeeze()
        return u_safe