from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation as R
import numpy as np

def soft_update(target, source, tau):
    # Soft update model parameters.
    # θ_target = τ*θ_local + (1 - τ)*θ_target
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    # Hard update model parameters.
    # θ_target = θ_local
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def prYellow(prt): print("\033[93m {}\033[00m".format(prt))

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