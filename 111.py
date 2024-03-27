import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import time
from scipy.spatial.transform import Rotation as R


########## axes draw class
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


# Generate some example data

t = np.linspace(0, 10, 1001)
x = np.cos(2*np.pi*t)
y = np.sin(2*np.pi*t)
z = 0.4 * t

roll = 0 * t
pitch = 0 * t
yaw = 2*np.pi*t


################################
#plotting 
################################    

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')
for i in range(1001):
    ax.plot(x[:i], y[:i], z[:i], 'o', markersize=2, color='red', alpha=0.5)

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
    ax.set_xlim3d(-3,3)
    ax.set_ylim3d(-3,3)
    ax.set_zlim3d(0,8)
    
    plt.title('Eigenvectors')
    plt.draw()
    plt.show(block=False)

    plt.pause(0.01)
    plt.cla()