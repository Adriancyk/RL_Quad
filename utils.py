from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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



# def uni_animation():
    
#     env = QuadrotorEnv(args=None)
#     state_list= []
#     env.dt = 0.02
#     for env.steps in range(env.max_steps):
#         state = env.get_unicycle_state(env.steps, shape='figure8', size=2)
#         state_list.append(state)
#     state_list = np.array(state_list)

#     fig, ax = plt.subplots()
#     line, = ax.plot(state_list[0, 0], state_list[0, 1])
#     quiver_x = ax.quiver(state_list[0, 0], state_list[0, 1], state_list[0, 2], 0, color='pink', scale=env.uni_vel/0.3)
#     quiver_y = ax.quiver(state_list[0, 0], state_list[0, 1], 0, state_list[0, 3], color='b', scale=env.uni_vel/0.3)
#     quiver_total = ax.quiver(state_list[0, 0], state_list[0, 1], state_list[0, 2], state_list[0, 3], color='g', scale=env.uni_vel/0.3)

#     def update(frame):
#         line.set_data(state_list[:frame, 0], state_list[:frame, 1])
#         quiver_x.set_UVC(state_list[frame, 2], 0)
#         quiver_y.set_UVC(0, state_list[frame, 3])
#         quiver_total.set_UVC(state_list[frame, 2], state_list[frame, 3])
#         quiver_x.set_offsets(state_list[frame, :2])
#         quiver_y.set_offsets(state_list[frame, :2])
#         quiver_total.set_offsets(state_list[frame, :2])
#         return line, quiver_x, quiver_y, quiver_total,

#     ani = animation.FuncAnimation(fig, update, frames=range(len(state_list)), blit=True, repeat=False)

#     ax.set_xlim([min(state_list[:, 0]-0.5), max(state_list[:, 0]+0.5)])
#     ax.set_ylim([min(state_list[:, 1]-0.5), max(state_list[:, 1]+0.5)])
#     ax.set_aspect('equal')
#     plt.show()


def render(quad_state, quad_angles, uni_states, actions, enable_cone=True):

    x = quad_state[:, 0]
    y = quad_state[:, 1]
    z = quad_state[:, 2]

    vx = quad_state[:, 3]
    vy = quad_state[:, 4]
    vz = quad_state[:, 5]

    roll = quad_angles[:, 0]
    pitch = quad_angles[:, 1]
    yaw = quad_angles[:, 2]

    fx = actions[:, 0]*2
    fy = actions[:, 1]*2
    fz = actions[:, 2]/25

    in_safe_set = False

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    
    

    for i in range(len(quad_state)):
        pos_x = ax.text2D(0.00, 0.95, "", transform=ax.transAxes, color='darkorange', weight='bold') # position text
        pos_y = ax.text2D(0.00, 0.90, "", transform=ax.transAxes, color='fuchsia', weight='bold') # position text
        pos_z = ax.text2D(0.00, 0.85, "", transform=ax.transAxes, color='lightseagreen', weight='bold') # position text
        # vel_text = ax.text2D(0.00, 0.80, "", transform=ax.transAxes) # velocity text
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

        if enable_cone:
            # Define cone parameters
            height = 2.5
            radius = 0.26
            num_points = 50
            d = 1.0 # offset from the peak to unicycle

            # Create theta values for the circle base
            theta = np.linspace(0, 2*np.pi, num_points)

            # Create x, y, z coordinates for the cone vertices
            x_base = radius * np.cos(theta) + uni_states[i, 0]
            y_base = radius * np.sin(theta) + uni_states[i, 1]
            z_base = np.zeros_like(theta) + height - d

            x_apex = np.ones(num_points) * uni_states[i, 0]
            y_apex = np.ones(num_points) * uni_states[i, 1]
            z_apex = np.ones(num_points) * -d

            # Plot the triangular surface
            safe_radius = (z[i] + d) * np.tan(5/180*np.pi)
            distance = np.linalg.norm([x[i] - uni_states[i, 0], y[i] - uni_states[i, 1]])

            color = 'k' # black if cbf is not enabled
            if in_safe_set is False and distance + 0.015 <= safe_radius:
                in_safe_set = True # cbf is enabled
            if distance >= safe_radius and in_safe_set is True:
                color = 'r' # red if cbf is enabled but violated
            if distance <= safe_radius and in_safe_set is True:
                color = 'g' # green if cbf is enabled and satisfied

            ax.plot_trisurf(np.concatenate([x_base, x_apex]),
                            np.concatenate([y_base, y_apex]),
                            np.concatenate([z_base, z_apex]),
                            color=color, alpha=0.2)
            
        pos_x.set_text(f"X = {x[i]:.2f} m, Vx = {vx[i]:.2f} m/s")
        pos_y.set_text(f"Y = {y[i]:.2f} m, Vy = {vy[i]:.2f} m/s")
        pos_z.set_text(f"Z = {z[i]:.2f} m, Vz = {vz[i]:.2f} m/s")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_xlim3d(-10,10)
        # ax.set_ylim3d(-10,10)
        # ax.set_zlim3d(0,10)
        ax.set_xlim3d(-4,4)
        ax.set_ylim3d(-4,4)
        ax.set_zlim3d(0,4)
        
        plt.title('Quadrotor trajectory in 3D')
        plt.draw()
        plt.show(block=False)
        plt.pause(0.01)
        plt.cla()

def render_video(quad_state, quad_angles, uni_states, actions, enable_cone=True):

    x = quad_state[:, 0]
    y = quad_state[:, 1]
    z = quad_state[:, 2]

    roll = quad_angles[:, 0]
    pitch = quad_angles[:, 1]
    yaw = quad_angles[:, 2]
    fx = actions[:, 0]*2
    fy = actions[:, 1]*2
    fz = actions[:, 2]/25

    in_safe_set = False

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(-4,4)
    ax.set_ylim3d(-4,4)
    ax.set_zlim3d(0,4)

    plt.title('Quadrotor trajectory in 3D')

    def update(frame):
        ax.cla()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim3d(-4,4)
        ax.set_ylim3d(-4,4)
        ax.set_zlim3d(0,4)
        plt.title('Quadrotor trajectory in 3D')
        artists = []
        nonlocal in_safe_set
        # unicycle
        ax.plot(uni_states[:frame, 0], uni_states[:frame, 1], 0, 'o', markersize=2, color='dodgerblue', alpha=0.5)
        # artists.append(line1)

        a = Arrow3D([uni_states[frame, 0], uni_states[frame, 0] + uni_states[frame, 2]/0.3], [uni_states[frame, 1], uni_states[frame, 1]], [0, 0], mutation_scale=14, lw=1, arrowstyle="->", color="darkorange")
        ax.add_artist(a)
        artists.append(a)
        
        a = Arrow3D([uni_states[frame, 0], uni_states[frame, 0]], [uni_states[frame, 1], uni_states[frame, 1] + uni_states[frame, 3]/0.3], [0, 0], mutation_scale=14, lw=1, arrowstyle="->", color="fuchsia")
        ax.add_artist(a)
        artists.append(a)
        ### x-axis = darkorange, y-axis = fuchsia, z-axis = lightseagreen
        # quadrotor
        ax.plot(x[:frame], y[:frame], z[:frame], 'o', markersize=2, color='darkviolet', alpha=0.5)
        # artists.append(line2)

        a = Arrow3D([x[frame], x[frame]+fx[frame]], [y[frame], y[frame]], [z[frame], z[frame]], mutation_scale=14, lw=1, arrowstyle="->", color="darkorange")
        ax.add_artist(a)
        artists.append(a)
        
        a = Arrow3D([x[frame], x[frame]], [y[frame], y[frame] + fy[frame]], [z[frame], z[frame]], mutation_scale=14, lw=1, arrowstyle="->", color="fuchsia")
        ax.add_artist(a)
        artists.append(a)
        
        a = Arrow3D([x[frame], x[frame]], [y[frame], y[frame]], [z[frame], z[frame] + fz[frame]], mutation_scale=14, lw=1, arrowstyle="->", color="lightseagreen")

        ax.add_artist(a)
        artists.append(a)
        if enable_cone:
            # Define cone parameters
            height = 2.5
            radius = height * np.tan(6/180*np.pi)
            num_points = 50
            d = 1.0 # offset from the peak to unicycle

            # Create theta values for the circle base
            theta = np.linspace(0, 2*np.pi, num_points)

            # Create x, y, z coordinates for the cone vertices
            x_base = radius * np.cos(theta) + uni_states[frame, 0]
            y_base = radius * np.sin(theta) + uni_states[frame, 1]
            z_base = np.zeros_like(theta) + height - d

            x_apex = np.ones(num_points) * uni_states[frame, 0]
            y_apex = np.ones(num_points) * uni_states[frame, 1]
            z_apex = np.ones(num_points) * -d

            # Plot the triangular surface
            safe_radius = (z[frame] + d) * np.tan(5/180*np.pi)
            distance = np.linalg.norm([x[frame] - uni_states[frame, 0], y[frame] - uni_states[frame, 1]])

            color = 'k' # black if cbf is not enabled
            if in_safe_set is False and distance + 0.015 <= safe_radius:
                in_safe_set = True # cbf is enabled
            if distance >= safe_radius and in_safe_set is True:
                color = 'r' # red if cbf is enabled but violated
            if distance <= safe_radius and in_safe_set is True:
                color = 'g' # green if cbf is enabled and satisfied

            surf = ax.plot_trisurf(np.concatenate([x_base, x_apex]),
                            np.concatenate([y_base, y_apex]),
                            np.concatenate([z_base, z_apex]),
                            color=color, alpha=0.2)
            artists.append(surf)


        return artists
        
        # plt.pause(0.01)
    # run following commands to install ffmpeg before use video generator
    # pip install ffmpeg-downloader
    # ffdl install --add-path
    ani = animation.FuncAnimation(fig, update, frames=range(len(quad_state)), blit=True, repeat=False)

    # plt.draw()
    # plt.show()

    ani.save('RCBF_on.mp4', writer='ffmpeg', fps=30)