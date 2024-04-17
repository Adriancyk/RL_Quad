import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define cone parameters
height = 2.5
radius = 0.22
num_points = 50

# Create theta values for the circle base
theta = np.linspace(0, 2*np.pi, num_points)

# Create x, y, z coordinates for the cone vertices
x_base = radius * np.cos(theta)
y_base = radius * np.sin(theta)
z_base = np.zeros_like(theta) + height

x_apex = np.zeros(num_points)
y_apex = np.zeros(num_points)
z_apex = np.ones(num_points) * height

# Plot the cone
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the triangular surface
ax.plot_trisurf(np.concatenate([x_base, x_apex]),
                np.concatenate([y_base, y_apex]),
                np.concatenate([z_base, z_apex]),
                color='r', alpha=0.5)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cone')

ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([0, height])

plt.show()

