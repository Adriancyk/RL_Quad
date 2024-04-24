# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define cone parameters
# height = 2.5
# radius = 0.22
# num_points = 50

# # Create theta values for the circle base
# theta = np.linspace(0, 2*np.pi, num_points)

# # Create x, y, z coordinates for the cone vertices
# x_base = radius * np.cos(theta)
# y_base = radius * np.sin(theta)
# z_base = np.zeros_like(theta) + height

# x_apex = np.zeros(num_points)
# y_apex = np.zeros(num_points)
# z_apex = np.ones(num_points) * height

# # Plot the cone
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the triangular surface
# ax.plot_trisurf(np.concatenate([x_base, x_apex]),
#                 np.concatenate([y_base, y_apex]),
#                 np.concatenate([z_base, z_apex]),
#                 color='r', alpha=0.5)

# # Set labels and limits
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Cone')

# ax.set_xlim([-radius, radius])
# ax.set_ylim([-radius, radius])
# ax.set_zlim([0, height])

# plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import cv2
# import numpy as np

# # Initialize your plot
# fig, ax = plt.subplots()
# line, = ax.plot([], [])
# ax.set_xlim(0, 2 * np.pi)
# ax.set_ylim(-1, 1)

# # Define your animation update function
# def update(frame):
#     x = np.linspace(0, 2 * np.pi, 100)
#     y = np.sin(x + 2 * np.pi * frame / num_frames)
#     line.set_data(x, y)
#     return line,

# # Set the number of frames
# num_frames = 100
# import matplotlib
# print(matplotlib.animation.writers.list())
# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
# ani.save('output.avi', writer='ffmpeg', fps=30)
# plt.close()
# fig.canvas.draw()
# # Create a VideoWriter object for saving the video
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

# # Convert the animation to video frames and write them to the video file
# for frame in ani.new_frame_seq():
#     # ani.frame_seq = ani.new_frame_seq()  # Reset the frame sequence for each frame
#     fme = ani._draw_frame(frame)  # Draw the current frame
#     fme = np.array(fig.canvas.renderer.buffer_rgba())  # Convert the frame to a numpy array
#     fme = cv2.cvtColor(fme, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR format (required for OpenCV)
#     out.write(fme)  # Write the frame to the video

# # Release the VideoWriter object and close the plot
# out.release()
# plt.close()

import numpy as np
import matplotlib.pyplot as plt

# Define the parameter for the figure-eight pattern
t = np.linspace(0, 2 * np.pi, 400)  # Time parameter
size = 2
# Create the figure-eight function
x = size*np.sin(t)
y = size*np.sin(t)*np.cos(t)  # The figure-eight pattern

# Plot the figure-eight curve
plt.figure(figsize=(6, 6))  # Optional: Set the figure size
plt.plot(x, y, label="Figure 8", color="b")
plt.title("Figure-Eight Pattern")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()


