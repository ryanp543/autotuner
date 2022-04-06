"""
Python program that uses scipy for generating a valid arm trajectory
A few useful resources
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html#scipy.optimize.NonlinearConstraint
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
https://docs.scipy.org/doc/scipy/reference/optimize.html

Most importantly, we note that nonlinear constraints are only supported with COBYLA, SLSQP and trust-constr solvers.

Note: To set desired orientations, use

Date: 2/2/2021
"""

from __future__ import print_function

import numpy as np
import csv
import time
import math
import scipy.optimize
import matplotlib.pyplot as plt
import klampt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FILEPATH_TRAJECTORY = './DynamicsInteractionTrajectory_interact.csv'


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def update_arrow_pos(self, xs3d, ys3d, zs3d):
        self._verts3d = xs3d, ys3d, zs3d

    def update_arrow_properties(self, lw, clr):
        self.set(color=clr, linewidth=lw)


def ExtractTrajectoryData():
    # Extracts data and puts into list
    with open(FILEPATH_TRAJECTORY, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)

    dt = float(data[0][0])
    test_length = float(data[0][1])

    # Create empty list
    x = np.asarray([data[1][:]])
    x = x.astype(np.float)

    for k in range(2, len(data)):
        newrow = np.asarray([data[k][:]])
        newrow = newrow.astype(np.float)
        x = np.append(x, newrow, axis=0)

    # Joint positions, time step, and length of test time
    return x, dt, test_length


def position():
    x = np.asarray([
                   link1.getWorldPosition([0, 0, 0])[0],
                   link2.getWorldPosition([0, 0, 0])[0],
                   link3.getWorldPosition([0, 0, 0])[0],
                   link4.getWorldPosition([0, 0, 0])[0]])
    y = np.asarray([
                   link1.getWorldPosition([0, 0, 0])[1],
                   link2.getWorldPosition([0, 0, 0.05275])[1],
                   link3.getWorldPosition([0, 0, 0])[1],
                   link4.getWorldPosition([0, 0, 0])[1]])
    z = np.asarray([
                   link1.getWorldPosition([0, 0, 0])[2],
                   link2.getWorldPosition([0, 0, 0])[2],
                   link3.getWorldPosition([0, 0, 0])[2],
                   link4.getWorldPosition([0, 0, 0])[2]])

    return (x, y, z)


def getNewTime(frame):
    return frame * dt


def drawChassis(x_conf):
    # Set z pitch and roll based on configuration
    z = x_conf[2]
    pitch = x_conf[4]
    roll = -x_conf[5]

    # Draw chassis body
    points = np.array([[-0.1, -0.2, 0.116],
                       [0.25, -0.2, 0.116],
                       [0.25, 0.2, 0.116],
                       [-0.1, 0.2, 0.116],
                       [-0.1, -0.2, 0.196],
                       [0.25, -0.2, 0.196],
                       [0.25, 0.2, 0.196],
                       [-0.1, 0.2, 0.196]])

    # Rotation and translation matrix
    Rot_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                      [0, 1, 0],
                      [-math.sin(pitch), 0, math.cos(pitch)]])
    Rot_x = np.array([[1, 0, 0],
                      [0, math.cos(roll), -math.sin(roll)],
                      [0, math.sin(roll), math.cos(roll)]])
    Rot = np.matmul(Rot_x, Rot_y)

    trans_z = np.tile([0, 0, z], (np.shape(points)[0], 1))

    # Rotation and translation of vertices
    Z = np.zeros((8, 3))
    for i in range(8):
        Z[i, :] = np.dot(points[i, :], Rot)
    Z = Z + trans_z

    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]]]

    return verts


def animate(frame):
    # Step to the next joint positions
    k = frame * mult
    x_config = np.concatenate(([0.0, 0.0], x[k, 0], 0, x[k, 1:], 0), axis=None)
    robot.setConfig(x_config)

    # Change chassis
    vertices = drawChassis(x_config)
    collection.set_verts(vertices)
    collection.do_3d_projection(collection.axes.get_figure().canvas.get_renderer())

    # Calculate the new positions
    coords = position()
    line.set_data(coords[:2])
    line.set_3d_properties(coords[2])

    # Change the timer text
    time_text.set_text('time = %.1f' % getNewTime(k))

    # Update end effector environment force arrow positions
    ee_arrow.update_arrow_pos(xs3d=[coords[0][3], contact[0]], ys3d=[coords[1][3], contact[1]],
                              zs3d=[coords[2][3], contact[2]])
    # mag = np.linalg.norm(np.array([coords[0][3]-contact[0], coords[1][3]-contact[1], coords[2][3]-contact[2]]))
    # new_lw = 4*mag/task_rad
    # ee_arrow.update_arrow_properties(lw=new_lw, clr='red')

    # Update spring arrow positions
    for n in range(len(sus_arrows)):
        sus_arrows[n].update_arrow_pos(xs3d=[vertices[0][n][0], vertices[0][n][0]],
                                          ys3d=[vertices[0][n][1], vertices[0][n][1]],
                                          zs3d=[0, vertices[0][n][2] + 0.01])

    return sus_arrows[0], sus_arrows[1], sus_arrows[2], sus_arrows[3], collection, line, time_text, ee_arrow


def init():
    for n in range(len(sus_arrows)):
        sus_arrows[n].update_arrow_pos(xs3d=[0,0], ys3d=[0,0], zs3d=[0,0])

    ax.add_collection3d(collection)
    line.set_data([], [])
    line.set_3d_properties([])
    time_text.set_text('')
    ee_arrow.update_arrow_pos(xs3d=[contact[0], contact[0]], ys3d=[contact[1], contact[1]],
                              zs3d=[contact[2], contact[2]])

    return sus_arrows[0], sus_arrows[1], sus_arrows[2], sus_arrows[3], collection, line, time_text, ee_arrow


if __name__ == "__main__":
    world = klampt.WorldModel()
    res = world.readFile("./robot_sim.xml")
    robot = world.robot(0)

    link1 = robot.link(6)
    link2 = robot.link(7)
    link3 = robot.link(8)
    link4 = robot.link(9)

    print("Extracting trajectory from simulation .csv file...")
    x, dt, test_length = ExtractTrajectoryData()

    print("Running simulation visualization...")
    # Joint coords: [z pitch roll 3DOF]
    # x_config = np.zeros(10)
    x_config = np.concatenate(([0.0, 0.0], x[0,0], 0, x[0,1:], 0), axis=None)
    robot.setConfig(x_config)
    contact = link4.getWorldPosition([0, 0, 0])

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=-60)
    time_text = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)

    # Get vertices of the chassis
    vertices = drawChassis(x_config)

    # Spring arrows. Vertices[0] is the corners of the bottom square face
    sus_arrows = []
    for k in range(len(vertices[0])):
        new_spring = Arrow3D([vertices[0][k][0], vertices[0][k][0]], [vertices[0][k][1], vertices[0][k][1]],
                             [0, vertices[0][k][2]+0.01], lw=1, mutation_scale=10, color='green')
        sus_arrows.append(new_spring)
        ax.add_artist(sus_arrows[k])

    # Drawing robotic arm skeleton
    line, = ax.plot([], [], [], 'o-', linewidth=2, color='blue')

    # Drawing chassis
    collection = Poly3DCollection(vertices, alpha=0.25, facecolor="green", linewidths=1, edgecolors='k')
    ax.add_collection3d(collection)

    # End effector environmental force error
    ee_arrow = Arrow3D([contact[0], 0.25], [contact[1], 0.25], [contact[2], 0.25], arrowstyle="->",
                       lw=2, mutation_scale=15, color='red')
    ax.add_artist(ee_arrow)

    # Plot sphere
    task_rad = 0.2
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x_sphere = task_rad * np.outer(np.cos(u), np.sin(v)) + contact[0]
    y_sphere = task_rad * np.outer(np.sin(u), np.sin(v)) + contact[1]
    z_sphere = task_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + contact[2]

    ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=4, cstride=4, color='y', linewidth=0, alpha=0.25)
    ax.plot(task_rad*np.sin(u)+contact[0], task_rad*np.cos(u)+contact[1], 0, color='k', linestyle='dashed')
    ax.plot([contact[0]]*100, task_rad*np.sin(u)+contact[1], task_rad*np.cos(u)+contact[2], color='k', linestyle='dashed')

    # Setting the axes properties
    ax.set_xlim3d([-0.2, 0.8])
    ax.set_xlabel('X')

    ax.set_ylim3d([-0.5, 0.5])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')

    # coords = position()
    # line.set_data(coords[:2])
    # line.set_3d_properties(coords[2])
    #
    # plt.show()

    # Speed multiplier
    mult = 30

    # Creating the Animation object fargs=(data, lines)
    line_ani = animation.FuncAnimation(fig, animate, frames=2001, interval=1, blit=True, init_func=init, repeat=False)
    # line_ani.save('animation.mp4', fps=30)

    plt.show()
