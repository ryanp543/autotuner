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
import scipy.optimize
import matplotlib.pyplot as plt
import klampt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from klampt import vis

FILEPATH_TRAJECTORY = './DynamicsInteractionTrajectory_default.csv'


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
    x = np.asarray([0,
                   link1.getWorldPosition([0, 0, 0])[0],
                   link2.getWorldPosition([0, 0, 0])[0],
                   link3.getWorldPosition([0, 0, 0])[0],
                   link4.getWorldPosition([0, 0, 0])[0]])
    y = np.asarray([0,
                   link1.getWorldPosition([0, 0, 0])[1],
                   link2.getWorldPosition([0, 0, 0.05275])[1],
                   link3.getWorldPosition([0, 0, 0])[1],
                   link4.getWorldPosition([0, 0, 0])[1]])
    z = np.asarray([0,
                   link1.getWorldPosition([0, 0, 0])[2],
                   link2.getWorldPosition([0, 0, 0])[2],
                   link3.getWorldPosition([0, 0, 0])[2],
                   link4.getWorldPosition([0, 0, 0])[2]])

    return (x, y, z)


def step(k):
    x_config = np.concatenate(([0.0, 0.0], x[k, 0], 0, x[k, 1:], 0), axis=None)
    robot.setConfig(x_config)


def getNewTime(frame):
    return frame * dt


def animate(frame):
    # Step to the next joint positions
    step(frame)

    # Calculate the new positions
    coords = position()
    line.set_data(coords[:2])
    line.set_3d_properties(coords[2])

    # Change the timer text
    # time_text.set_text('time = %.2f' % getNewTime(frame))

    return line, # time_text


def init():
    line.set_data([], [])
    line.set_3d_properties([])
    # time_text.set_text('')

    return line, # time_text


if __name__ == "__main__":
    world = klampt.WorldModel()
    res = world.readFile("./robot_sim.xml")
    robot = world.robot(0)

    link1 = robot.link(6)
    link2 = robot.link(7)
    link3 = robot.link(8)
    link4 = robot.link(9)

    x, dt, test_length = ExtractTrajectoryData()

    x_config = np.concatenate(([0.0, 0.0], x[0,0], 0, x[0,1:], 0), axis=None)
    # x_config = np.zeros(10)
    robot.setConfig(x_config)

    # vis.add("world", world)
    # vis.run()

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line, = ax.plot([], [], [], 'o-', lw=2, mew=1)
    # time_text = ax.text2D(0.05, 0.90, '', transform=ax.transAxes)

    # Setting the axes properties
    ax.set_xlim3d([-0.25, 0.75])
    ax.set_xlabel('X')

    ax.set_ylim3d([-0.5, 0.5])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')


    # Creating the Animation object fargs=(data, lines)
    line_ani = animation.FuncAnimation(fig, animate, frames=2000, interval=1, blit=True, init_func=init)

    plt.show()
