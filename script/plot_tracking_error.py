#!/usr/bin/env python2

"""
Python program to use all the trajectory optimization position commands and simulation data to plot the tracking error
of the arm as it moves to the desired orientation according to the planned path.

Note: Make sure to change filepaths before extracting the .csv files (if they're any different). Ideally, run this
script using rosrun.

Date 2/2/2021
"""

from __future__ import print_function

import time
import sys
import csv
import rospy
import numpy as np
import matplotlib.pyplot as plt

# joint 1: -3.1415 to 3.1415
# joint 2: -3.1415 to 0
# joint 3: -1.0472 to 2.6
# joint 4: -1.571 to 1.571


# Function: Calculate RMS
# Calculates the root mean square error of the tracking error between the arm position and position commands
def calculate_rms(error_set):
    rms_set = []
    for ii in range(len(error_set)):
        error_set_abs = [abs(item)**2 for item in error_set[ii][:]]

        rms = sum(error_set_abs) / len(error_set[ii][:])
        rms_set.append(rms**0.5)

    rms_total = sum(rms_set) / len(rms_set)
    return rms_total


# Function: Main
# Extracts data from simulation and trajectory planner, compares them side by side, and then plots the RMS tracking
# error. Also provides the average RMS error.
if __name__ == "__main__":
    rospy.init_node('plot_tracking_error')

    q1_d = [-2.0944, -1.0472, 1.0472, 2.0944] 
    q2_d = [-1.0472, -2.0944, -3.1415] 
    q3_d = [0.5236, 1.000, 2.094]
    t_set, t1_set, t2_set, t3_set = [], [], [], []
    qC_r_set, qC_p_set, q1_set, q2_set, q3_set = [], [], [], [], []

    # ADD DATA TO CSV FILE
    # Joint 1 data
    filepath = './src/robot_description/script/SimulationData/BothData/sim_data_joint1.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        data = [[float(y) for y in x] for x in data]
        for row in range(0, len(data), 2):
            t1_set.append(data[row][:])
            q1_set.append(data[row+1][:])

    # Joint 2 data
    filepath = './src/robot_description/script/SimulationData/BothData/sim_data_joint2.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        data = [[float(y) for y in x] for x in data]
        for row in range(0, len(data), 2):
            t2_set.append(data[row][:])
            q2_set.append(data[row+1][:])

    # Joint 3 data
    filepath = './src/robot_description/script/SimulationData/BothData/sim_data_joint3.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        data = [[float(y) for y in x] for x in data]
        for row in range(0, len(data), 2):
            t3_set.append(data[row][:])
            q3_set.append(data[row+1][:])

    # Extract trajectory planner data
    filepath_traj = './src/robot_description/script/TrajectoryPlanner/planned_trajectory3.csv'
    with open(filepath_traj, 'r') as csvfile:
        traj = []
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            traj.append(row)

        traj = [[float(y) for y in x] for x in traj]
        [T, dt, num_steps, t_cmd, num_coeff] = traj[0][:]

    # Print time steps, etc
    print("T = " + str(T))
    print("t_cmd = " + str(t_cmd))
    print("num_steps = " + str(num_steps))
    print("num_coeff = " + str(num_coeff))
    t_traj = list(np.linspace(0, T, num_steps))
    cmd_step = int(len(t_traj)/(T/t_cmd))

    # For each trajectory path, compare to actual joint position to calculate tracking error. Note that the time x-axis
    # data are different for trajectory path and simulation. Also note that the trajectory path is a series of stepped
    # positions of time length t_cmd.
    ind_set = 0
    error1_set, error2_set, error3_set = [], [], []
    for num in range(1, len(traj), 5):
        q_d = traj[num][:]
        u_1 = traj[num + 1][:]
        u_2 = traj[num + 2][:]
        u_3 = traj[num + 3][:]
        # u_4 = traj[num + 4][:]

        error1, error2, error3 = [], [], []

        # Error 1 tracking error
        step = 10
        t_step = t_cmd
        for index in range(len(q1_set[ind_set][:])):
            if t1_set[ind_set][index] > t_step:
                step += cmd_step
                t_step += t_cmd
            if step < len(u_1):
                error1.append((q1_set[ind_set][0] - q1_set[ind_set][index]) - u_1[step])
            else:
                error1.append((q1_set[ind_set][0] - q1_set[ind_set][index]) - q_d[0])
        error1_set.append(error1)

        # Error 2 tracking error
        step = 10
        t_step = t_cmd
        for index in range(len(q2_set[ind_set][:])):
            if t2_set[ind_set][index] > t_step:
                step += cmd_step
                t_step += t_cmd
            if step < len(u_2):
                error2.append((q2_set[ind_set][0] - q2_set[ind_set][index]) - u_2[step])
            else:
                error2.append((q2_set[ind_set][0] - q2_set[ind_set][index]) - q_d[1])
        error2_set.append(error2)

        # Error 3 tracking error
        step = 10
        t_step = t_cmd
        for index in range(len(q3_set[ind_set][:])):
            if t3_set[ind_set][index] > t_step:
                step += cmd_step
                t_step += t_cmd
            if step < len(u_3):
                error3.append((q3_set[ind_set][0] - q3_set[ind_set][index]) - u_3[step])
            else:
                error3.append((q3_set[ind_set][0] - q3_set[ind_set][index]) - q_d[2])
        error3_set.append(error3)

        ind_set += 1

    # Calculate average RMS error of each joint and overall average
    print(calculate_rms(error1_set))
    print(calculate_rms(error2_set))
    print(calculate_rms(error3_set))
    total_average = (calculate_rms(error1_set) + calculate_rms(error2_set) + calculate_rms(error3_set)) /3
    print(total_average)

    # Plot the tracking error of each joint
    fig = plt.figure(1, figsize=(10,10), dpi=80)
    plt.rc('font', size=30)
    ax = fig.add_subplot(3,1,1)
    ax.tick_params(width=4, length=8)
    ax.xaxis.set_ticklabels([])
    plt.setp(ax.spines.values(), linewidth=4)
    # plt.text(-2, 3, "(b)", fontsize=30)
    plt.subplots_adjust(left=0.25, top=0.95, bottom=0.11)
    for k in range(len(error1_set)):
        plt.plot(t1_set[k][:], error1_set[k][:]) #, label=labels[k])
    plt.ylabel(r"$\tilde{q}_1$"+" (rad)")
    # plt.xlabel("Time (s)")
    plt.yticks(np.arange(-1, 1.5, 0.5))
    plt.ylim((-1.4, 1.4))
    plt.xticks(np.arange(0, 6, 1))
    plt.grid()

    ax = fig.add_subplot(3,1,2)
    ax.tick_params(width=4, length=8)
    ax.xaxis.set_ticklabels([])
    plt.setp(ax.spines.values(), linewidth=4)
    for k in range(len(error2_set)):
        plt.plot(t2_set[k][:], error2_set[k][:]) #, label=labels[k])
    plt.ylabel(r"$\tilde{q}_2$"+" (rad)")
    # plt.xlabel("Time (s)")
    plt.yticks(np.arange(-1.2, 4.8, 1.2))
    plt.ylim((-1.7, 4.1))
    plt.xticks(np.arange(0, 6, 1))
    plt.grid()

    ax = fig.add_subplot(3,1,3)
    ax.tick_params(width=4, length=8)
    plt.setp(ax.spines.values(), linewidth=4)
    for k in range(len(error3_set)):
        plt.plot(t3_set[k][:], error3_set[k][:]) #, label=labels[k])
    plt.ylabel(r"$\tilde{q}_3$" + " (rad)")
    plt.xlabel("Time (s)")
    plt.yticks(np.arange(-1.6, 1.6, 0.8))
    plt.ylim((-1.9, 1.1))
    plt.xticks(np.arange(0, 6, 1))
    plt.grid()

    fig.align_ylabels()
    plt.show()

