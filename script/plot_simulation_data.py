#!/usr/bin/env python2

"""
Python program to plot all the data from the .csv file generated from the simulation containing the joint states.
Primarily used for quickly editting plot formats without having to run the simulation over and over again.

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


# Function: Calculate Settling Time
# Calculates thet settling times and average settling time for a specific linkage set.
def calculate_settling_time(t_set, q_set):
    settling_time_list = []
    settling_percent = 0.05
    for k in range(len(t_set)):
        avg_q_ss = sum(q_set[k][-100:len(q_set[k])]) / 100
        for j in range(len(q_set[k])):

            if abs((q_set[k][j]-avg_q_ss)/avg_q_ss) < settling_percent:
                avg_next = sum(q_set[k][j:j+10]) / 10

                if abs(avg_next-avg_q_ss)/avg_q_ss < settling_percent:
                    settling_time_list.append(t_set[k][j])
                    break;

    settling_time_avg = sum(settling_time_list) / len(settling_time_list)

    return settling_time_avg


# Function: Calculate Overshoot
# Calculates the overshoot and average overshoot for a specific linkage set
def calculate_overshoot(t_set, q_set):
    overshoot_list = []
    for k in range(len(t_set)):
        sign = int(q_set[k][0] / abs(q_set[k][0]))
        FINDING_MAX = False
        start_index = 0
        end_index = 0

        for j in range(len(q_set[k])):
            sign_check = int(q_set[k][j] / abs(q_set[k][j]))
            if (FINDING_MAX is False) and ((sign > 0 and sign_check < 0) or (sign < 0 and sign_check > 0)):
                start_index = j
                FINDING_MAX = True
            if (FINDING_MAX is True) and ((sign > 0 and sign_check > 0) or (sign < 0 and sign_check < 0)):
                end_index = j
                break

        # print(t_set[k][start_index])
        # print(t_set[k][end_index])

        if end_index == 0:
            overshoot = 0
        else:
            if sign > 0:
                overshoot = min(q_set[k][start_index:end_index]) / -q_set[k][0]
            else:
                overshoot = max(q_set[k][start_index:end_index]) / -q_set[k][0]
        overshoot_list.append(overshoot)

    overshoot_avg = sum(overshoot_list) / len(overshoot_list)
    return overshoot_avg


# Function: Main
# Extracts all the simulation data from the .csv files, calculates settling time and overshoot, and plots them.
if __name__ == "__main__":
    rospy.init_node('plot_simulation_results')

    q1_d = [-2.0944, -1.0472, 1.0472, 2.0944] 
    q2_d = [-1.0472, -2.0944, -3.1415]
    q3_d = [0.5236, 1.000, 2.094]
    t_set, t1_set, t2_set, t3_set = [], [], [], []
    qC_r_set, qC_p_set, q1_set, q2_set, q3_set = [], [], [], [], []

    # ADD DATA TO CSV FILE
    # Chassis angle data
    filepath = './src/autotuner/script/SimulationData/BothData/sim_data_chassis.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        data = [[float(y) for y in x] for x in data]
        for row in range(0, len(data), 3):
            t_set.append(data[row][:])
            qC_r_set.append(data[row+1][:])
            qC_p_set.append(data[row+2][:])

    # Joint 1 data
    filepath = './src/autotuner/script/SimulationData/BothData/sim_data_joint1.csv'
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
    filepath = './src/autotuner/script/SimulationData/BothData/sim_data_joint2.csv'
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
    filepath = './src/autotuner/script/SimulationData/BothData/sim_data_joint3.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        data = [[float(y) for y in x] for x in data]
        for row in range(0, len(data), 2):
            t3_set.append(data[row][:])
            q3_set.append(data[row+1][:])

    # CALCULATING SETTLING TIME
    averages = []
    averages.append(calculate_settling_time(t1_set, q1_set))
    averages.append(calculate_settling_time(t2_set, q2_set))
    averages.append(calculate_settling_time(t3_set, q3_set))
    for n in range(len(averages)):
        print("Joint " + str(n) + " settling_time: " + str(averages[n]))

    # CALCULATING OVERSHOOT
    overshoots = []
    overshoots.append(calculate_overshoot(t1_set, q1_set))
    overshoots.append(calculate_overshoot(t2_set, q2_set))
    overshoots.append(calculate_overshoot(t3_set, q3_set))
    for n in range(len(overshoots)):
        print("Joint " + str(n) + " overshoot: " + str(overshoots[n]))


    # PLOTTING EVERYTHING
    fmt = lambda x: "{:.1f}".format(x)

    print("Generating plots...")
    fig = plt.figure(1, figsize=(10,22), dpi=80)
    plt.rc('font', size=30)
    ax = fig.add_subplot(5,1,1)
    ax.tick_params(width=4, length=8)
    ax.xaxis.set_ticklabels([])
    plt.setp(ax.spines.values(), linewidth=4)
    # plt.text(-2, 3, "(b)", fontsize=30)
    plt.subplots_adjust(left=0.23, top=0.95, bottom=0.07)
    for k in range(len(t1_set)):
        plt.plot(t1_set[k], q1_set[k]) #, label=labels[k])
    plt.ylabel(r"$q_1 - {q_1}^{ref}$"+"\n(rad/s)")
    # plt.xlabel("Time (s)")
    plt.yticks(np.arange(-3.0, 3.0, 1.0), [fmt(i) for i in np.arange(-3.0, 3.0, 1.0)])
    plt.ylim((-2.4, 2.4))
    plt.xticks(np.arange(0, 6, 1))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')

    ax = fig.add_subplot(5,1,2)
    ax.tick_params(width=4, length=8)
    ax.xaxis.set_ticklabels([])
    plt.setp(ax.spines.values(), linewidth=4)
    for k in range(len(t2_set)):
        plt.plot(t2_set[k], q2_set[k]) #, label=labels[k])
    plt.ylabel(r"$q_2 - {q_2}^{ref}$"+"\n(rad/s)")
    # plt.xlabel("Time (s)")
    plt.yticks(np.arange(-3, 3, 1), [fmt(i) for i in np.arange(-3.0, 3.0, 1.0)])
    plt.ylim((-3.5, 1.5))
    plt.xticks(np.arange(0, 6, 1))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')

    ax = fig.add_subplot(5,1,3)
    ax.tick_params(width=4, length=8)
    ax.xaxis.set_ticklabels([])
    plt.setp(ax.spines.values(), linewidth=4)
    for k in range(len(t3_set)):
        plt.plot(t3_set[k], q3_set[k]) #, label=labels[k])
    plt.ylabel(r"$q_3 - {q_3}^{ref}$"+"\n(rad/s)")
    # plt.xlabel("Time (s)")
    plt.yticks(np.arange(-0.7, 2.8, 0.7))
    plt.ylim((-1, 2.3))
    plt.xticks(np.arange(0, 6, 1))
    # ax.autoscale()
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')

    ax = fig.add_subplot(5,1,4)
    ax.tick_params(width=4, length=8)
    ax.xaxis.set_ticklabels([])
    plt.setp(ax.spines.values(), linewidth=4)
    # plt.text(-2, 0.006, "(d)", fontsize=30)
    for k in range(len(t_set)):
        avg_qCr_ss = sum(qC_r_set[k][-100:len(qC_r_set[k])]) / 100
        plt.plot(t_set[k], [1000*(qC_r_set[k][g]-avg_qCr_ss) for g in range(0, len(qC_r_set[k]))]) #, label=labels[k])
    plt.ylabel(r"$\phi - {\phi}^{ref}$"+ "\n" + r"($10^{-3}$ rad/s)")
    # plt.xlabel("Time (s)")
    plt.yticks(np.arange(-5, 7.5, 2.5))
    plt.ylim((-6, 6))
    plt.xticks(np.arange(0, 6, 1))
    # ax.autoscale()
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')

    ax = fig.add_subplot(5,1,5)
    ax.tick_params(width=4, length=8)
    plt.setp(ax.spines.values(), linewidth=4)
    for k in range(len(t_set)):
        avg_qCp_ss = sum(qC_p_set[k][-100:len(qC_p_set[k])]) / 100
        plt.plot(t_set[k], [100*(qC_p_set[k][g]-avg_qCp_ss) for g in range(0, len(qC_p_set[k]))]) #, label=labels[k])
    plt.ylabel(r"$\theta - {\theta}^{ref}$"+ "\n" + r"($10^{-2}$ rad/s)")
    plt.xlabel("Time (s)")
    plt.yticks(np.arange(-5, 7.5, 2.5))
    plt.ylim((-6, 6))
    plt.xticks(np.arange(0, 6, 1))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5) , fontsize='x-small')

    fig.align_ylabels()
    plt.show()