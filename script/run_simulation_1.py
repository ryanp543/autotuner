#!/usr/bin/env python2

"""
Python program that runs a simulation based on desired end positions only, without any planned trajectory.

Note: This script is run using rosrun robot_description run_simulation_1.py (or however you decide to organize your
ROS package) from the outermost directory of your ROS package. However, this really depends on how you organize your
package.

Date 2/2/2021
"""

from __future__ import print_function

import time
import sys
import csv
import rospy
import numpy as np
import gazebo_msgs
import matplotlib.pyplot as plt

from std_msgs.msg import Float64
from std_srvs.srv import Empty
from control_msgs.msg import JointControllerState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import GetLinkState

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

# joint 1: -3.1415 to 3.1415
# joint 2: -3.1415 to 0
# joint 3: -1.0472 to 2.6
# joint 4: -1.571 to 1.571


# Function: Run Motion
# Sends a single position command to arm
def run_motion(q_d):
    print(q_d)
    for n in range(0, len(q_d)):
        joint_cmd_pub_dic[n].publish(q_d[n])


# Function: Run Motion Sets
# Runs motion sets based on the list of desired arm orientations generated in main function
def run_motion_sets(q1_d, q2_d, q3_d):
    global t_start, t, t1, t2, t3, qC_r, qC_p, q1, q2, q3
    t_set, t1_set, t2_set, t3_set, qC_r_set, qC_p_set, q1_set, q2_set, q3_set = [], [], [], [], [], [], [], [], []
    num_simulations = len(q1_d) * len(q2_d) * len(q3_d)
    sim_count = 1

    # For each desired orientation
    for ii in range(len(q1_d)):
        for jj in range(len(q2_d)):
            for kk in range(len(q3_d)):
                print("Running Simulation", sim_count, "of", num_simulations)

                # Run the motion and collect angle data
                q_d = [q1_d[ii], q2_d[jj], q3_d[kk]]
                run_motion(q_d)
                listener()

                # Append states and times to respective set for future plotting
                t_set.append(t)
                t1_set.append(t1)
                t2_set.append(t2)
                t3_set.append(t3)
                qC_r_set.append(qC_r)
                qC_p_set.append(qC_p)
                q1_set.append(q1)
                q2_set.append(q2)
                q3_set.append(q3)
                reset_simulation()
                t, t1, t2, t3, qC_r, qC_p, q1, q2, q3 = [], [], [], [], [], [], [], [], []
                t_start = rospy.get_time()
                sim_count += 1

    return t_set, t1_set, t2_set, t3_set, qC_r_set, qC_p_set, q1_set, q2_set, q3_set


# Function: Chassis State Callback
# Appends the next recorded state of the chassis pitch and roll
def callback_chassis(link_states):
    global t, qC_r, qC_p, t_start
    ind_chassis = link_states.name.index("robot::link_chassis") # "robot::Link_1"
    ind_l1 = link_states.name.index("robot::Link_1")
    ind_l2 = link_states.name.index("robot::Link_2")
    ind_l3 = link_states.name.index("robot::Link_3")

    # x is roll, y is pitch, z is yaw
    t.append(rospy.get_time() - t_start)
    qC_r.append(link_states.pose[ind_chassis].orientation.x)
    qC_p.append(link_states.pose[ind_chassis].orientation.y)


# Function: Joint 1 State Callback
# Appends the next recorded state of joint 1
def callback_joint1(joint_state):
    global t1, q1, t_start
    t_stamp = float(joint_state.header.stamp.secs) + float(joint_state.header.stamp.nsecs) * (10 ** -9) - t_start
    t1.append(t_stamp)
    q1.append(joint_state.error)


# Function: Joint 2 State Callback
# Appends the next recorded state of joint 2
def callback_joint2(joint_state):
    global t2, q2, t_start
    t_stamp = float(joint_state.header.stamp.secs) + float(joint_state.header.stamp.nsecs) * (10 ** -9) - t_start
    t2.append(t_stamp)
    q2.append(joint_state.error)


# Function: Joint 3 State Callback
# Appends the next recorded state of joint 3
def callback_joint3(joint_state):
    global t3, q3, t_start
    t_stamp = float(joint_state.header.stamp.secs) + float(joint_state.header.stamp.nsecs) * (10 ** -9) - t_start
    t3.append(t_stamp)
    q3.append(joint_state.error)


# Function: Listeners
# Starts the listeners for data collection of the link states, adds them to list of subscribers
def listener():
    sub = []
    sub.append(rospy.Subscriber("gazebo/link_states", LinkStates, callback_chassis))
    sub.append(rospy.Subscriber("robot/joint1_position_controller/state", JointControllerState, callback_joint1))
    sub.append(rospy.Subscriber("robot/joint2_position_controller/state", JointControllerState, callback_joint2))
    sub.append(rospy.Subscriber("robot/joint3_position_controller/state", JointControllerState, callback_joint3))

    rospy.sleep(5)
    for k in range(len(sub)):
        sub[k].unregister()


# Function: Reset Simulation
# Resets the simulation space in preparation for next trial of motions
def reset_simulation():
    print("Resetting world...")
    reset_sim = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
    run_motion([0, 0, 0])
    reset_sim()
    try:
        rospy.sleep(2)
    except rospy.exceptions.ROSTimeMovedBackwardsException:
        pass
    rospy.sleep(2)


# Function: Get Legend Names
# Generates the names of each desired orientation for future use in a legend.
def get_legend_names(q1_d, q2_d, q3_d):
    labels = []
    for ii in range(len(q1_d)):
        for jj in range(len(q2_d)):
            for kk in range(len(q3_d)):
                labels.append("[" + str(q1_d[ii]) + ", " + str(q2_d[jj]) + ", " + str(q3_d[kk]) + "]")

    return labels


# Function: Calculate Settling Time
# Calculates thet settling times and average settling time for a specific linkage set.
def calculate_settling_time(t_set, q_set):
    settling_time_list = []
    settling_percent = 0.02
    for k in range(len(t_set)):
        avg_q_ss = sum(q_set[k][-100:len(q_set[k])]) / 100
        for j in range(len(q_set[k])):

            if abs((q_set[k][j]-avg_q_ss)/avg_q_ss) < settling_percent:
                avg_next = sum(q_set[k][j:j+100]) / 100

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
# Runs simulation by sending single position command containing target desired positions. All the joint/link data is
# uploaded to a .csv file to be plotted and are also plotted here.
if __name__ == "__main__":
    rospy.init_node('run_simulation')
    rospy.wait_for_service('gazebo/reset_simulation')
    rate = rospy.Rate(30)

    # Define the ROS publishers and put them in a dictionnary
    joint1_cmd_pub = rospy.Publisher("robot/joint1_position_controller/command",Float64,queue_size=100)
    joint2_cmd_pub = rospy.Publisher("robot/joint2_position_controller/command",Float64,queue_size=100)
    joint3_cmd_pub = rospy.Publisher("robot/joint3_position_controller/command",Float64,queue_size=100)
    joint_cmd_pub_dic = [joint1_cmd_pub,joint2_cmd_pub,joint3_cmd_pub]
    time.sleep(1)

    t, t1, t2, t3 = [], [], [], []
    qC_r, qC_p, q1, q2, q3 = [], [], [], [], []
    t_start = rospy.get_time()

    q1_d = [-2.0944, -1.0472, 1.0472, 2.0944]
    q2_d = [-1.0472, -2.0944, -3.1415]
    q3_d = [0.5236, 1.000, 2.094]
    labels = get_legend_names(q1_d, q2_d, q3_d)
    t_set, t1_set, t2_set, t3_set, qC_r_set, qC_p_set, q1_set, q2_set, q3_set = run_motion_sets(q1_d, q2_d, q3_d)

    # ADD DATA TO CSV FILE
    # Chassis angle data
    filepath = './src/autotuner/script/SimulationData/sim_data_chassis.csv'
    with open(filepath, 'wb') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        for k in range(0, len(t_set)):
            csvwriter.writerow(t_set[k])
            csvwriter.writerow(qC_r_set[k])
            csvwriter.writerow(qC_p_set[k])

    # Joint 1 data
    filepath = './src/autotuner/script/SimulationData/sim_data_joint1.csv'
    with open(filepath, 'wb') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        for k in range(0, len(t1_set)):
            csvwriter.writerow(t1_set[k])
            csvwriter.writerow(q1_set[k])

    # Joint 2 data
    filepath = './src/autotuner/script/SimulationData/sim_data_joint2.csv'
    with open(filepath, 'wb') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        for k in range(0, len(t2_set)):
            csvwriter.writerow(t2_set[k])
            csvwriter.writerow(q2_set[k])

    # Joint 3 data
    filepath = './src/autotuner/script/SimulationData/sim_data_joint3.csv'
    with open(filepath, 'wb') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        for k in range(0, len(t3_set)):
            csvwriter.writerow(t3_set[k])
            csvwriter.writerow(q3_set[k])

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
    print("Generating plots...")
    plt.figure(1, figsize=(10,5), dpi=80)
    plt.rc('font', size=20)
    plt.subplots_adjust(left=0.17, bottom=0.15)
    plt.subplot(2, 1, 1)
    for k in range(len(t_set)):
        avg_qCr_ss = sum(qC_r_set[k][-100:len(qC_r_set[k])]) / 100
        plt.plot(t_set[k], [qC_r_set[k][g]-avg_qCr_ss for g in range(0, len(qC_r_set[k]))]) #, label=labels[k])
    plt.ylabel(r"$||\tilde{q}_{C,r}||$")
    plt.xlabel("Time (s)")
    plt.ylim((-0.005, 0.005))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.subplot(2, 1, 2)
    for k in range(len(t_set)):
        avg_qCp_ss = sum(qC_p_set[k][-100:len(qC_p_set[k])]) / 100
        plt.plot(t_set[k], [qC_p_set[k][g]-avg_qCp_ss for g in range(0, len(qC_p_set[k]))], label=labels[k])
    plt.ylabel(r"$||\tilde{q}_{C,p}||$")
    plt.xlabel("Time (s)")
    plt.ylim((-0.06, 0.01))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')

    plt.figure(2, figsize=(10,8), dpi=80)
    plt.rc('font', size=20)
    plt.subplot(3, 1, 1)
    for k in range(len(t1_set)):
        plt.plot(t1_set[k], q1_set[k]) #, label=labels[k])
    plt.ylabel(r"$||\tilde{q}_{L1}||$")
    plt.xlabel("Time (s)")
    plt.ylim((-2.4, 2.4))
    plt.yticks(np.arange(-2.4, 3.2, 1.2))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.subplot(3, 1, 2)
    for k in range(len(t2_set)):
        plt.plot(t2_set[k], q2_set[k]) #, label=labels[k])
    plt.ylabel(r"$||\tilde{q}_{L2}||$")
    plt.xlabel("Time (s)")
    plt.ylim((-3, 0.5))
    plt.yticks(np.arange(-3.6, 2.4, 1.2))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.subplot(3, 1, 3)
    for k in range(len(t3_set)):
        plt.plot(t3_set[k], q3_set[k]) #, label=labels[k])
    plt.ylabel(r"$||\tilde{q}_{L3}||$")
    plt.xlabel("Time (s)")
    plt.ylim((-0.5, 2.5))
    plt.yticks(np.arange(-0.5, 3.1, 1.2))
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')

    plt.show()