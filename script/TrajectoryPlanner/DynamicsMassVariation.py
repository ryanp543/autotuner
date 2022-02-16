#!/usr/bin/env python

"""
Python program that calculates the necessary constants needed to find bounds on alpha and PID gains. Running this file
generates a .csv file containing all these constants as well as the bounds on alpha.

Date 2/2/2021
"""


import sys
import csv
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from sympy import symbols, Eq, solve

import klampt
from SuspensionMatrices import Suspension_8legs

# NOTE: Edit for specific robot
FILEPATH = './robot_sim.xml'
FILEPATH_CSV = './DynamicsMassConstants.csv'

# Function: Max Eigenvalue H
# Returns the maximum eigenvalue of the H matrix for a given state. Used with minimizer to find maximum possible
# eigenvalue of the H matrix
def FindMaxEigH(var_init, robot, sus):
    state = [0, 0, var_init[0], 0]
    state.extend(var_init[1:])
    robot.setConfig(state)
    H_array = np.asarray(robot.getMassMatrix())
    H = np.delete(H_array, [0,1,3], 0)
    H = np.delete(H, [0,1,3], 1)
    w, v = np.linalg.eig(H)
    return -max(w)


# Function: Min Eigenvalue H
# Returns the minimum eigenvalue of the H matrix for a given state. Used with minimizer to find smallest possible
# eigenvalue of the H matrix
def FindMinEigH(var_init, robot, sus):
    state = [0, 0, var_init[0], 0]
    state.extend(var_init[1:])
    robot.setConfig(state)
    H_array = np.asarray(robot.getMassMatrix())
    H = np.delete(H_array, [0,1,3], 0)
    H = np.delete(H, [0,1,3], 1)
    w, v = np.linalg.eig(H)
    return min(w)


# Function: Find Constant kG
# Calculates the constant kG for determining alpha.
def FindKg(var_init, robot, sus):
    gravity = (0, 0, -9.81)
    list_dGdx = []
    list_max = []

    x = [0, 0, 0, 0] + list(var_init[0:len(var_init)])  # roll pitch 4dof
    dx = 0.001 # [0, 0, 0, 0] + list(var_init[len(var_init)/2:len(var_init)])
    x_original = x[:]

    # For each individual DOF derivative (with the other derivatives held at 0), dGdx is calculated
    for j in range(4, len(x)):
        x = x_original[:]

        robot.setConfig(x)
        G1 = robot.getGravityForces(gravity)
        x[j] = x[j] + dx
        robot.setConfig(x)
        G2 = robot.getGravityForces(gravity)

        for i in range(4, len(G1)): # range(4, len(G1)):
            list_dGdx.append(abs((G2[i]-G1[i])/dx))

        # Find the maximum dGdx of dx[j]
        list_max.append(max(list_dGdx))
        list_dGdx = []

    # Returns the maximum dGdx overall
    return -max(list_max)


# Function: Maximum Gravity Magnitude
# Calculates and returns the magnitude of the gravity vector for a given robot state. Used with the minimizer to find
# the largest possible magnitude.
def FindMaxG(var_init, robot, sus):
    gravity = (0, 0, -9.81)
    x = [0, 0, var_init[0], 0] + list(var_init[1:])
    robot.setConfig(x)
    G = robot.getGravityForces(gravity)

    return -np.linalg.norm(G)


# Function: Find Constant kC
# Calculates the kC constant used for plotting stability regions
def FindKc(var_init, robot, sus):
    # This sub-function calculates the coriolis constant kC using scipy's fmin() function for a specific state vector
    # and provided velocity
    def GetMaxKc(x0, dx, robot):
        x = [0, 0, x0[0], 0] + list(x0[1:])
        robot.setConfig(x)
        C_v = np.asarray(robot.getCoriolisForces())
        kC = np.linalg.norm(C_v) / (np.linalg.norm(dx) ** 2)
        return -kC

    dx = [0, 0, var_init[0], 0] + list(var_init[1:])
    robot.setVelocity(dx)

    x0 = [0] * (robot.numLinks()-3) # z, p, r, 4 dof (the -3 is ignoring x y and yaw)
    kC_states = scipy.optimize.fmin(GetMaxKc, x0, args=(dx, robot), disp=False)

    return GetMaxKc(kC_states, dx, robot)


def GetAdjustedConstants(robot, sus, added_mass, attachment_point):
    # This function calculates the adjusted constants that depend on the end effector mass. Because the EE mass changes,
    # these constants change, too.
    Ixx_added = added_mass*attachment_point[1]**2 # Ixx = my^2
    Iyy_added = added_mass*attachment_point[0]**2 # Iyy = mx^2
    Izz_added = added_mass*(attachment_point[0]**2 + attachment_point[1]**2) # Izz = m(x^2 + y^2)
    Ixy_added = -added_mass*attachment_point[0]*attachment_point[1] # Ixy = -mxy
    added_inertia = [Ixx_added, Ixy_added, 0, Ixy_added, Iyy_added, 0, 0, 0, Izz_added]

    # calculate mass
    current_mass = link4_mass.getMass()
    new_mass = current_mass + added_mass
    #print new_mass

    # calculate com
    current_com = link4_mass.getCom()
    new_com = []
    for k in range(0, len(current_com)):
        new_com.extend([(added_mass*attachment_point[k] + current_mass*current_com[k]) / (added_mass + current_mass)])
    #print new_com

    # calculate inertia
    current_inertia = link4_mass.getInertia()
    new_inertia = []
    for k in range(0, len(current_inertia)):
        new_inertia.extend([current_inertia[k] + added_inertia[k]])
    #print new_inertia

    # Set new mass properties
    link4_mass.setMass(new_mass)
    link4_mass.setCom(new_com)
    link4_mass.setInertia(new_inertia)
    link4.setMass(link4_mass)

    # Initializing state vectors
    # full state: (x, y, z, qyaw, qroll, qpitch, q1, q2, q3, q4)
    x_init = [0] * robot.numLinks()
    dx_init = [0.1] * robot.numLinks()
    states = x_init + dx_init

    # CALCULATING MAX EIGEN VALUE OF H MATRIX (z, qroll, qpitch, 4 DOF)
    print "Calculating Max Eig H..."
    CURRENT_FUNCTION = FindMaxEigH
    stateMaxEigH = scipy.optimize.fmin(FindMaxEigH, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    maxEigH = -FindMaxEigH(stateMaxEigH, robot, sus)
    print maxEigH

    # CALCULATING MIN EIGENVALUE OF H MATRIX (z, qroll, qpitch, 4 DOF)
    print "Calculating Min Eig H..."
    CURRENT_FUNCTION = FindMinEigH
    stateMinEigH = scipy.optimize.fmin(FindMinEigH, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    minEigH = FindMinEigH(stateMinEigH, robot, sus)
    print minEigH

    # CALCULATING MAX K_G OF dg_i/dx_j MATRIX (qroll, qpitch, 4 DOF + velocities)
    print "Calculating Kg..."
    CURRENT_FUNCTION = FindKg
    stateMaxKg = scipy.optimize.fmin(FindKg, states[4:robot.numLinks()], args=(robot,sus), maxiter=4000) # callback=fminCallback) +states[14:20]
    kG = -FindKg(stateMaxKg, robot, sus)
    print kG

    # CALCULATING MAX MAGNITUDE G VECTOR (z, qroll, qpitch, 4 DOF)
    print "Calculating Max G magnitude..."
    CURRENT_FUNCTION = FindMaxG
    stateMaxG = scipy.optimize.fmin(FindMaxG, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    maxG = -FindMaxG(stateMaxG, robot, sus)
    print maxG

    # CALCULATING K_C (dz, dqroll, dqpitch, 4dof vel)
    print "Calculating Kc..."
    CURRENT_FUNCTION = FindKc
    stateMaxKc = scipy.optimize.fmin(FindKc, states[(robot.numLinks()+2):(robot.numLinks()+3)]+states[(robot.numLinks()+4):(2*robot.numLinks())], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    kC = -FindKc(stateMaxKc, robot, sus)
    print kC

    adjusted_constants = [maxEigH, minEigH, kG, maxG, kC]
    return adjusted_constants


# Function: Main
# Running this main function will generate the constants and alpha bounds and then output a .csv file containing these
# values (for future plotting, etc).
if __name__ == "__main__":
    # Upload URDF file, klampt models.
    world = klampt.WorldModel()
    res = world.readFile(FILEPATH)

    robot = world.robot(0)
    sus = Suspension_8legs()
    print "Number of DOFs: " + str(robot.numLinks())
    print "Number of Links: " + str(robot.numDrivers())

    link4 = robot.link(9)
    link4_mass = link4.getMass()

    # CREATE ADDED MASS LIST AND ATTACHMENT POINT CONSTANTS
    added_mass_list = [x * 0.1 for x in range(0, 2)]
    attachment_point = [0.042, -0.036475, 0]

    # Open .csv file
    with open(FILEPATH_CSV, 'w') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        csvwriter.writerow(["mass", "maxEigH", "minEigH", "kG", "maxG", "kC"])

        # Calculate masses/adjusted constants and put them in .csv
        for added_mass in added_mass_list:
            adjusted_constants = GetAdjustedConstants(robot, sus, added_mass, attachment_point)
            constants = [added_mass] + adjusted_constants
            print adjusted_constants
            print added_mass
            csvwriter.writerow(constants)

    print "\nCSV file generated, script complete"