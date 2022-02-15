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
FILEPATH_CSV = './GenerateGainsConstants.csv'

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

    link1 = robot.link(6)
    link2 = robot.link(7)
    link3 = robot.link(8)
    link4 = robot.link(9)
    link4_mass = link4.getMass()

    added_mass = 1
    attachment_point = [0.042, -0.036475, 0]
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
    gravity = (0,0,-9.81)
    print robot.getGravityForces(gravity)
    link4_mass.setMass(new_mass)
    link4_mass.setCom(new_com)
    link4_mass.setInertia(new_inertia)
    link4.setMass(link4_mass)
    print robot.getGravityForces(gravity)

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
    # default 26.43113890337936