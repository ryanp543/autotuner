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
FILEPATH_MASS_CSV = './DynamicsMassConstants.csv'
FILEPATH_DEFAULT_CSV = './GenerateGainsConstants_default.csv'

# Function: Extract Data
# Extracts data from the .csv file generated by GenerateConstants.py
def ExtractData_GainBounds():
    with open(FILEPATH_DEFAULT_CSV, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)

    constants = [None] * 14
    for k in range(1,len(constants)+1):
        constants[k-1] = float(data[k][1])

    alpha = float(data[24][1])
    K_I = float(data[25][1])

    return constants, alpha, K_I


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


def FindMaxEigKenv(var_init, robot, sus):
    state = [0, 0, var_init[0], 0]
    state.extend(var_init[1:])
    robot.setConfig(state)
    link4 = robot.link(9)
    J = np.asarray(link4.getPositionJacobian([0,0,0]))
    J = np.delete(J, [0,1,2,3,7], 1)
    J_trans = np.transpose(J)
    Kenv_rob = np.linalg.multi_dot([J_trans, Kenv, J])
    w, v = np.linalg.eig(Kenv_rob)

    return -max(w)


def FindKkEnv(var_init, robot, sus):
    gravity = (0, 0, -9.81)
    list_dKenvdx = []
    list_max = []

    x = [0, 0, 0, 0] + list(var_init[0:len(var_init)])  # roll pitch 4dof
    dx = 0.001 # [0, 0, 0, 0] + list(var_init[len(var_init)/2:len(var_init)])
    x_original = x[:]

    # For each individual DOF derivative (with the other derivatives held at 0), dGdx is calculated
    for j in range(4, len(x)):
        x = x_original[:]

        robot.setConfig(x)
        J1 = np.asarray(link4.getPositionJacobian([0, 0, 0]))
        J1 = np.delete(J1, [0, 1, 3], 1)
        J1_trans = np.transpose(J1)
        Kenv_rob1 = np.linalg.multi_dot([J1_trans, Kenv, J1])
        print Kenv_rob1

        x[j] = x[j] + dx
        robot.setConfig(x)
        J2 = np.asarray(link4.getPositionJacobian([0, 0, 0]))
        J2 = np.delete(J2, [0, 1, 3], 1)
        J2_trans = np.transpose(J2)
        Kenv_rob2 = np.linalg.multi_dot([J2_trans, Kenv, J2])

        for i in range(4, len(G1)): # range(4, len(G1)):
            list_dGdx.append(abs((Kenv_rob1[i]-Kenv_rob2[i])/dx))

        # Find the maximum dGdx of dx[j]
        list_max.append(max(list_dGdx))
        list_dGdx = []

    # Returns the maximum dGdx overall
    return -max(list_max)


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
    print link4.getID()

    # Initializing state vectors
    # full state: (x, y, z, qyaw, qroll, qpitch, q1, q2, q3, q4)
    x_init = [0] * robot.numLinks()
    dx_init = [0.1] * robot.numLinks()
    states = x_init + dx_init

    Kenv = np.asarray([[70, 0, 0], [0, 70, 0], [0, 0, 70]])

    # CALCULATING MAX ROW SUM OF JACOBIAN TRANSPOSE (z, qroll, qpitch, 4 DOF)
    print "Calculating max row sum of Jacobian transpose..."
    stateMaxEigKenv = scipy.optimize.fmin(FindMaxEigKenv, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    maxEigKenv = -FindMaxEigKenv(stateMaxEigKenv, robot, sus)
    print maxEigKenv
    print stateMaxEigKenv

    # CALCULATING K_k,env
    # print "Calculating K_k,env..."
    # FindKkEnv(states[4:robot.numLinks()], robot, sus)

    # CALCULATING MAX K_G OF dg_i/dx_j MATRIX (qroll, qpitch, 4 DOF + velocities)
    # print "Calculating Kg..."
    # CURRENT_FUNCTION = FindKg
    # stateMaxKg = scipy.optimize.fmin(FindKg, states[4:robot.numLinks()], args=(robot,sus), maxiter=4000) # callback=fminCallback) +states[14:20]
    # kG = -FindKg(stateMaxKg, robot, sus)

    # CALCULATING MAX EIGEN VALUE OF H MATRIX (z, qroll, qpitch, 4 DOF)
    # print "Calculating Max Eig H..."
    # CURRENT_FUNCTION = FindMaxEigH
    # stateMaxEigH = scipy.optimize.fmin(FindMaxEigH, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    # maxEigH = -FindMaxEigH(stateMaxEigH, robot, sus)
    # print maxEigH

