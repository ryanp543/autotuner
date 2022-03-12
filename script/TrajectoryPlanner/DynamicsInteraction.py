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


def FindKjTranspose(var_init, robot, sus):
    state = [0, 0, var_init[0], 0]
    state.extend(var_init[1:])
    robot.setConfig(state)
    link4 = robot.link(9)
    J = np.asarray(link4.getPositionJacobian([0,0,0]))
    J = np.delete(J, [0,1,2,3], 1)
    J_trans = np.transpose(J)
    J_trans_rowsums = np.absolute(np.sum(J_trans, axis=1))

    return -np.amax(J_trans_rowsums)


def FindKj(var_init, robot, sus):
    state = [0, 0, var_init[0], 0]
    state.extend(var_init[1:])
    robot.setConfig(state)
    link4 = robot.link(9)
    J = np.asarray(link4.getPositionJacobian([0,0,0]))
    J = np.delete(J, [0,1,3], 1)
    J_rowsums = np.absolute(np.sum(J, axis=1))

    return -np.amax(J_rowsums)


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

    # CALCULATING MAX ROW SUM OF JACOBIAN TRANSPOSE (z, qroll, qpitch, 4 DOF)
    print "Calculating max row sum of Jacobian transpose..."
    stateMaxJtrans = scipy.optimize.fmin(FindKjTranspose, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    max_Jtrans_rowsum = -FindKjTranspose(stateMaxJtrans, robot, sus)
    if max_Jtrans_rowsum < 1.0:
        max_Jtrans_rowsum = 1.0
    print max_Jtrans_rowsum
    # this is noting that there is a [0 0 1] row in the jacobian corresponding to the partial of tau_z w.r.t. F_z

    # CALCULATING MAX ROW SUM OF JACOBIAN (z, qroll, qpitch, 4 DOF)
    print "Calculating max row sum of Jacobian..."
    stateMaxJ = scipy.optimize.fmin(FindKj, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500)
    max_J_rowsum = -FindKj(stateMaxJ, robot, sus)
    print max_J_rowsum

    # CALCULATING MAX EIGEN VALUE OF H MATRIX (z, qroll, qpitch, 4 DOF)
    # print "Calculating Max Eig H..."
    # CURRENT_FUNCTION = FindMaxEigH
    # stateMaxEigH = scipy.optimize.fmin(FindMaxEigH, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    # maxEigH = -FindMaxEigH(stateMaxEigH, robot, sus)
    # print maxEigH

