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

# Function: Optimization Callback
# After ever iteration, the cost value is added to a list for future plotting of cost over iteration
def fminCallback(states_current_iter):
    global cost, iteration, iter_count, CURRENT_FUNCTION, robot, sus
    iter_count = iter_count + 1
    iteration.append(iter_count)
    cost.append(CURRENT_FUNCTION(states_current_iter, robot, sus))


# Function: Plot Cost
# Plots the cost value with respect to optimizer iteration
def costPlot():
    global cost, iteration
    plt.plot(iteration, cost)
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()


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


# Function: Max Eigenvalue K
# Returns the maximum eigenvalue of the stiffness matrix. Used with minimizer to find largest possible eigenvalue of
# the stiffness matrix.
def FindMaxEigK(var_init, robot, sus):
    K = sus.GetStiffnessMatrix(qPitch=var_init[1], qRoll=var_init[0])
    w, v = np.linalg.eig(K)

    return -max(w)


# Function: Max Eigenvalue B
# Returns the maximum eigenvalue of the damping matrix. Used with minimizer to find largest possible eigenvalue of
# the damping matrix.
def FindMaxEigB(var_init, robot, sus):
    B = sus.GetDampingMatrix(qPitch=var_init[1], qRoll=var_init[0])
    w, v = np.linalg.eig(B)

    return -max(w)


# Function: Maximum Gravity Magnitude
# Calculates and returns the magnitude of the gravity vector for a given robot state. Used with the minimizer to find
# the largest possible magnitude.
def FindMaxG(var_init, robot, sus):
    gravity = (0, 0, -9.81)
    x = [0, 0, var_init[0], 0] + list(var_init[1:])
    robot.setConfig(x)
    G = robot.getGravityForces(gravity)

    return -np.linalg.norm(G)


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


# Function: Find Constant kK
# Calculates the kK constant used for determining alpha
def FindKk(var_init, robot, sus):
    list_dKdx = []
    list_max = []

    x_p = [var_init[0], var_init[1]] # only pitch and roll are considered
    dx_p = 0.001 #var_init[2:4]
    # print dx_p

    x_p_original = x_p[:]

    # For each relevant passive state derivative
    for k in range(len(x_p)):
        x_p = x_p_original[:]

        K1 = sus.GetStiffnessMatrix(qPitch=x_p[1], qRoll=x_p[0])
        x_p[k] = x_p[k] + dx_p # [k]
        K2 = sus.GetStiffnessMatrix(qPitch=x_p[1], qRoll=x_p[0])

        for i in range(len(K1)):
            for j in range(len(K1[0])):
                dK = K2[i][j]-K1[i][j]
                list_dKdx.append(abs(dK/dx_p))  # 1D array of dK/dx for x[elem]

        list_max.append(max(list_dKdx))
        list_dKdx = []

    return -(len(var_init)/2)*max(list_max)


# Function: Find Constant kB
# Calculates the kB constant used for determining alpha
def FindKb(var_init, robot, sus):
    list_dBdx = []
    list_max = []

    x_p = [var_init[0], var_init[1]]
    dx_p = 0.001 # var_init[2:4]

    x_p_original = x_p[:]

    # For each relevant passive state variable
    for elem in range(len(x_p)):
        x_p = x_p_original[:]

        B1 = sus.GetDampingMatrix(qPitch=x_p[1], qRoll=x_p[0])
        x_p[elem] = x_p[elem] + dx_p
        B2 = sus.GetDampingMatrix(qPitch=x_p[1], qRoll=x_p[0])

        for i in range(len(B1)):
            for j in range(len(B1[0])):
                dB = B2[i][j]-B1[i][j]
                list_dBdx.append(abs(dB/dx_p))  # 1D array of dK/dx for x[elem]

        list_max.append(max(list_dBdx))
        list_dBdx = []

    return -(len(var_init)/2)*max(list_max)  # times 3 because K is 3x3


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


# Function: Solve Passive States
# This function returns the square of the static state at which only the suspension stiffness and gravity are acting
# on all states. The square is returned so a minimizing optimization function can be used to solve for the passive
# states.
def SolveXp(x_p, sus, robot, var_init, gravity):
    x = [0, 0, x_p[0], 0, x_p[1], x_p[2]] + list(var_init)
    robot.setConfig(x)
    G = robot.getGravityForces(gravity)
    K = np.asarray(sus.GetStiffnessMatrix(qPitch=x_p[2], qRoll=x_p[1]))
    G_p = np.array([G[2], G[4], G[5]])
    v = G_p + np.dot(K, x_p)
    return np.dot(v, v)


# Function: Find Constant kX
# Calculates the magnitude of the passive states by solving the SolveXp() function. Used with optimizer to find the kX
# constant
def FindKx(var_init, robot, sus):
    gravity = (0, 0, -9.81)

    x_p = [0, 0, 0]
    x_p_optimized = scipy.optimize.fmin(SolveXp, x_p, args=(sus, robot, var_init, gravity), xtol=0.000001, disp=False)
    return -np.linalg.norm(x_p_optimized)


# Function: Find Minimum K Eigenvalue (Passive States)
# Finds the final passive states using the SolveXp() function and then uses this "optimized" passive state to calculate
# the eigenvalues of the stiffness matrix.
def FindMinEigK_passive(var_init, robot, sus):
    gravity = (0, 0, -9.81)

    x_p = [0, 0, 0]
    x_p_optimized = scipy.optimize.fmin(SolveXp, x_p, args=(sus, robot, var_init, gravity), xtol=0.000001, disp=False)
    K_p = sus.GetStiffnessMatrix(qPitch=x_p_optimized[2], qRoll=x_p_optimized[1])
    w, v = np.linalg.eig(K_p)

    return min(w)


# Function: Find Minimum B Eigenvalue (Passive States)
# Finds the final passive states using the SolveXp() function and then uses this "optimized" passive state to calculate
# the eigenvalues of the damping matrix.
def FindMinEigB_passive(var_init, robot, sus):
    gravity = (0, 0, -9.81)

    x_p = [0, 0, 0]
    x_p_optimized = scipy.optimize.fmin(SolveXp, x_p, args=(sus, robot, var_init, gravity), xtol=0.000001, disp=False)
    B_p = sus.GetDampingMatrix(qPitch=x_p_optimized[2], qRoll=x_p_optimized[1])
    w, v = np.linalg.eig(B_p)

    return min(w)


# Function: Calculates All Constants and Eigenvalues
# This calculates all constants and eigenvalues necessary for finding the bounds on alpha and PID gains.
def GetConstants(robot, sus):
    # For plotting cost function only
    global CURRENT_FUNCTION, iter_count, iteration, cost
    iter_count = 0
    iteration = []
    cost = []

    # Prints out data about the robot
    print("Number of DOFs: ", robot.numLinks())
    print("Number of Links: ", robot.numDrivers())

    total_mass = 0
    for index in range(robot.numLinks()):
        link = robot.link(index)
        # print link.getName()
        total_mass = total_mass + link.getMass().mass

    # Initializes alphas and constants list
    alphas = []
    constants = []

    # Initializing state vectors
    x_init = [0] * robot.numLinks()
    dx_init = [0.1] * robot.numLinks()
    states = x_init + dx_init

    # CALCULATING MAX EIGEN VALUE OF H MATRIX (z, qroll, qpitch, 4 DOF)
    print "Calculating Max Eig H..."
    CURRENT_FUNCTION = FindMaxEigH
    stateMaxEigH = scipy.optimize.fmin(FindMaxEigH, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    maxEigH = -FindMaxEigH(stateMaxEigH, robot, sus)

    # CALCULATING MIN EIGENVALUE OF H MATRIX (z, qroll, qpitch, 4 DOF)
    print "Calculating Min Eig H..."
    CURRENT_FUNCTION = FindMinEigH
    stateMinEigH = scipy.optimize.fmin(FindMinEigH, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    minEigH = FindMinEigH(stateMinEigH, robot, sus)

    # CALCULATING MAX MAGNITUDE G VECTOR (z, qroll, qpitch, 4 DOF)
    print "Calculating Max G magnitude..."
    CURRENT_FUNCTION = FindMaxG
    stateMaxG = scipy.optimize.fmin(FindMaxG, states[2:3]+states[4:robot.numLinks()], args=(robot,sus), maxiter=1500) # callback=fminCallback)
    maxG = -FindMaxG(stateMaxG, robot, sus)

    # CALCULATING MAX K_G OF dg_i/dx_j MATRIX (qroll, qpitch, 4 DOF + velocities)
    print "Calculating Kg..."
    CURRENT_FUNCTION = FindKg
    stateMaxKg = scipy.optimize.fmin(FindKg, states[4:robot.numLinks()], args=(robot,sus), maxiter=4000) # callback=fminCallback) +states[14:20]
    kG = -FindKg(stateMaxKg, robot, sus)

    # CALCULATING MAX K_K OF dK/dx MATRIX (qpitch, qroll, no z (doesn't change K matrix), dz, dp, dr)
    print "Calculating Kk..."
    CURRENT_FUNCTION = FindKk
    stateMaxKk = scipy.optimize.fmin(FindKk, states[4:6], args=(robot,sus), maxiter=1000) # callback=fminCallback) +states[14:16]
    kK = -FindKk(stateMaxKk, robot, sus)

    # CALCULATING MAX K_B OF dB/dx MATRIX (qroll, qpitch, nof z (doesn't change B matrix), dz, dp, dr)
    print "Calculating Kb..."
    CURRENT_FUNCTION = FindKb
    stateMaxKb = scipy.optimize.fmin(FindKb, states[4:6], args=(robot,sus), maxiter=1000) # callback=fminCallback) +states[14:16]
    kB = -FindKb(stateMaxKb, robot, sus)

    # CALCULATING K_X (4 DOF)
    print "Calculating Kx..."
    CURRENT_FUNCTION = FindKx
    stateMaxKx = scipy.optimize.fmin(FindKx, states[6:robot.numLinks()], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    kX = -FindKx(stateMaxKx, robot, sus)

    # CALCULATING K_C (dz, dqroll, dqpitch, 4dof vel)
    print "Calculating Kc..."
    CURRENT_FUNCTION = FindKc
    stateMaxKc = scipy.optimize.fmin(FindKc, states[(robot.numLinks()+2):(robot.numLinks()+3)]+states[(robot.numLinks()+4):(2*robot.numLinks())], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    kC = -FindKc(stateMaxKc, robot, sus)

    # GET MAXIMUM EIGENVALUES OF K AND B MATRICES
    print "Calculating Max Eig K..."
    CURRENT_FUNCTION = FindMaxEigK
    stateMaxEigK = scipy.optimize.fmin(FindMaxEigK, states[4:6], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    maxEigK = -FindMaxEigK(stateMaxEigK, robot, sus)

    print "Calculating Max Eig B..."
    CURRENT_FUNCTION = FindMaxEigB
    stateMaxEigB = scipy.optimize.fmin(FindMaxEigB, states[4:6], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    maxEigB = -FindMaxEigB(stateMaxEigB, robot, sus)

    # CALCULATING MIN EIG K MATRIX
    print "Calculating Min Eig K (passive)..."
    CURRENT_FUNCTION = FindMinEigK_passive
    stateMinEigK_p = scipy.optimize.fmin(FindMinEigK_passive, states[6:robot.numLinks()], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    minEigK_p = FindMinEigK_passive(stateMinEigK_p, robot, sus)

    # CALCULATING MIN EIG B MATRIX
    print "Calculating Min Eig B (passive)..."
    CURRENT_FUNCTION = FindMinEigB_passive
    stateMinEigB_p = scipy.optimize.fmin(FindMinEigB_passive, states[6:robot.numLinks()], args=(robot,sus), maxiter=1000) # callback=fminCallback)
    minEigB_p = FindMinEigB_passive(stateMinEigB_p, robot, sus)

    # CALCULATING K
    k = kG + (kK * kX)

    # CREATE LIST OF CONSTANTS
    print "\n==== Constants ===="
    print "Max Eig H =", maxEigH
    print "Min Eig H =", minEigH
    print "Max g mag =", maxG
    print "Kg =", kG
    print "Kk =", kK
    print "Kb =", kB
    print "Kx =", kX
    print "Kc =", kC
    print "k =", k
    print "Max Eig K =", maxEigK
    print "Max Eig B =", maxEigB
    print "Min Eig K (passive) =", minEigK_p
    print "Min Eig B (passive) =", minEigB_p
    print "Total Mass =", total_mass
    constants.extend([maxEigH, minEigH, maxG, kG, kK, kB, kX, maxEigK, maxEigB, minEigK_p, minEigB_p, kC, k, total_mass])

    # CALCULATING ALPHA AND PID GAINS
    alpha1 = ((minEigK_p - k) / maxEigH) ** 0.5
    print "\n==== Alpha1 ====\n", alpha1
    alphas.append(alpha1)

    alpha2 = symbols('alpha2')
    eq = (alpha2 * maxEigH) + (((kK * kX) ** 2) / (4 * maxEigH * alpha2 ** 3)) - minEigB_p
    sol = solve(eq)
    print "\n==== Alpha2 ===="
    for k in range(len(sol)):
        print(sol[k])
        alphas.append(sol[k])

    # CREATE LISTS AND RETURN
    return constants, alphas


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

    # Initialize variable names for constants and alpha inputs into the .csv file
    names = ["maxEigH", "minEigH", "maxG", "kG", "kK", "kB", "kX", "maxEigK", "maxEigB", "minEigK_p", "minEigB_p",
             "kC", "k", "total_mass"]
    alpha_names = ["alpha1", "alpha2_r1", "alpha2_r2", "alpha2_r3", "alpha2_r4"]  # i.e. alpha2_r1 = "alpha 2, root 1"
    adj_names = ["Selected Alpha", "I Gain"]
    remaining_names = ["", "PD Lower Bounds", "P_lb", "D_lb", "", "Selected PID Gains", "P", "I", "D"]

    # Get constants and alphas
    constants, alphas = GetConstants(robot, sus)
    K_I_init = 20

    # Generates .csv file
    with open(FILEPATH_CSV, 'w') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        csvwriter.writerow(["Constants"])
        for i in range(len(constants)):
            csvwriter.writerow([names[i], constants[i]])

        csvwriter.writerow([""])
        csvwriter.writerow(["Alphas"])
        for k in range(len(alphas)):
            csvwriter.writerow([alpha_names[k], alphas[k]])

        csvwriter.writerow([""])
        csvwriter.writerow(["Adjustable Parameters"])
        csvwriter.writerow([adj_names[0], alphas[0]])
        csvwriter.writerow([adj_names[1], K_I_init])

        for k in range(len(remaining_names)):
            csvwriter.writerow([remaining_names[k]])

    print "\nCSV file generated, script complete"
