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
import matplotlib as mpl
import klampt


# Function: Get Passive State Target
# Returns the target static passive state, calculated by using the stiffness matrix and gravity vectors
# calculated using Klampt
def getXpTarget(x_p, x_a_target, dx_a_target):
    x = [0, 0, x_p[0], 0, x_p[1], x_p[2]] + list(x_a_target) + [0]
    robot.setConfig(x)
    G = robot.getGravityForces([0, 0, -9.81])
    K = np.asarray(sus.GetStiffnessMatrix(z=x_p[0], qPitch=x_p[1], qRoll=x_p[2]))
    G_p = np.array([G[2], G[4], G[5]])
    v = G_p + np.dot(K, x_p)
    return np.dot(v, v)


# For the minimization function to find the joint angles associated with desired end effector position
def getJointStart(xa, robot, sus, ee_pos_des):
    ee_pos, _ = getEndEffectorPos(xa)
    error = ee_pos - ee_pos_des

    return np.dot(error, error)


# Calculates end effector position and passive joint positions from provided active joint positions
def getEndEffectorPos(xa):
    xp_init = [0, 0, 0]
    xp = scipy.optimize.fmin(getXpTarget, xp_init, args=(xa, [0,0,0]), xtol=0.000001, disp=False)
    x = np.concatenate((xp, xa))
    x_config = np.concatenate(([0.0, 0.0], x[0], 0, x[1:], 0), axis=None)
    robot.setConfig(x_config)
    ee_pos = np.asarray(ee_link.getWorldPosition([0, 0, 0]))

    return ee_pos, xp


# Uses a minimization function to get the joint positions associated with desired end effector position
def getJointsFromEE(ee_pos_des):
    # Initializing lower and upper bounds of robot DOFs
    lowerbounds_a = [-np.pi, -np.pi, -1.0472]
    upperbounds_a = [np.pi, 0, 2.6]
    bnds_a = scipy.optimize.Bounds(lowerbounds_a, upperbounds_a)

    res = scipy.optimize.minimize(getJointStart, np.asarray([0, 0, 0]), args=(robot, sus, ee_pos_des), bounds=bnds_a)
    xa_init = res.x
    ee_pos_init, xp_init = getEndEffectorPos(xa_init)

    x_init = np.concatenate((xp_init, xa_init))
    print("Calculated x_init: " + str(x_init))
    print("EE position from calculated x_init: " + str(ee_pos_init))

    return x_init


# Function: Forward Simulation Step
# Calculates the passive and active states and derivatives of the next time step
# (x_init, dx_init, a_hat_init, da_hat_init, x_d, dx_d)
def forwardSimStep(x, dx, a_hat, da_hat, x_d, dx_d, dt, KP, KI, KD):
    # State is [x y z yaw pitch roll 4DOF]
    x_config = np.concatenate(([0.0, 0.0], x[0], 0, x[1:], 0), axis=None)
    dx_config = np.concatenate(([0.0, 0.0], dx[0], 0, dx[1:], 0), axis=None)

    robot.setConfig(x_config)
    robot.setVelocity(dx_config)

    # Calculate error vectors
    x_err = x - x_d

    # Collect H, G, C matrices and vectors. Resize for calculations
    zero_mat = np.zeros((3,3))

    H_mat = np.asarray(robot.getMassMatrix())
    H_mat = np.delete(H_mat, [0, 1, 3, 9], 0)
    H_mat = np.delete(H_mat, [0, 1, 3, 9], 1)
    g_vec = np.asarray(robot.getGravityForces((0, 0, -9.81)))
    g_vec = np.delete(g_vec, [0, 1, 3, 9])
    c_vec = np.asarray(robot.getCoriolisForces())
    c_vec = np.delete(c_vec, [0, 1, 3, 9])

    # Create K matrix
    Ks_mat = np.asarray(sus.GetStiffnessMatrix(z=x[0], qPitch=x[1], qRoll=x[2]))
    K_top = np.concatenate((Ks_mat, zero_mat), axis=1)
    # K_bottom = np.concatenate((zero_mat, np.add(KP, -(1/alpha)*KI)), axis=1)
    K_bottom = np.concatenate((zero_mat, KP), axis=1)
    K = np.concatenate((K_top, K_bottom), axis=0)

    # Create B matrix
    Kb_mat = np.asarray(sus.GetDampingMatrix(z=x[0], qPitch=x[1], qRoll=x[2]))
    B_top = np.concatenate((Kb_mat, zero_mat), axis=1)
    B_bottom = np.concatenate((zero_mat, KD), axis=1)
    B = np.concatenate((B_top, B_bottom), axis=0)

    # Create I vector
    I_top = -np.matmul(Ks_mat, x_d[:3])
    I_bottom = -np.matmul(KI, a_hat)
    I = np.concatenate((I_top, I_bottom), axis=0)

    # Right hand side of dynamics equation (without inertial matrix inverse)
    rhs = -c_vec - g_vec - np.matmul(K, x_err) - np.matmul(B, dx) + I
    (ddx, _, _, _) = np.linalg.lstsq(H_mat, rhs, rcond=None)

    # Kinematics calculations
    dx_next = dx + ddx*dt
    x_next = x + dx*dt + 0.5*ddx*(dt**2)

    # Calculate a_hat_dot and a_hat
    da_hat_next = -x_err[3:] - (1/alpha)*dx[3:]
    a_hat_next = a_hat + x_err[3:]*dt

    return x_next, dx_next, a_hat_next, da_hat_next


# Function: Forward Simulation
# Returns full list of passive and active states, derivatives, and errors for plotting given control input.
def simulate(x_init, dx_init, a_hat_init, da_hat_init, x_d, dx_d, dt, KP, KI, KD):
    # Initializing passive and active states and derivatives
    x = np.asarray([x_init])
    dx = np.asarray([dx_init])
    a_hat = np.asarray([a_hat_init])
    da_hat = np.asarray([da_hat_init])

    print("Starting simulation...")

    for k in range(np.size(t)-1):
        # Calculate new states
        x_new, dx_new, a_new, da_new = forwardSimStep(x[k], dx[k], a_hat[k], da_hat[k], x_d[k], dx_d[k], dt, KP, KI, KD)

        # Add to arrays
        x = np.append(x, [x_new], axis=0)
        dx = np.append(dx, [dx_new], axis=0)
        a_hat = np.append(a_hat, [a_new], axis=0)
        da_hat = np.append(da_hat, [da_new], axis=0)

    print("Simulation complete.")

    return x, dx, a_hat, da_hat


def calculateError(x, dx, a_hat, da_hat, x_d, dx_d, KP, KI, KD):
    print("Calculating error...")

    # Calculate error in position
    x_err = np.subtract(x, x_d)

    # Calculate error in a
    x_target = np.concatenate(([0.0, 0.0], x_d[-1,0], 0, x_d[-1,1:], 0), axis=None)
    robot.setConfig(x_target)
    g_v = np.asarray(robot.getGravityForces((0, 0, -9.81)))
    KI_inv = np.linalg.inv(KI)
    a_target = np.matmul(KI_inv, g_v[6:9])
    a = np.tile(a_target, (np.shape(a_hat)[0], 1))
    a_err = np.subtract(a_hat, a)

    return x_err, a_err


def calculateEndEffectorPosition(x):
    print("Calculating end effector position...")

    ee_pos_all = np.asarray([[0, 0 ,0]])
    for k in range(np.shape(x)[0]):0
        ee_pos, _ = getEndEffectorPos(x[k, 3:])

        ee_pos_all = np.append(ee_pos_all, [ee_pos], axis=0)

    print(np.shape(ee_pos_all))


def calculateEnergy(x, dx, a_hat, da_hat, x_d, dx_d, x_err, a_err, KP, KI, KD, alpha):
    s = np.add(dx, alpha*x_err)
    print(np.shape(s))


def plotResults(x, dx, a_hat, da_hat, x_d, dx_d, x_err, a_err, t):
    # Plot errors of all motors and passive joints
    fig = plt.figure(1, figsize=(10,22), dpi=80)

    ax = fig.add_subplot(6, 1, 1)
    plt.plot(t, x_err[:, 0])
    plt.ylabel(r"$z - {z}^{ref}$" + "\n(rad)") # + "\n"  + r"($10^{-2}$ rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 2)
    plt.plot(t, x_err[:, 1])
    plt.ylabel(r"$\theta - {\theta}^{ref}$" + "\n(rad)") # + "\n" + r"($10^{-2}$ rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 3)
    plt.plot(t, x_err[:, 2])
    plt.ylabel(r"$\phi - {\phi}^{ref}$" + "\n(rad)") # + "\n" + r"($10^{-3}$ rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 4)
    plt.plot(t, x_err[:, 3])
    plt.ylabel(r"$q_1 - {q_1}^{ref}$" + "\n(rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 5)
    plt.plot(t, x_err[:, 4])
    plt.ylabel(r"$q_2 - {q_2}^{ref}$" + "\n(rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 6)
    plt.plot(t, x_err[:, 5])
    plt.ylabel(r"$q_3 - {q_3}^{ref}$" + "\n(rad)")
    plt.grid()
    plt.xlabel("Time (s)")

    fig = plt.figure(2, figsize=(10,22), dpi=80)

    ax = fig.add_subplot(6, 1, 1)
    plt.plot(t, x[:, 0], color='green')
    plt.ylabel("z (rad)") # + "\n"  + r"($10^{-2}$ rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 2)
    plt.plot(t, x[:, 1], color='green')
    plt.ylabel(r"$\theta$ (rad)") # + "\n" + r"($10^{-2}$ rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 3)
    plt.plot(t, x[:, 2], color='green')
    plt.ylabel(r"$\phi$ (rad)") # + "\n" + r"($10^{-3}$ rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 4)
    plt.plot(t, x[:, 3], color='green')
    plt.ylabel(r"$q_1$ (rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 5)
    plt.plot(t, x[:, 4], color='green')
    plt.ylabel(r"$q_2$ (rad)")
    plt.grid()

    ax = fig.add_subplot(6, 1, 6)
    plt.plot(t, x[:, 5], color='green')
    plt.ylabel(r"$q_3$ (rad)")
    plt.grid()
    plt.xlabel("Time (s)")


# Function: Main
# Uses a least-squares optimizer to generate a trajectory path to minimize base vibrations, joint acceleration,
# settling time, and overshoot.
if __name__ == "__main__":
    # Intializations suspension and robot
    # Note that these must be global variables in order for the library functions to work
    from SuspensionMatrices import Suspension_8legs
    sus = Suspension_8legs()  # 3DOF model of the suspension, use with GetStiffnessMatrix and GetDampingMatrix
    world = klampt.WorldModel()
    res = world.readFile("./robot_sim.xml")
    robot = world.robot(0)

    # Getting the end effector link of the robot assembly
    ee_link = robot.link(robot.numLinks() - 1)

    # Comment out to calculate start position from contact_pt
    # contact_pt = np.asarray([0.542, -0.10475, 0])
    # x_init = getJointsFromEE(contact_pt)

    # Initial state. Joint coords: [z pitch roll 3DOF]
    x_init = np.asarray([-0.04703266, 0.0114661, 0.10795938, 0.01809096, -2.7547281, -0.31116971])
    dx_init = np.zeros((6))
    a_hat_init = np.zeros((3))
    da_hat_init = np.zeros((3))

    # PID gain matrices
    alpha = 0.5
    P, I, D = 59.0, 20.0, 13.5
    KP = np.asarray([[P, 0, 0], [0, P, 0], [0, 0, P]])
    KI = np.asarray([[I, 0, 0], [0, I, 0], [0, 0, I]])
    KD = np.asarray([[D, 0, 0], [0, D, 0], [0, 0, D]])

    # Create time parameters and axis
    dt = 0.001
    test_length = 3 # in seconds
    t = np.asarray([n * dt for n in range(0, int(test_length/dt)+1)])

    # Creating x_d and dx_d over time
    dx_d = np.zeros((np.size(t), 6))
    # xa_d = [1.0, -2.0, 0]
    # xp_d = scipy.optimize.fmin(getXpTarget, x_init[:3], args=(xa_d, dx_d[-1,:3]), xtol=0.000001, disp=False)
    # x_d_row = np.asarray([np.concatenate((xp_d, xa_d), axis=0)])

    # final_ee_pos_des = np.asarray([0.542, -0.10475, -0.1])
    # x_d_row = getJointsFromEE(final_ee_pos_des)

    x_d_row = np.asarray([-0.04703289,  0.01151737,  0.10796366, -0.01255453, -3.04393167, -0.08769358])
    x_d = np.tile(x_d_row, (np.size(t), 1))

    # Simulate
    x, dx, a, da = simulate(x_init, dx_init, a_hat_init, da_hat_init, x_d, dx_d, dt, KP, KI, KD)

    # Calculate errors
    x_err, a_err = calculateError(x, dx, a, da, x_d, dx_d, KP, KI, KD)

    # Calculate end effector position x, y, z
    calculateEndEffectorPosition(x)

    quit()

    # Calculate energy and energy derivative, V and V_dot
    # calculateEnergy(x, dx, a, da, x_d, dx_d, x_err, a_err, KP, KI, KD, alpha)

    # Plot
    plotResults(x, dx, a, da, x_d, dx_d, x_err, a_err, t)

    plt.show()