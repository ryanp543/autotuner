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
from scipy import signal
from scipy.optimize import least_squares, Bounds, fmin
import matplotlib.pyplot as plt
import matplotlib as mpl
import klampt


# Function: Get Passive State Target
# Returns the target static passive state, calculated by using the stiffness matrix and gravity vectors
# calculated using Klampt
def get_xp_target(x_p, x_a_target, dx_a_target):
    x = [0, 0, x_p[0], 0, x_p[1], x_p[2]] + list(x_a_target) + [0]
    robot.setConfig(x)
    G = robot.getGravityForces([0, 0, -9.81])
    K = np.asarray(sus.GetStiffnessMatrix(z=x_p[0], qPitch=x_p[1], qRoll=x_p[2]))
    G_p = np.array([G[2], G[4], G[5]])
    v = G_p + np.dot(K, x_p)
    return np.dot(v, v)


# Function: Forward Simulation Step
# Calculates the passive and active states and derivatives of the next time step
# (x_init, dx_init, a_hat_init, da_hat_init, x_d, dx_d)

def forward_sim_step(x, dx, a_hat, da_hat, x_d, dx_d, dt):
    # State is [x y z yaw pitch roll 4DOF]
    x_config = np.concatenate(([0.0, 0.0], x[0], 0, x[1:], 0), axis=None)
    dx_config = np.concatenate(([0.0, 0.0], dx[0], 0, dx[1:], 0), axis=None)

    robot.setConfig(x_config)
    robot.setVelocity(dx_config)

    # Set PID gain matrices
    alpha = 0.5
    P, I, D = 59.0, 20.0, 13.5
    KP = np.asarray([[P, 0, 0], [0, P, 0], [0, 0, P]])
    KI = np.asarray([[I, 0, 0], [0, I, 0], [0, 0, I]])
    KD = np.asarray([[D, 0, 0], [0, D, 0], [0, 0, D]])

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
    K_bottom = np.concatenate((zero_mat, np.add(KP, -(1/alpha)*KI)), axis=1)
    K = np.concatenate((K_top, K_bottom), axis=0)

    # Create B matrix
    Kb_mat = np.asarray(sus.GetDampingMatrix(z=x[0], qPitch=x[1], qRoll=x[2]))
    B_top = np.concatenate((Kb_mat, zero_mat), axis=1)
    B_bottom = np.concatenate((zero_mat, KD), axis=1)
    B = np.concatenate((B_top, B_bottom), axis=0)

    # Create I vector
    I_top = -np.matmul(Ks_mat, x_d[:3])
    I_bottom = np.matmul(KI, a_hat)
    I = np.concatenate((I_top, I_bottom), axis=0)

    # Calculate error vectors
    x_err = x - x_d

    # Right hand side of dynamics equation (without inertial matrix inverse)
    rhs = -c_vec - g_vec - np.matmul(K, x_err) - np.matmul(B, dx) + I
    (ddx, _, _, _) = np.linalg.lstsq(H_mat, rhs, rcond=None)

    # Kinematics calculations
    dx_next = dx + ddx*dt
    x_next = x + dx*dt + 0.5*ddx*(dt**2)

    # Calculate a_hat_dot and a_hat
    da_hat_next = -x_err[3:] - (1/alpha)*dx[3:]
    a_hat_next = a_hat + da_hat_next*dt
    # print("da: " + str(da_hat_next*dt))
    # print("a: " + str(a_hat_next))

    return x_next, dx_next, a_hat_next, da_hat_next


# Function: Forward Simulation
# Returns full list of passive and active states, derivatives, and errors for plotting given control input.
def simulate(x_init, dx_init, a_hat_init, da_hat_init, x_d, dx_d, dt):
    # Initializing passive and active states and derivatives
    test_length = 20 # in seconds
    t = np.asarray([n * dt for n in range(0, int(test_length/dt)+1)])

    x = np.asarray([x_init])
    dx = np.asarray([dx_init])
    a = np.asarray([a_hat_init])
    da = np.asarray([da_hat_init])


    for k in range(np.size(t)):
        # Calculate new states
        x_new, dx_new, a_new, da_new = forward_sim_step(x[k], dx[k], a[k], da[k], x_d, dx_d, dt)
        # print(x_new)

        # Add to arrays
        x = np.append(x, [x_new], axis=0)
        dx = np.append(dx, [dx_new], axis=0)
        a = np.append(a, [a_new], axis=0)
        da = np.append(da, [da_new], axis=0)
        # print(x)


    x_target = np.concatenate(([0.0, 0.0], x_d[0], 0, x_d[1:], 0), axis=None)
    robot.setConfig(x_target)
    g_v = np.asarray(robot.getGravityForces((0, 0, -9.81)))
    KI = np.asarray([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
    KI_inv = np.linalg.inv(KI)
    print(np.matmul(KI_inv, g_v[6:9]))

    # print(x[-1])

    motor_pos = x[1:, 4]
    plt.figure(1)
    plt.plot(t, motor_pos)
    plt.grid()
    plt.show()


# Function: Main
# Uses a least-squares optimizer to generate a trajectory path to minimize base vibrations, joint acceleration,
# settling time, and overshoot.
if __name__ == "__main__":
    # Intializations suspension and robot
    from SuspensionMatrices import Suspension_8legs
    sus = Suspension_8legs()  # 3DOF model of the suspension, use with GetStiffnessMatrix and GetDampingMatrix
    world = klampt.WorldModel()
    res = world.readFile("./robot_sim.xml")
    robot = world.robot(0)

    # Initial states
    # Note: In realtime implementation, these are updated to current state whenever the trajectory planner is calleds
    # The state is [z pitch roll 4DOF]
    x_init = np.zeros((6))
    dx_init = np.zeros((6))
    a_hat_init = np.zeros((3))
    da_hat_init = np.zeros((3))

    dx_d = np.zeros((6))
    # xa_d = [1.0472, -2.0, 1.0]
    xa_d =[1.0472, -np.pi/4, 0.0]
    xp_d = fmin(get_xp_target, x_init[:3], args=(xa_d, dx_d[:3]), xtol=0.000001, disp=False)
    x_d = np.concatenate((xp_d, xa_d), axis=0)

    dt = 0.001

    simulate(x_init, dx_init, a_hat_init, da_hat_init, x_d, dx_d, dt)


    print(robot.numDrivers())