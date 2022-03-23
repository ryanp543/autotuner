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
def forward_sim_step(xp, dxp, xa, dxa, xp_d, dxp_d, xa_d, dx_d):
    # For robot class function callbacks, state is [x y z yaw pitch roll 4DOF]
    x = np.concatenate(([0.0, 0.0], xp[0], 0, xp[1], xp[2], xa), axis=None)
    dx = np.concatenate(([0.0, 0.0], dxp[0], 0, dxp[1], dxp[2], dxa), axis=None)
    robot.setConfig(x)
    robot.setVelocity(dx)

    # Set PID gain matrices
    alpha = 0.5
    P, I, D = 59, 20, 13.5
    KP = np.asarray([[P, 0, 0], [0, P, 0], [0, 0, P]])
    KD = np.asarray([[D, 0, 0], [0, D, 0], [0, 0, D]])
    KI = np.asarray([[I, 0, 0], [0, I, 0], [0, 0, I]])

    # Collect H, G, C matrices and vectors. Resize for calculations
    zero_mat = np.zeros((3,3))

    H_mat = np.asarray(robot.getMassMatrix())
    H_mat = np.delete(H_mat, [0, 1, 3, len(x)-1], 0)
    H_mat = np.delete(H_mat, [0, 1, 3, len(x)-1], 1)
    g_vec = np.asarray(robot.getGravityForces((0, 0, -9.81)))
    g_vec = np.delete(g_vec, [0, 1, 3, len(x)-1])
    c_vec = np.asarray(robot.getCoriolisForces())
    c_vec = np.delete(c_vec, [0, 1, 3, len(x)-1])

    # Create K matrix
    Ks_mat = np.asarray(sus.GetStiffnessMatrix(z=xp[0], qPitch=xp[1], qRoll=xp[2]))
    K_top = np.concatenate((Ks_mat, zero_mat), axis=1)
    K_bottom = np.concatenate((zero_mat, np.add(KP, -(1/alpha)*KI)), axis=1)
    K = np.concatenate((K_top, K_bottom), axis=0)

    # Create B matrix
    Kb_mat = np.asarray(sus.GetDampingMatrix(z=xp[0], qPitch=xp[1], qRoll=xp[2]))
    B_top = np.concatenate((Kb_mat, zero_mat), axis=1)
    B_bottom = np.concatenate((zero_mat, KD), axis=1)
    B = np.concatenate((B_top, B_bottom), axis=0)

    # Create I vector

    I_top = -np.dot(Ks_mat, xp_d)


    quit()

    # Right hand side of dynamics equation (without inertial matrix inverse)
    rhs = -(Cp_vec + np.matmul(Kb_mat, dxp) + np.matmul(Ks_mat, xp) + Gp_vec + np.matmul(H12_mat, ddxa))

    (ddxp, _, _, _) = np.linalg.lstsq(H11_mat, rhs, rcond=None)

    quit()

    # Kinematics calculations
    dxp_next = dxp + ddxp * self.dt
    dxa_next = dxa + ddxa * self.dt
    xp_next = xp + dxp * self.dt + 0.5 * ddxp * self.dt ** 2
    xa_next = xa + dxa * self.dt + 0.5 * ddxa * self.dt ** 2

    return (xp_next, dxp_next, xa_next, dxa_next)


# Function: Forward Simulation
# Returns full list of passive and active states, derivatives, and errors for plotting given control input.
def forward_sim(self, u):
    # Initializing passive and active states and derivatives
    xp = np.zeros((self.n_steps, self.n_p))
    xa = np.zeros((self.n_steps, self.n_a))
    dxp = np.zeros((self.n_steps, self.n_p))
    dxa = np.zeros((self.n_steps, self.n_a))
    ddxa = u.reshape((self.n_steps, self.n_a))

    delta_xp = np.zeros((self.n_steps, self.n_p))
    delta_xa = np.zeros((self.n_steps, self.n_a))

    # Assigning initial conditions
    xp[0, :] = self.xp_init
    xa[0, :] = self.xa_init
    dxp[0, :] = self.dxp_init
    dxa[0, :] = self.dxa_init

    delta_xp[0, :] = self.xp_init - self.xp_target
    delta_xa[0, :] = self.xa_init - self.xa_target

    # Forward simulation step
    for ii in range(self.n_steps - 1):
        xp[ii + 1, :], dxp[ii + 1, :], xa[ii + 1, :], dxa[ii + 1, :] = self.forward_sim_step(xp[ii, :], dxp[ii, :],
                                                                                            xa[ii, :], dxa[ii, :],
                                                                                            ddxa[ii, :])
        delta_xp[ii + 1, :] = xp[ii + 1, :] - self.xp_target
        delta_xa[ii + 1, :] = xa[ii + 1, :] - self.xa_target

    return xp, xa, dxp, dxa, delta_xp, delta_xa


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
    xa_init = np.array([0.0] * robot.numDrivers())
    dxa_init = np.zeros((robot.numDrivers()))
    xp_init = np.zeros((3)) # xp_target[:]
    dxp_init = np.zeros((3))

    xa_d = [1.0472, -2.0, 1.0]
    dxa_d = [0, 0, 0]
    xp_d = fmin(get_xp_target, xp_init, args=(xa_d, dxa_d), xtol=0.000001, disp=False)
    dxp_d = [0, 0, 0]

    forward_sim_step(xp_init, dxp_init, xa_init, dxa_init, xp_d, dxp_d, xa_d, dxa_d)

    print(robot.numDrivers())