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


# Note: edit for specific robot
FILEPATH = './robot_sim.xml'

if __name__ == "__main__":
    # Upload URDF file, klampt models.
    world = klampt.WorldModel()
    res = world.readFile(FILEPATH)

    robot = world.robot(0)
    sus = Suspension_8legs()
    print "Number of DOFs: " + str(robot.numLinks())
    print "Number of Links: " + str(robot.numDrivers())

    # xp_init = [0, 0, 0]
    # xp = scipy.optimize.fmin(getXpTarget, xp_init, args=([0,0,0], [0,0,0]), xtol=0.000001, disp=False)
    # print(xp)
    # At rest position: [-0.04712687  0.00408563  0.11022556]

    x_config = [0, 0, -0.04712687, 0, 0.00408563, 0.11022556, 0, 0, 0, 0]
    dx_config = [0] * robot.numLinks()

    robot.setConfig(x_config)
    robot.setVelocity(dx_config)
    gravity = (0, 0, -9.81)

    H = np.asarray(robot.getMassMatrix())
    H = np.delete(H, [0, 1, 3, 6, 7, 8, 9], 0)
    H = np.delete(H, [0, 1, 3, 6, 7, 8, 9], 1)
    g = np.asarray(robot.getGravityForces(gravity))
    g = np.delete(g, [0, 1, 3, 6, 7, 8, 9])
    C = np.asarray(robot.getCoriolisForceMatrix())
    C = np.delete(C, [0, 1, 3, 6, 7, 8, 9], 0)
    C = np.delete(C, [0, 1, 3, 6, 7, 8, 9], 1)

    Ks = np.asarray(sus.GetStiffnessMatrix(z=x_config[2], qPitch=x_config[4], qRoll=x_config[5]))
    Bs = np.asarray(sus.GetDampingMatrix(z=x_config[2], qPitch=x_config[4], qRoll=x_config[5]))

    zero = np.zeros((3, 3))
    I = np.identity(3)

    freq = np.logspace(-1, 2, num=10000)

    fig = plt.figure(1, figsize=(12, 14), dpi=80)
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.rc("font", size=15)

    fig2 = plt.figure(2, figsize=(12, 14), dpi=80)
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.rc("font", size=15)

    labels = ["z", r"$\theta$", r"$\phi$"]
    ylimits = [(0, 0.005), (0, 0.005), (0, 0.05)]

    ax = []
    ax2 = []

    for k in range(1,4):
        plt.figure(1)
        ax_sub = fig.add_subplot(3, 1, k)
        ax.append(ax_sub)
        ax_sub.tick_params(width=2, length=8)
        ax_sub.xaxis.set_ticklabels([])
        plt.ylabel("Normalized " + labels[k-1] + " Amp (m/N)")
        plt.ylim(ylimits[k-1])
        plt.xlim((0.1, 100))
        plt.setp(ax_sub.spines.values(), linewidth=2)
        plt.grid()

        plt.figure(2)
        ax_sub = fig2.add_subplot(3, 1, k)
        ax2.append(ax_sub)
        ax_sub.tick_params(width=2, length=8)
        ax_sub.xaxis.set_ticklabels([])
        plt.ylabel("Phase " + labels[k-1] + " (rad)")
        # plt.ylim
        plt.xlim((0.1, 100))
        plt.setp(ax_sub.spines.values(), linewidth=2)
        plt.grid()


    fraction = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
    for frac in fraction:

        M_top = np.concatenate((I, zero), axis=1)
        M_bot = np.concatenate((zero, H), axis=1)
        M = np.concatenate((M_top, M_bot), axis=0)

        D_top = np.concatenate((zero, -I), axis=1)
        D_bot = np.concatenate((Ks, frac*(C+Bs)), axis=1)
        D = np.concatenate((D_top, D_bot), axis=0)

        norm_amp_z = []
        norm_amp_theta = []
        norm_amp_psi = []
        phase_z = []
        phase_theta = []
        phase_psi = []
        f = np.array([0, 0, 0, 1, 1, 1])

        mult = 0
        for k in range(np.size(freq)):
            TF = M*2*np.pi*freq[k]*1j + D

            Y = np.dot(np.linalg.inv(TF), f)
            Y_conj = np.conjugate(Y)
            amp = np.sqrt(np.real(np.multiply(Y, Y_conj)))
            phase = np.real((1/(2*1j))*np.log(np.divide(Y_conj, Y)))

            # Append amplitude vector
            norm_amp_z.append(amp[0])
            norm_amp_theta.append(amp[1])
            norm_amp_psi.append(amp[2])

            # Append phase vector
            phase_z.append(phase[0])
            phase_theta.append(phase[1])
            phase_psi.append(phase[2])

        # Figure 1 amplitude plots
        plt.figure(1)
        plt.sca(ax[0])
        plt.semilogx(freq, norm_amp_z)

        plt.sca(ax[1])
        plt.semilogx(freq, norm_amp_theta)

        plt.sca(ax[2])
        plt.semilogx(freq, norm_amp_psi)
        plt.xlabel("Frequency (Hz)")

        # Figure 2 phase plots
        plt.figure(2)
        plt.sca(ax2[0])
        plt.semilogx(freq, np.unwrap(2 * np.asarray(phase_z)) / 2)

        plt.sca(ax2[1])
        plt.semilogx(freq, np.unwrap(2 * np.asarray(phase_theta)) / 2)

        plt.sca(ax2[2])
        plt.semilogx(freq, np.unwrap(2 * np.asarray(phase_psi)) / 2)
        plt.xlabel("Frequency (Hz)")


    fig.align_labels()

    plt.sca(ax[0])
    fraction_str = [str(elem) for elem in fraction]
    plt.legend(fraction_str, title="Damping Fraction")

    plt.show()



