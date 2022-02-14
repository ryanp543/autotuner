#!/usr/bin/env python

"""
Python program that first generates two plots: 1) One figure shows the lower bound of K_D with respect to a range of
alphas, with the selected K_D value shown as a red x, while 2) the second figure shows a gradient of K_P lower bounds
with respect to a range of alphas and K_I values. The next two plots show the stability regions when 3) active states/
derivatives are set to zero and when 4) passive states/derivatives are set to zero.

Date 2/2/2021
"""

import sys
import csv
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from sympy import symbols, Eq, solve
from matplotlib import colors, cm

import GenerateGainBounds as GGB
from SuspensionMatrices import Suspension_8legs

# CONSTANTS AND FILEPATHS
# Change filepath location path to your GenerateGainsConstants.csv (as if you were running this code from the
# Trajectory Planner directory
FILEPATH = './GenerateGainsConstants.csv'

# Function: Extract Data
# Extracts data from the .csv file generated from GenerateConstants.py and GenerateGainBounds.py
def ExtractData():
    # Extracts data and puts into list
    with open(FILEPATH, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)

    # Create list of constants
    constants = []
    for k in range(1, 15):
        constants.append(float(data[k][1]))

    # Create list of alpha bounds
    alpha_bounds = []
    for k in range(17, 22):
        try:
            alpha_bounds.append(float(data[k][1]))
        except ValueError:
            pass

    # Initialize selected alpha variable
    alpha = float(data[24][1])

    # Create list of selected gains
    gains = []
    for k in range(32, 35):
        gains.append(float(data[k][1]))

    return constants, alpha_bounds, alpha, gains


# Function: Get Gain Ranges
# Returns the ranges of alpha and PID gains, where the K_P range and K_D range are dependent on alpha and K_I.
def GetGainRanges(constants, alpha_bounds, gains):
    K_I, K_P, K_D = gains[0], gains[1], gains[2]

    alpha_lb = alpha_bounds[1]
    alpha_ub = min([alpha_bounds[0], alpha_bounds[2]])

    alpha_range = list(np.linspace(alpha_lb, alpha_ub, 1000))
    K_I_range = list(np.linspace(0, 100, 1200))
    K_D_range = []
    K_P_range = []

    for i in range(len(alpha_range)):
        K_D_range.append(GGB.GetGains(constants, alpha_range[i], K_I, printing=False)[2])

        K_P_row = []
        for j in range(len(K_I_range)):
            K_P_row.append(GGB.GetGains(constants, alpha_range[i], K_I_range[j], printing=False)[1])
        K_P_range.append(K_P_row)

    K_P_range = list(np.transpose(np.asarray(K_P_range)))

    return alpha_range, K_I_range, K_D_range, K_P_range


# Function: Get Xi
# Calculates the value of Xi according to the equations in the paper.
def GetXi(x_p, dx_p, x_a, dx_a, sus):
    xi = 0.5 * maxEigH * ((dx_p ** 2) + (dx_a ** 2) + (alpha ** 2) * ((x_p ** 2) + (x_a ** 2))) + \
         0.5 * maxEigK * (x_p ** 2) + \
         0.5 * K_P * (x_a ** 2) + \
         0.5 * alpha * maxEigB * (dx_p ** 2) + \
         0.5 * K_D * (dx_a ** 2) + \
         -0.5 * ((alpha ** 2) * minEigH) * ((x_p ** 2) + (x_a ** 2)) + \
         total_mass * 9.81 * x_p + \
         maxG * np.sqrt((x_p ** 2) + (x_a ** 2))
    return xi


# Function: Get L
# Calculates the value of L according to the equations in the paper
def GetL(x_p, dx_p, x_a, dx_a, sus):
    L = -alpha * (FindMaxEigKB_passive(x_p, sus)[0] - k) * (x_p ** 2) + \
        -(FindMaxEigKB_passive(x_p, sus)[1]) * (dx_p ** 2) + \
        -alpha * ((alpha ** 2) * maxEigH + kG - kG) * (x_a ** 2) + \
        -K_D * (dx_a ** 2) + \
        alpha * maxEigH * ((dx_p ** 2) + (dx_a ** 2)) + kK * kX * (dx_p + alpha * x_p) * x_p + \
        alpha * kC * np.sqrt((x_p ** 2) + (x_a ** 2)) * ((dx_p ** 2) + (dx_a ** 2)) + \
        0.5 * (kK + alpha * kB) * (x_p ** 2) * dx_p
    return L


# Function: Find Max Stiffness and Damping Eigenvalues (Passive States )
# Calculates the minimum eigenvalues of the stiffness and damping matrices for a range of passive states
def FindMaxEigKB_passive(x_p, sus):
    eigenK = []
    eigenB = []
    for weight in np.linspace(0, 1, 10):
        # Vary weights on pitch and roll to get all K's then take max min eigenvalue
        qPitch = np.sqrt((x_p ** 2) * weight)
        qRoll = np.sqrt((x_p ** 2) * (1 - weight))
        K = sus.GetStiffnessMatrix(qPitch=qPitch, qRoll=qRoll)
        B = sus.GetDampingMatrix(qPitch=qPitch, qRoll=qRoll)
        w_K, v_K = np.linalg.eig(K)
        w_B, v_B = np.linalg.eig(B)
        eigenK.append(min(w_K))
        eigenB.append(min(w_B))

    return [min(eigenK), min(eigenB)]


# Function: Find Largest Xi
# Calculates the largest Xi value for the stable region of the stability region plot
def FindLargestXi(x_p, dx_p, x_a, dx_a, xi, sus):
    xi_smallest = np.amax(xi) # smallest xi for unstable
    for ii in range(np.size(dx_p)):
        for jj in range(np.size(x_p)):
            if (GetL(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], sus) > 0) and (xi[ii,jj] < xi_smallest):
                xi_smallest = xi[ii, jj]

    return xi_smallest

# Function: Plot PD Gain Lower Bounds
# Plots the lower bounds of the PD gains given a range of alpha and K_I values
def PlotRangesFigure(alpha_range, K_I_range, K_D_range, K_P_range):
    plt.rc('font', size=20)

    # Plotting K_D lower bounds
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.tick_params(width=3, length=6)
    plt.subplots_adjust(left=0.15, bottom=0.2)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.plot(alpha_range, K_D_range, linewidth=3)
    plt.plot(0.5, 13.5, marker='x', markersize=10, markeredgewidth = 3, color="red")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\lambda_{min}(K_D)$", rotation='horizontal')
    ax.yaxis.set_label_coords(0, 1.05)
    # plt.yticks([10, 20, 30, 40])
    plt.grid()
    plt.savefig('lowerboundkd.eps', format='eps')

    # Plotting K_P lower bounds
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.tick_params(width=3, length=6)
    plt.subplots_adjust(left=0.15, bottom=0.2)
    plt.setp(ax.spines.values(), linewidth=3)
    # levels = range(10, 300, 1)
    # tick_levels = range(50, 300, 50)
    cs = ax.contourf(alpha_range, K_I_range, K_P_range, cmap=cm.PuBu_r) #, levels=levels)

    cbar = fig.colorbar(cs)
    # cbar.set_ticks(tick_levels)
    cbar.set_label(r"$\lambda_{min}(K_P)$", labelpad=-40, y=1.15, rotation=0)
    cbar.outline.set_linewidth(3)
    cbar.ax.tick_params(width=3)

    cs2 = ax.contour(cs, colors='k') #, levels=tick_levels)
    ax.clabel(cs2, inline=True, fmt='%1.1f', manual=True)
    plt.plot(0.5, 20, marker='x', markersize=10, markeredgewidth=3, color="red")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\lambda_{min}(K_I)$", rotation='horizontal')
    ax.yaxis.set_label_coords(0, 1.05)
    # plt.yticks([10, 35, 60, 90])
    plt.savefig('lowerboundkp.eps', format='eps')


# Function: Plot Contour Figures
# Plots the stability regions of the system for passive and active states/derivatives
def PlotContourFigure(x_p, dx_p, x_a, dx_a, sus):
    # Generate Xi and L datasets based on range of passive/active states
    Xi = np.zeros((np.size(dx_p), np.size(x_p)))
    L = np.zeros((np.size(dx_p), np.size(x_p)))

    print "Generating xi and l datasets..."
    for ii in range(np.size(dx_p)):
        for jj in range(np.size(x_p)):
            Xi[ii,jj] = GetXi(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], sus)
            L[ii,jj] = GetL(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], sus)

    # Set up contour plots
    print "Generating colormaps, finding max xi..."
    plt.rc('font', size=20)
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    plt.subplots_adjust(left=0.15, top=0.85, bottom=0.25)
    plt.setp(ax.spines.values(), linewidth=3)
    ax.tick_params(width=3, length=6)

    # Create colormap for unstable region
    cmap_unstable = colors.ListedColormap(['white', 'red'])
    bounds_unstable = [np.amin(L), 0, np.amax(L)]
    norm_unstable = colors.BoundaryNorm(bounds_unstable, cmap_unstable.N)

    # Create colormap for stable region (bounded by largest Xi for stable region)
    xi_limit = FindLargestXi(x_p, dx_p, x_a, dx_a, Xi, sus)
    cmap_stable = colors.ListedColormap(['green', 'white'])
    bounds_stable = [np.amin(Xi), xi_limit, np.amax(Xi)]
    norm_stable = colors.BoundaryNorm(bounds_stable, cmap_stable.N)

    # Plot contour plots
    print "Generating contour plots..."
    if x_p[len(x_p)-1] == 0:
        x = x_a
        dx = dx_a
        labels = [r"$||\tilde{x}_a||$", r"$||\dot{\tilde{x}}_a||$"]
        setlevels = [500, 1500]
        x_ticks = np.arange(0.5, 3, 1)
        y_ticks = np.arange(0.0, 12, 2.5)
        # plt.text(-0.02, 1, '(b)', fontsize=20)
    else:
        x = x_p
        dx = dx_p
        labels = [r"$||\tilde{x}_p||$", r"$||\dot{\tilde{x}}_p||$"]
        setlevels = [200, 500, 1800]
        x_ticks = np.arange(0.15, 0.6, 0.2) # 0.35
        y_ticks = np.arange(0.0, 2.5, 0.5) # 1
        # plt.text(-0.25, 12, '(a)', fontsize=20)

    plt.locator_params(nbins=10)
    cs_xi = ax.contour(x, dx, Xi, colors='black', linewidths=3, levels=setlevels)
    cs_xi_lim = ax.contour(x, dx, Xi, levels=[0, float(xi_limit)], colors='black', linestyles='dashed', linewidths=3)
    cs_l_s = ax.contourf(x, dx, Xi, levels=[0, xi_limit], cmap=cmap_stable, norm=norm_stable)
    cs_l_us = ax.contourf(x, dx, L, levels=0, cmap=cmap_unstable, norm=norm_unstable, alpha=0.7)
    ax.clabel(cs_xi, inline=1, fmt='%1.1f')
    ax.clabel(cs_xi_lim, inline=1, fmt='%1.1f')
    plt.ylabel(labels[1], rotation='horizontal')
    ax.yaxis.set_label_coords(0, 1.07)
    plt.xlabel(labels[0])
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)

    if x_p[len(x_p)-1] == 0:
        plt.savefig('active3d.eps', format='eps')
    else:
        plt.savefig('passive3d.eps', format='eps')


# Function: Main
# Plots PD gain lower bounds and stability regions
if __name__ == "__main__":
    # Initialize and extract all needed constants
    sus = Suspension_8legs()
    constants, alpha_bounds, alpha, gains = ExtractData()
    K_I, K_P, K_D = gains[0], gains[1], gains[2]
    c = constants[:]
    maxEigH, minEigH, maxG, kG, kK, kB, kX, maxEigK, maxEigB, minEigK_p, minEigB_p, kC, k, total_mass = \
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13]

    # Start generating plots
    print "==== Plots ===="
    print "Generating gain boundary plots..."

    # Plot PD gain lower bounds
    alpha_range, K_I_range, K_D_range, K_P_range = GetGainRanges(constants, alpha_bounds, gains)
    PlotRangesFigure(alpha_range, K_I_range, K_D_range, K_P_range)

    # Plot stability region when active states are held at zero
    x_p = np.linspace(0, 0.6, 60)
    dx_p = np.linspace(0, 2, 60)
    x_a = np.zeros(np.size(x_p))
    dx_a = np.zeros(np.size(dx_p))

    PlotContourFigure(x_p, dx_p, x_a, dx_a, sus)

    # Plot stability region when passive states are held at zero
    x_a = np.linspace(0, 3, 60)
    dx_a = np.linspace(0, 10, 60)
    x_p = np.zeros(np.size(x_a))
    dx_p = np.zeros(np.size(dx_a))

    PlotContourFigure(x_p, dx_p, x_a, dx_a, sus)

    # Plot colorbar for stability region (if needed for nice plots)
    # plt.figure()
    # plt.rc('font', size=30)
    # stable = mpatches.Patch(color='green', alpha=0.4, label='Stable Region')
    # unstable = mpatches.Patch(color='red', label=r"$\dot{V} > 0$")
    # plt.legend(handles=[stable, unstable], ncol=2)

    plt.show()

   