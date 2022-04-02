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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import GenerateGainBounds as GGB
from SuspensionMatrices import Suspension_8legs

# CONSTANTS AND FILEPATHS
# Change filepath location path to your GenerateGainsConstants.csv (as if you were running this code from the
# Trajectory Planner directory
FILEPATH_INTERACTION_CSV = './DynamicsInteractionConstants.csv'
FILEPATH_DEFAULT_CSV = './GenerateGainsConstants_default.csv'

# Function: Extract Data
# Extracts data from the .csv file generated from GenerateConstants.py and GenerateGainBounds.py
def ExtractDefaultData():
    # Extracts data and puts into list
    with open(FILEPATH_DEFAULT_CSV, 'rb') as myfile:
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


# Function: Extract data from mass variation .csv
# Extracts the constants data that is altered by changing the mass
def ExtractInteractionData():
    # Extracts data and puts into list
    with open(FILEPATH_INTERACTION_CSV, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)

    # Create empty lists
    radius_list = []
    stiffness_list = []
    damping_list = []
    kTau_list = []
    maxTau_list = []
    maxPs_list = []
    k_list = []
    minEigBenv_a_list = []
    minEigBenv_p_list = []
    maxEigBenv_list = []
    kBenv_a_list = []
    kBenv_p_list =[]

    for k in range(1, len(data)):
        radius_list.append(float(data[k][0]))
        stiffness_list.append(float(data[k][1]))
        damping_list.append(float(data[k][2]))
        kTau_list.append(float(data[k][3]))
        maxTau_list.append(float(data[k][4]))
        maxPs_list.append(float(data[k][5]))
        k_list.append(float(data[k][6]))
        minEigBenv_a_list.append(float(data[k][7]))
        minEigBenv_p_list.append(float(data[k][8]))
        maxEigBenv_list.append(float(data[k][9]))
        kBenv_a_list.append(float(data[k][10]))
        kBenv_p_list.append(float(data[k][11]))


    # Returns radius, stiffness, kTau, k lists
    return radius_list, stiffness_list, damping_list, kTau_list, maxTau_list, maxPs_list, k_list, \
           minEigBenv_a_list, minEigBenv_p_list, maxEigBenv_list, kBenv_a_list, kBenv_p_list


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
         maxG * np.sqrt((x_p ** 2) + (x_a ** 2)) + \
         maxTau * np.sqrt((x_p ** 2) + (x_a ** 2)) + \
         alpha * maxEigBenv * ((x_p ** 2) + (x_a ** 2))
    return xi


# Function: Get L
# Calculates the value of L according to the equations in the paper
def GetL(x_p, dx_p, x_a, dx_a, minEigK_p, minEigB_p, sus):
    L = -alpha * (minEigK_p - k) * (x_p ** 2) + \
        -(minEigB_p) * (dx_p ** 2) - minEigBenv_p * (dx_p**2) + \
        -alpha * (K_P - (1/alpha) * K_I - kG - kTau) * (x_a ** 2) + \
        -K_D * (dx_a ** 2) - minEigBenv_a * (dx_a ** 2) + \
        alpha * maxEigH * ((dx_p ** 2) + (dx_a ** 2)) + kK * kX * (dx_p + alpha * x_p) * x_p + \
        alpha * kC * np.sqrt((x_p ** 2) + (x_a ** 2)) * ((dx_p ** 2) + (dx_a ** 2)) + \
        0.5 * (kK + alpha * kB + alpha * kBenv_p) * (x_p ** 2) * dx_p + \
        0.5 * alpha * kBenv_a * (x_a ** 2) * dx_a

    return L


# Function: Find Min Stiffness and Damping Eigenvalues (Passive States)
# Calculates the minimum eigenvalues of the stiffness and damping matrices for a range of passive states
def FindMinEigKB_passive(x_p, sus):
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

    return min(eigenK), min(eigenB)


# Function: Find Largest Xi
# Calculates the largest Xi value for the stable region of the stability region plot
def FindLargestXi(x_p, dx_p, x_a, dx_a, xi, L, sus):
    xi_smallest = np.amax(xi) # smallest xi for unstable
    for ii in range(np.size(dx_p)):
        for jj in range(np.size(x_p)):
            if (L[ii,jj] > 0) and (xi[ii,jj] < xi_smallest):
                xi_smallest = xi[ii, jj]

    return xi_smallest


# Function: Plot Contour Figures
# Plots the stability regions of the system for passive and active states/derivatives
def PlotContourFigure(x_p, dx_p, x_a, dx_a, sus):
    # Reference global first_time, minEigK_p, minEigB_p to speed up plotting
    global first_time, minEigK_p, minEigB_p

    # Generate Xi and L datasets based on range of passive/active states
    Xi = np.zeros((np.size(dx_p), np.size(x_p)))
    L = np.zeros((np.size(dx_p), np.size(x_p)))
    if first_time:
        minEigK_p = np.zeros((np.size(dx_p), np.size(x_p)))
        minEigB_p = np.zeros((np.size(dx_p), np.size(x_p)))

    print "Generating xi and l datasets..."
    for ii in range(np.size(dx_p)):
        for jj in range(np.size(x_p)):
            if first_time:
                minEigK_p[ii,jj], minEigB_p[ii,jj] = FindMinEigKB_passive(x_p[jj], sus)
            Xi[ii,jj] = GetXi(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], sus)
            L[ii,jj] = GetL(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], minEigK_p[ii,jj], minEigB_p[ii,jj], sus)

    # If first time generating minEigK_p and minEigB_p, no need to calculate for rest of script
    if first_time:
        first_time = False

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
    xi_limit = FindLargestXi(x_p, dx_p, x_a, dx_a, Xi, L, sus)
    cmap_stable = colors.ListedColormap(['green', 'white'])
    bounds_stable = [np.amin(Xi), xi_limit, np.amax(Xi)]
    norm_stable = colors.BoundaryNorm(bounds_stable, cmap_stable.N)

    # Plot contour plots
    print "Generating contour plots..."
    if x_p[len(x_p)-1] == 0:
        x = x_a
        dx = dx_a
        labels = [r"$||\tilde{x}_a||$", r"$||\dot{\tilde{x}}_a||$"]
        # setlevels = [500, 1500]
        x_ticks = np.arange(0.5, 3, 1)
        y_ticks = np.arange(0.0, 12, 2.5)
        # plt.text(-0.02, 1, '(b)', fontsize=20)
    else:
        x = x_p
        dx = dx_p
        labels = [r"$||\tilde{x}_p||$", r"$||\dot{\tilde{x}}_p||$"]
        # setlevels = [200, 500, 1800]
        x_ticks = np.arange(0.15, 0.6, 0.2) # 0.35
        y_ticks = np.arange(0.0, 2.5, 0.5) # 1
        # plt.text(-0.25, 12, '(a)', fontsize=20)

    plt.locator_params(nbins=10)
    cs_xi = ax.contour(x, dx, Xi, colors='black', linewidths=3) #, levels=setlevels)
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

    # if x_p[len(x_p)-1] == 0:
    #     plt.savefig('active3d.eps', format='eps')
    # else:
    #     plt.savefig('passive3d.eps', format='eps')


# Function: Plot Limit Line Figure
# Plots only the limit line of the stability regions of the system for passive and active states/derivatives
def PlotLimitLineFigure(x_p, dx_p, x_a, dx_a, sus):
    # Reference global first_time, minEigK_p, minEigB_p to speed up plotting
    global first_time, minEigK_p, minEigB_p

    # Generate Xi and L datasets based on range of passive/active states
    Xi = np.zeros((np.size(dx_p), np.size(x_p)))
    L = np.zeros((np.size(dx_p), np.size(x_p)))
    if first_time:
        minEigK_p = np.zeros((np.size(dx_p), np.size(x_p)))
        minEigB_p = np.zeros((np.size(dx_p), np.size(x_p)))

    # print "Generating xi and l datasets..."
    for ii in range(np.size(dx_p)):
        for jj in range(np.size(x_p)):
            if first_time:
                minEigK_p[ii,jj], minEigB_p[ii,jj] = FindMinEigKB_passive(x_p[jj], sus)
            Xi[ii,jj] = GetXi(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], sus)
            L[ii,jj] = GetL(x_p[jj], dx_p[ii], x_a[jj], dx_a[ii], minEigK_p[ii,jj], minEigB_p[ii,jj], sus)

    # If first time generating minEigK_p and minEigB_p, no need to calculate for rest of script
    if first_time:
        first_time = False

    # Set up contour plots
    # print "Generating colormaps, finding max xi..."
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
    xi_limit = FindLargestXi(x_p, dx_p, x_a, dx_a, Xi, L, sus)
    cmap_stable = colors.ListedColormap(['green', 'white'])
    bounds_stable = [np.amin(Xi), xi_limit, np.amax(Xi)]
    norm_stable = colors.BoundaryNorm(bounds_stable, cmap_stable.N)

    # Plot contour plots
    # print "Generating contour plots..."
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
        setlevels = [300, 1300, 1800]
        x_ticks = np.arange(0.15, 0.6, 0.2) # 0.35
        y_ticks = np.arange(0.0, 2.5, 0.5) # 1
        # plt.text(-0.25, 12, '(a)', fontsize=20)

    plt.locator_params(nbins=10)
    cs_xi_lim = ax.contour(x, dx, Xi, levels=[0, float(xi_limit)], colors='black', linestyles='dashed', linewidths=3)

    cs_xi = ax.contour(x, dx, Xi, colors='black', linewidths=3, levels=setlevels)
    cs_l_s = ax.contourf(x, dx, Xi, levels=[0, xi_limit], cmap=cmap_stable, norm=norm_stable)
    cs_l_us = ax.contourf(x, dx, L, levels=0, cmap=cmap_unstable, norm=norm_unstable, alpha=0.7)

    ax.clabel(cs_xi, inline=1, fmt='%1.1f')
    # ax.clabel(cs_xi_lim, inline=1, fmt='%1.1f')
    plt.ylabel(labels[1], rotation='horizontal')
    ax.yaxis.set_label_coords(0, 1.07)
    plt.xlabel(labels[0])
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)

    # get limit curve first, then label limit curve (label will interfere with curve points
    curve = np.array(cs_xi_lim.collections[0].get_paths()[0].vertices, dtype = object)
    ax.clabel(cs_xi_lim, inline=1, fmt='%1.1f')

    return curve


# Function: Main
# Plots PD gain lower bounds and stability regions
if __name__ == "__main__":
    # Initialize and extract all needed constants
    sus = Suspension_8legs()
    c, alpha_bounds, alpha, gains = ExtractDefaultData()
    maxEigH, minEigH, maxG, kG, kK, kB, kX, maxEigK, maxEigB, minEigK_p, minEigB_p, kC, k, total_mass =\
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13]

    # Extract lists of constants adjusted by the variations of mass
    radius_list, stiffness_list, damping_list, kTau_list, maxTau_list, maxPs_list, k_list, \
    minEigBenv_a_list, minEigBenv_p_list, maxEigBenv_list, kBenv_a_list, kBenv_p_list = ExtractInteractionData()

    # Set maxTau and maxPs
    maxTau = maxTau_list[-1]
    maxPs = maxPs_list[-1]

    # Set PID gains based on variation in masses (I gain and alphas held constant)
    # K_P, K_I, K_D = gains[0], gains[1], gains[2] # default .csv values
    K_I = 20.0
    alpha = 0.5
    K_P_list = []
    K_D_list = []

    for index in range(len(radius_list)):
        # # Calculate PD gains and add to the list
        # K_P = (alpha ** 2) * maxEigH + kG + (1 / alpha) * K_I + kTau_list[index]
        # K_D = alpha * maxEigH - minEigBenv_a_list[index]

        # Or make the PD gains constant
        K_P = 96 #  88.6
        K_D = 13.5 # 13.5 # gains[2]

        K_P_list.append(K_P)
        K_D_list.append(K_D)

    # 100 N/m stiffness environment: PID = 95.73, 20, 13.5
    #

    # Retrieves curves of all xi_limit lines in preparation for 3D plottings
    curve_p_total = []
    curve_a_total = []
    first_time = True
    for n in range(0, len(radius_list)):
        print "Radius: " + str(radius_list[n])
        print "Stiffness: " + str(stiffness_list[n])
        print "Damping: " + str(damping_list[n])
        
        kTau, k = kTau_list[n], k_list[n]
        minEigBenv_a = minEigBenv_a_list[n]
        minEigBenv_p = minEigBenv_p_list[n]
        maxEigBenv = maxEigBenv_list[n]
        kBenv_a = kBenv_a_list[n]
        kBenv_p = kBenv_p_list[n]

        K_P = K_P_list[n]
        K_D = K_D_list[n]

        # Plotting curves of each added mass value
        # Plot stability region when active states are held at zero
        x_p = np.linspace(0, 0.6, 60) # (0, 0.6, 120)
        dx_p = np.linspace(0, 2, 60) # (0, 2, 120)
        x_a = np.zeros(np.size(x_p))
        dx_a = np.zeros(np.size(dx_p))

        curve_p = PlotLimitLineFigure(x_p, dx_p, x_a, dx_a, sus)    # note curve_p is an np.array object
        curve_p_total.append(curve_p)                               # note curve_p_total is a list of np.array objects

        # Plot stability region when passive states are held at zero
        x_a = np.linspace(0, 6, 60) # (0, 6, 120)
        dx_a = np.linspace(0, 20, 60) # (0, 20, 120)
        x_p = np.zeros(np.size(x_a))
        dx_p = np.zeros(np.size(dx_a))

        curve_a = PlotLimitLineFigure(x_p, dx_p, x_a, dx_a, sus)
        curve_a_total.append(curve_a)

        # Close generated contour plots
        if n != len(radius_list)-1:
            plt.close("all")

    # Then set up new 3d figure to plot all the curves
    plt.rc('font', size=10)

    fig_3d_passive = plt.figure()
    ax_passive = fig_3d_passive.gca(projection='3d')
    ax_passive.set_xlabel(r"$||\tilde{x}_p||$")
    ax_passive.set_ylabel("Environment Damping (Ns/m)")
    ax_passive.set_zlabel(r"$||\dot{\tilde{x}}_p||$")
    ax_passive.set_xlim(0.0, 0.6) # position mag
    ax_passive.set_ylim(0.0, 15) # radius is (0.0, 0.4)
    ax_passive.set_zlim(0.0, 2.0) # velocity mag
    ax_passive.invert_yaxis()

    fig_3d_active = plt.figure()
    ax_active = fig_3d_active.gca(projection='3d')
    ax_active.set_xlabel(r"$||\tilde{x}_a||$")
    ax_active.set_ylabel("Environment Damping (Ns/m)")
    ax_active.set_zlabel(r"$||\dot{\tilde{x}}_a||$")
    ax_active.set_xlim(0.0, 6.0) # position mag
    ax_active.set_ylim(0.0, 15) # radius is (0.0, 0.4)
    ax_active.set_zlim(0.0, 20.0) # velocity mag
    ax_active.invert_yaxis()

    for m in range(len(curve_p_total)):
        # Plot active state curves
        # if len(curve_p_total[m][:, 0]) != 2:
        x_p = curve_p_total[m][:, 0] # This gives x
        z_p = curve_p_total[m][:, 1] # This gives dx
        y_p = damping_list[m] * np.ones(np.size(x_p))

        ax_passive.plot(x_p, y_p, z_p)

        # Plot active state curves
        # if len(curve_a_total[m][:,0]) != 2:
        # if m < 120:
        x_a = curve_a_total[m][:, 0] # This gives x
        z_a = curve_a_total[m][:, 1] # This gives dx
        y_a = damping_list[m] * np.ones(np.size(x_a))

        ax_active.plot(x_a, y_a, z_a)

        # Square where the selected stability region is
        if m == 100:
            xa_square = [0, 3, 3, 0, 0]
            ya_square = [10, 10, 10, 10, 10]
            za_square = [0, 0, 10, 10, 0]
            ax_active.plot(xa_square, ya_square, za_square, linewidth=4, color='r')

            xp_square = [0, 0.6, 0.6, 0, 0]
            yp_square = [10, 10, 10, 10, 10]
            zp_square = [0, 0, 2, 2, 0]
            ax_passive.plot(xp_square, yp_square, zp_square, linewidth=4, color='r')

    plt.show()

   