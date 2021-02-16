#!/usr/bin/env python

import sys
import numpy as np
import scipy.optimize
import scipy.io
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sympy import symbols, Eq, solve
from matplotlib import colors


# import klampt
# import GenerateGainsConstants as GGC
# from SuspensionMatrices import Suspension_8legs
# FILEPATH = './agrobot_sim.xml'

def FindLargestXi(x_p, dx_p, x_a, dx_a, xi, L):
    xi_smallest = np.amax(xi) # smallest xi for unstable
    for ii in range(np.size(dx_p)):
        for jj in range(np.size(x_p)):
            if (L[ii,jj] > 0) and (xi[ii,jj] < xi_smallest):
                xi_smallest = xi[ii, jj]

    return xi_smallest


def PlotFigure(x_p, dx_p, x_a, dx_a, Xi, L):


    print "Generating colormaps, finding max xi..."
    plt.rc('font', size=20)
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    plt.subplots_adjust(left=0.15, top=0.85, bottom=0.25)
    plt.setp(ax.spines.values(), linewidth=3)
    ax.tick_params(width=3, length=6)

    cmap_unstable = colors.ListedColormap(['white', 'red'])
    bounds_unstable = [np.amin(L), 0, np.amax(L)]
    norm_unstable = colors.BoundaryNorm(bounds_unstable, cmap_unstable.N)

    xi_limit = FindLargestXi(x_p, dx_p, x_a, dx_a, Xi, L)
    cmap_stable = colors.ListedColormap(['green', 'white'])
    bounds_stable = [np.amin(Xi), xi_limit, np.amax(Xi)]
    norm_stable = colors.BoundaryNorm(bounds_stable, cmap_stable.N)

    print "Generating contour plots..."
    if x_p[len(x_p)-1] == 0:
        x = x_a
        dx = dx_a
        labels = [r"$||\tilde{x}_a||$", r"$||\dot{\tilde{x}}_a||$"]
        setlevels = [50, 150]
        x_ticks = np.arange(0.3, 1.3, 0.4)
        y_ticks = np.arange(0.0, 2.4, 0.4)
        # plt.text(-0.02, 1, '(b)', fontsize=20)
    else:
        x = x_p
        dx = dx_p
        labels = [r"$||\tilde{x}_p||$", r"$||\dot{\tilde{x}}_p||$"]
        setlevels = [30, 80]
        x_ticks = np.arange(0.1, 0.39, 0.1)
        y_ticks = np.arange(0.0, 1.2, 0.2)
        # plt.text(-0.25, 12, '(a)', fontsize=20)

    # plt.figure(1, figsize=(10, 10), dpi=80)
    plt.locator_params(nbins=10)
    cs_xi = ax.contour(x, dx, Xi, levels=setlevels, colors='black', linewidths=3)
    cs_xi_lim = ax.contour(x, dx, Xi, levels=[0, float(xi_limit)], colors='black', linestyles='dashed', linewidths=3)
    cs_l_s = ax.contourf(x, dx, Xi, levels=[0, xi_limit], cmap=cmap_stable, norm=norm_stable)
    cs_l_us = ax.contourf(x, dx, L, levels=0, cmap=cmap_unstable, norm=norm_unstable, alpha=0.7)
    plt.ylabel(labels[1], rotation='horizontal')
    ax.yaxis.set_label_coords(0, 1.07)
    plt.xlabel(labels[0])
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    ax.clabel(cs_xi_lim, inline=1, fmt='%1.1f', manual=True)
    ax.clabel(cs_xi, inline=1, fmt='%1.1f', manual=True)



if __name__ == "__main__":
    filepath = './ell_p.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        L = np.transpose(np.asarray([[float(y) for y in x] for x in data]))

    filepath = './xi_p.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        Xi = np.transpose(np.asarray([[float(y) for y in x] for x in data]))

    x_p = np.linspace(0, 0.4, 201)
    dx_p = np.linspace(0, 1, 501)
    x_a = np.zeros(np.size(x_p))
    dx_a = np.zeros(np.size(dx_p))

    PlotFigure(x_p, dx_p, x_a, dx_a, Xi, L)

    filepath = './ell_a.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        L = np.transpose(np.asarray([[float(y) for y in x] for x in data]))

    filepath = './xi_a.csv'
    with open(filepath, 'rb') as myfile:
        csvreader = csv.reader(myfile, delimiter=',')
        data = []
        for row in csvreader:
            data.append(row)
        Xi = np.transpose(np.asarray([[float(y) for y in x] for x in data]))

    x_a = np.linspace(0, 1.3, 66)
    dx_a = np.linspace(0, 2, 201)
    x_p = np.zeros(np.size(x_a))
    dx_p = np.zeros(np.size(dx_a))

    PlotFigure(x_p, dx_p, x_a, dx_a, Xi, L)
    plt.show()

    # PLOTTING PASSIVE X
   #  x_p = np.linspace(0, 0.35, 60)
   #  dx_p = np.linspace(0, 0.8, 60)
   #  x_a = np.zeros(np.size(x_p))
   #  dx_a = np.zeros(np.size(dx_p))
   #
   #  PlotFigure(x_p, dx_p, x_a, dx_a, c, gains, alpha, sus)
   #
   #  # PLOTTING ACTIVE X
   #  x_a = np.linspace(0, 3, 60)
   #  dx_a = np.linspace(0, 10, 60)
   #  x_p = np.zeros(np.size(x_a))
   #  dx_p = np.zeros(np.size(dx_a))
   #
   #  PlotFigure(x_p, dx_p, x_a, dx_a, c, gains, alpha, sus)
   #  plt.show()
   #
   #