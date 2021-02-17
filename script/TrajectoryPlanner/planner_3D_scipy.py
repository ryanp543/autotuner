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

# Class Trajectory Planner contains all the necessary getter and setter functions needed to either
# 1) Generate a csv file with step position commands for a specific acceleration profile
# 2) Implement class functions directly into simulation for real-time (albeit slow) optimization
class TrajectoryPlanner3D():

    # Function: Init
    # Pretty self-explanatory...
    def __init__(self, robot, sus):

        # Initializing necessary arrays for the active and passive states
        self.xa_init = None
        self.dxa_init = None
        self.xp_init = None
        self.dxp_init = None
        self.xa_target = None
        self.xp_target = None
        self.n_p = None
        self.n_a = None

        # Time parameters (default settings)
        self.T = 1.0
        self.dt = 0.005
        self.t_cmd = 10*self.dt
        self.n_coeffs = 5
        self.n_steps = int(round(self.T / self.dt))
        self.t = np.linspace(0, self.T, self.n_steps)

        # Optimizer parameters (default settings)
        self.w_xp = 3
        self.w_control = 0.07
        self.iteration = 0

        # Butterworth filter parameters (default settings)
        self.butter_order = 1
        self.butter_cutoff = 0.0625

        # Initializing parameters for stiff/damp matrices and Klampt models
        self.sus = sus
        self.robot = robot

        # Initializing parameters
        self.g = 9.81  # m/s^2, gravitationnal acceleration


    # Function: Set Initial
    # Sets up initial passive and active states and derivatives.
    def set_initial(self, xa_init, dxa_init, xp_init, dxp_init):
        self.xa_init = xa_init
        self.dxa_init = dxa_init
        self.xp_init = xp_init
        self.dxp_init = dxp_init
        self.n_p = int(np.shape(xp_init)[0])
        self.n_a = int(np.shape(xa_init)[0])


    # Function: Set Target
    # Sets up active state targets
    def set_target(self, xa_target):
        self.xa_target = xa_target


    # Function: Set Time Parameter
    # Sets up time parameters (i.e. time steps, time horizon, command time chunks, etc.)
    def set_time_parameters(self, T, dt, mult_cmd, n_coeffs):
        self.T = T                                  # T = secs, time horizon
        self.dt = dt                                # secs, time step
        self.n_steps = int(round(T / dt))           # number of time steps
        self.dt = T / self.n_steps                  # secs, actual time step
        self.t = np.linspace(0, T, self.n_steps)    # time vector in seconds
        self.t_cmd = mult_cmd*dt                    # time of command chunk

        self.n_coeffs = n_coeffs                    # number of coefficients (needs to be a factor of n_steps)


    # Function: Get Time Parameters
    # Returns a list of the time parameters
    def get_time_parameters(self):
        return [self.T, self.dt, self.n_steps, self.t_cmd, self.n_coeffs]


    # Function: Set Objective Weights
    # Sets up the weights of the least squares optimizer cost vector (i.e. w_xp is the weight on the passive states
    # and w_control is the weight on the active state acceleration control input)
    def set_objective_weights(self, w_xp, w_control):
        self.w_xp = w_xp
        self.w_control = w_control


    # Function: Set Butterworth Settings
    # Sets up butterworth filter used to smooth acceleration control input (order and cutoff frequency)
    def set_butter_settings(butter_order, butter_cutoff):
        self.butter_order = butter_order
        self.butter_cutoff = butter_cutoff


    # Function: Get Passive State Target
    # Returns the target static passive state, calculated by using the stiffness matrix and gravity vectors
    # calculated using Klampt
    def get_xp_target(self, x_p):
        x = [0, 0, x_p[0], 0, x_p[1], x_p[2]] + list(self.xa_target)
        self.robot.setConfig(x)
        G = self.robot.getGravityForces([0, 0, -self.g])
        K = np.asarray(self.sus.GetStiffnessMatrix(qPitch=x_p[1], qRoll=x_p[2]))
        G_p = np.array([G[2], G[4], G[5]])
        v = G_p + np.dot(K, x_p)
        return np.dot(v, v)


    # Function: Function Evaluation
    # Calculates the current polynomial at all time steps
    def f_eval(self, time, coeffs):
        result = np.zeros((len(time)))

        t_chunks = np.split(time, len(coeffs))  # split time in self.n_coeffs groups

        for idx, t in enumerate(time):
            for chunk_idx, chunk in enumerate(t_chunks):
                if t in chunk:
                    result[idx] = coeffs[chunk_idx]
        return result


    # Function: Input Evaluation
    # Objective function of least squares optimizer. For each active state, calculates control input using coeffs
    def u_eval(self, coeffs):
        u = list()
        b, a = signal.butter(self.butter_order, self.butter_cutoff) # first order, cutoff 62.5 hz
        for axis_idx in range(self.n_a):
            # Select cofficients of axis axis_idx
            axis_coeffs = coeffs[axis_idx * self.n_coeffs:(axis_idx + 1) * self.n_coeffs]

            # Evaluate the current polynomial at all time steps
            axis_u = self.f_eval(self.t, axis_coeffs)

            # Apply filter and append
            axis_u = signal.filtfilt(b, a, axis_u, padlen=20)
            u.append(axis_u)

        u = np.stack(u, axis=1)
        u = u.reshape((self.n_steps * self.n_a))

        return u


    # Function: Forward Simulation Step
    # Calculates the passive and active states and derivatives of the next time step
    def forward_sim_step(self, xp, dxp, xa, dxa, ddxa):
        # For robot class function callbacks, state is [x y z yaw pitch roll 4DOF]
        state = np.concatenate(([0.0, 0.0], xp[0], 0, xp[1], xp[2], xa), axis=None)
        dstate = np.concatenate(([0.0, 0.0], dxp[0], 0, dxp[1], dxp[2], dxa), axis=None)
        self.robot.setConfig(state)
        self.robot.setVelocity(dstate)

        # Collect H, G, C, K, and B matrices and vectors. Resize for calculations
        H_mat = np.asarray(self.robot.getMassMatrix())
        H_mat = np.delete(H_mat, [0, 1, 3], 0)
        H_mat = np.delete(H_mat, [0, 1, 3], 1)
        g_vec = self.robot.getGravityForces((0, 0, -self.g))
        g_vec = np.array(g_vec)
        g_vec = np.delete(g_vec, [0, 1, 3])
        c_vec = self.robot.getCoriolisForces()
        c_vec = np.array(c_vec)
        c_vec = np.delete(c_vec, [0, 1, 3])
        Ks_mat = np.asarray(self.sus.GetStiffnessMatrix(z=xp[0], qPitch=xp[1], qRoll=xp[2]))
        Kb_mat = np.asarray(self.sus.GetDampingMatrix(z=xp[0], qPitch=xp[1], qRoll=xp[2]))

        # Resizing dynamics term to state vector relevent to the problem
        H11_mat = H_mat[0:self.n_p, 0:self.n_p]
        H12_mat = H_mat[0:self.n_p, self.n_p:(self.n_p + self.n_a)]

        Gp_vec = g_vec[0:self.n_p]
        Ga_vec = g_vec[self.n_p:self.n_a + self.n_p]
        Cp_vec = c_vec[0:self.n_p]

        # Right hand side of dynamics equation (without inertial matrix inverse)
        rhs = -(Cp_vec + np.matmul(Kb_mat, dxp) + np.matmul(Ks_mat, xp) + Gp_vec + np.matmul(H12_mat, ddxa))
        # rhs = -(np.matmul(Kb_mat,dxp) + np.matmul(Ks_mat,xp) + Gp_vec )
        # print(-(np.matmul(Kb_mat,dxp) + np.matmul(Ks_mat,xp) + Gp_vec ) )
        (ddxp, _, _, _) = np.linalg.lstsq(H11_mat, rhs, rcond=None)

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


    # Function: Objective Function
    # Objective function to be solved using least squares optimizer
    def objective(self, coeffs):
        # Unwrap coeffs to polynomials
        u = self.u_eval(coeffs)

        _, _, _, _, delta_xp, delta_xa = self.forward_sim(u)

        # Calculate cost vector using weights and input
        cost_vector = list()
        cost_vector.append(self.w_xp * delta_xp.flatten())
        cost_vector.append(delta_xa.flatten())
        cost_vector.append(self.w_control * u.flatten())
        cost_vector = np.concatenate((cost_vector), axis=None)

        # Prints out "loading bar" to show optimization progress
        self.iteration += 1
        print("|"*self.iteration, end='\r')

        return cost_vector


    # Function: Get Commands
    # Double integrates control input to calculate position commands, then discretizes position command array into
    # step commands of length t_cmd. Used for practical real-life implementation
    def get_commands(self, ddu):

        # Double integration
        du = np.zeros((self.n_steps, self.n_a))
        u = np.zeros((self.n_steps-1, self.n_a)) # remember to add xa_init to u
        u[0, :] = xa_init[:]
        for ii in range(self.n_a):
            for kk in range(1, self.n_steps):
                du[kk, ii] = du[kk-1, ii] + ddu[kk-1, ii] * self.dt

            for jj in range(1, self.n_steps-1):
                u[jj, ii] = u[jj-1, ii] + du[jj-1, ii] * self.dt

        # Split u into discrete command steps
        for mm in range(self.n_a):
            for nn in range(0, self.n_steps, int(round(self.t_cmd/self.dt))):
                u[nn:nn+int(round(self.t_cmd/self.dt)), mm] = u[nn, mm]

        return u, du


    # Function: Get Input
    # Uses the least squares optimizer to calculate the control acceleration input
    def get_input(self):
        # Initialize initial coefficients
        coeffs_init = np.zeros((self.n_a * self.n_coeffs))

        # Calculating passive target state from solving the statices equation at xa_target
        xp_target0 = np.zeros((self.n_p))
        self.xp_target = np.asarray(fmin(self.get_xp_target, xp_target0, xtol=0.000001, disp=False))

        # Solve using least squares optimizer
        start = time.time()
        self.iteration = 0
        print("Optimizing trajectory:")
        sol = least_squares(self.objective, coeffs_init, method='lm', jac='2-point', verbose=2, max_nfev=10)
        coeffs = sol.x
        ddu = self.u_eval(coeffs)
        ddu = ddu.reshape((self.n_steps, self.n_a))
        print("Elapsed time: " + str(time.time() - start))

        return ddu


    # Function: Plot Results
    # Plots the input acceleration, associated position commands, etc
    def plot_results(self, u_command, ddu, xp, xa, dxp, dxa, delta_xp, delta_xa):
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        # Visualize
        fig, ax = plt.subplots(4, sharex=True, figsize=(16, 9))
        fig.align_ylabels()
        ax[0].plot(self.t, xp[:, 0], label='z')
        ax[0].plot(self.t, xp[:, 1], label='pitch')
        ax[0].plot(self.t, xp[:, 2], label='roll')
        ax[0].legend()
        # ax[1].plot(t, xa[:, 0], label='motor1')
        # ax[1].plot(t, xa[:, 1], label='motor2')
        # ax[1].plot(t, xa[:, 2], label='motor3')
        # ax[1].plot(t, xa[:, 3], label='motor4')
        # ax[1].legend()
        ax[1].plot(self.t, delta_xa[:, 0], label='error_motor1')
        ax[1].plot(self.t, delta_xa[:, 1], label='error_motor2')
        ax[1].plot(self.t, delta_xa[:, 2], label='error_motor3')
        ax[1].plot(self.t, delta_xa[:, 3], label='error_motor4')
        ax[1].legend()
        ax[2].plot(self.t, ddu[:, 0], label='ddx_a_1')
        ax[2].plot(self.t, ddu[:, 1], label='ddx_a_2')
        ax[2].plot(self.t, ddu[:, 2], label='ddx_a_3')
        ax[2].plot(self.t, ddu[:, 3], label='ddx_a_4')
        ax[2].legend()
        ax[3].plot(self.t[:-1], u_command[:, 0], label='u_cmd_1')
        ax[3].plot(self.t[:-1], u_command[:, 1], label='u_cmd_2')
        ax[3].plot(self.t[:-1], u_command[:, 2], label='u_cmd_3')
        ax[3].plot(self.t[:-1], u_command[:, 3], label='u_cmd_4')
        ax[3].legend()
        ax[3].set_xlabel('Time [s]')
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
    xa_init = np.array([0.0, 0.0, 0.0, 0.0])
    dxa_init = np.zeros((4))
    xp_init = np.zeros((3)) # xp_target[:]
    dxp_init = np.zeros((3))

    # Initialize planner object and set parameters
    planner = TrajectoryPlanner3D(robot, sus)

    # Set initial and target positions
    planner.set_initial(xa_init, dxa_init, xp_init, dxp_init)

    """
    INDIVIDUAL ACTIVE STATE TARGET
    Calculates position commands for single batch of target active states.
    """
    # # For one single active state target
    # # Target states
    # xa_target = np.array([1.00, -np.pi/2, 1.00, np.pi/4])
    # planner.set_target(xa_target)
    #
    # # Get input and double integrate to get position commands
    # ddu = planner.get_input()
    # u_command, du_command = planner.get_commands(ddu)
    #
    # # Plot results
    # xp, xa, dxp, dxa, delta_xp, delta_xa = planner.forward_sim(ddu)
    # planner.plot_results(u_command, ddu, xp, xa, dxp, dxa, delta_xp, delta_xa)

    """
    MULTIPLE ACTIVE STATE TARGETS
    Calculates position commands for multiple arm orientations, then adds position commands to a .csv file for 
    Gazebo simulation. 
    """

    # Array of desired target active states
    q1_d = [-2.0944, -1.0472, 1.0472, 2.0944]
    q2_d = [-1.0472, -2.0944, -3.1415]
    q3_d = [0.5236, 1.000, 2.094]
    planner.set_objective_weights(3, 0.01) # acceleration, control
    planner.set_time_parameters(1.0, 0.005, 10, 5) # T, dt, mult_cmd, n_coeffs

    # Calculate position commands and add to .csv file to be read by Gazebo simulation file run_simulation2.py
    with open('./planned_trajectory.csv', 'w') as myfile:
        csvwriter = csv.writer(myfile, delimiter=',')
        csvwriter.writerow(planner.get_time_parameters())

        sim_count = 1
        num_simulations = len(q1_d) * len(q2_d) * len(q3_d)
        for ii in range(len(q1_d)):
            for jj in range(len(q2_d)):
                for kk in range(len(q3_d)):
                    print("Running Simulation", sim_count, "of", num_simulations)
                    xa_target = np.array([q1_d[ii], q2_d[jj], q3_d[kk], 0])
                    planner.set_target(xa_target)
                    ddu = planner.get_input()
                    u_command, du_command = planner.get_commands(ddu)

                    # UNCOMMENT LINES BELOW to show plot of position commands, acceleration profile, etc
                    # xp, xa, dxp, dxa, delta_xp, delta_xa = planner.forward_sim(ddu)
                    # planner.plot_results(u_command, ddu, xp, xa, dxp, dxa, delta_xp, delta_xa)

                    csvwriter.writerow([q1_d[ii], q2_d[jj], q3_d[kk]])
                    for k in range(0, np.shape(u_command)[1]):
                        csvwriter.writerow(u_command[:, k].tolist())

                    sim_count += 1

