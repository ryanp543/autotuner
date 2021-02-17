# RaPID Autotuner

## Flexible-Base Manipulator Control Autotuner

Welcome to the RaPID Autotuner! This library allows one to take a Unified Robot Description Format (URDF) file and matrices describing the passive suspension as inputs in order to generate PID gains that guarantee stability for robotic arms mounted on flexible bases. 

The files provided in this repository are intended to be placed into a ROS package if you are using Gazebo to run simulations, as the provided example does. However, if you only desire to use the scripts to generate the PID gains and/or optimize a trajectory, feel free to just run the Python files in the `/script/TrajectoryPlanner` directory according to the steps below. Otherwise, make sure to create a catkin package in your `src` file, delete the default `package.xml` and `CMakeLists.txt` files, and clone this repository into your package folder (and run `catkin_make` or `catkin build`, of course). 

![alt text](/flowchart.JPG?raw=true)

These open-sourced scripts are associated with the following paper: (To be added soon hopefully)

* [Requirements](#Requirements)
* [Autotuner Inputs](#Autotuner-Inputs)
* [Generating PID Gains](#Generating-PID-Gains)
* [Trajectory Planner Optimizer](#Trajectory-Planner-Optimizer)
* [Provided Simulation Example](#Provided-Simulation-Example)
* [Contributors](#Contributors)
* [Citations](#Citations)

## Requirements

All Python files are compatible with Python 2. Development of this library was done on an Ubuntu 18.04.5 operating system.

## Autotuner Inputs

The inputs into the autotuner is a `.urdf` file and a Python file containing the stiffness and damping matrices of the suspension. In this example, the `.urdf` file was created using a [Solidworks to URDF exporter](http://wiki.ros.org/sw_urdf_exporter) for the main body and arm (excluding the "legs" of the suspension). The "legs" of the suspension are added separately to the `.urdf` file. For an example, see the `robot_controlsim_v3.urdf` file in the `urdf` folder.

The stiffness and damping matrix equations of the suspension were formulated using [Motion Genesis](http://www.motiongenesis.com/) and then written into a Python file. Nonlinearities were replaced with Taylor Series equivalents. 

## Generating PID Gains

First steps:

1. Download [Klamp't](https://github.com/krishauser/Klampt) (Build from source if pip install doesn't work. Follow online instructions.)
2. Convert `.xacro` to `.urdf` if the file is not already. There have been issues with having `.STL` files under the visual geometry tag, so try to avoid adding STLs. 
3. Edit `robot_sim.xml` in `/script/TrajectoryPlanner` such that "file=" points to your `.urdf` file
4. Run `GenerateConstants.py` to generate `.csv` file with constants

Repeat Step 5-8 until desired stability regions are achieved

5. Adjust alpha and K_I (under "Adjustable Parameters") as desired
6. Run `GenerateGains.py` to generate PID bounds based on alpha and K_I
7. Set desired "Selected PID Gains" in `.csv` file as desired based on the bounds generated by `GenerateGains.py`
8. Run `GeneratePlots.py` to view stability region plots and lower bounds on selected alpha/KI

Note: Running `GenerateGainBounds.py` with the `GenerateGainsConstants.csv` open will change values in the `.csv`, but you will not see them right away. To see the changes made to the `.csv` file, close and then reopen the `GenerateGainsConstants.csv`.

## Trajectory Planner Optimizer

The trajectory optimizer class can be found in the planner_3D_scipy.py of the /script/TrajectoryPlanner directory. The class and its functions can be imported into any desired Python file. An example of how to use the class can be seen in the main function of planner_3D_scipy.py.

For the provided example:

1. If using trajectory planner, run planner_3D_scipy.py from the Trajectory Planner directory after setting final arm orientations in the main function
2. This generates a trajectory plan in planned_trajectory.csv that can be used when running Gazebo simulations. 
3. An example of using this .csv file for Gazebo simulations can be seen in run_simulation_2.py in the "script" directory.

## Provided Simulation Example

In the outermost package directory, there are several folders involved in the simulation. The "meshes" folder contains all the STL files of the Bioinstrumentation Lab agricultural robot prototype, Rover II. The "src" folder contains C++ plugins for the Gazebo simulation. The other folders are relatively self-explanatory--"config" contains a .yaml file to change the PID gains of the ROS joint controllers, "launch" contains launch files, and "urdf" contains .xacro description files of Rover II.

The provided example (the Python files in the "script" directory) is specific to Rover II. The three clients were used for debugging purposes and Gazebo simulations. The other files are for running Gazebo simulations after spawning the world and then spawning the robot via "roslaunch autotuner spawn.launch". 

The first, run_simulation_1.py, sends single final state commands for 36 different orientations to the robotic arm, records the joint data over time, and outputs .csv files for plotting. The second, run_simulation_2.py, takes the series of position commands generated by the trajectory optimizer for the 36 orientations, sends them to the robotic arm, records the joint data over time, and outputs .csv files. The final two plotting files, plot_simulation_data.py and plot_tracking_error.py, take those .csv files and outputs plots for the simulations. 

To run these simulations, three terminal shells are needed: one to roslaunch the gazebo world, one to roslaunch autotuner spawn.launch, and one to rosrun autotuner run_simulation_1.py or rosrun autotuner run_simulation_2.py. Make sure to source your bash file and call all ROS commands from your catkin workspace directory, not the package directory. 

#### Note on Using Tuned PID Gains

In this example, ros_control is used to control the simulated rover and robotic arm. Specifically, each joint is monitored by an effort controller in the config directory. The PID gains generated by the autotuner are inputted into the joints.yaml file for the robotic arm joints. Note that Joint 4 is the stepper motor wrist and thus did not require PID inputs. 

## Contributors

#### Primary Contributors
Marc-Andre Begin and Ryan Poon are the primary contributors of this project.

#### Other Contributors
Professor Ian Hunter served as the main advisor.

## Citations:

#### For extracting dynamics of robotic arm from URDF file:
Kris Hauser. (2016) Robust Contact Generation for Robot Simulation with Unstructured Meshes. Robotics Research, 114. 

#### For trajectory optimization and generating constants:
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272




