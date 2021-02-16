# RaPID Autotuner

Welcome to the RaPID Autotuner! These open-sourced scripts are associated with the following paper: (To be added soon hopefully)

TLDR; This library allows one to take a Unified Robot Description Format (URDF) file and matrices describing the passive suspension as inputs in order to generate PID gains that guarantee stability for robotic arms mounted on flexible bases. 

The files provided in this repository are intended to be placed into a ROS package if you are using Gazebo to run simulations, as the provided example does. However, if you only desire to use the scripts to generate the PID gains and/or optimize a trajectory, feel free to just run the python files in the /script/TrajectoryPlanner directory according to the steps below. Otherwise, make sure to create a catkin package in your src file, delete the default package.xml and CMakeLists.txt files, and clone this repository into your package folder (and run catkin_make or catkin build, of course). 

![alt text](/flowchart.JPG?raw=true)

## Generating PID Gains

First steps:

1. Download Klamp't (Build from source if pip install doesn't work. Follow online instructions.)
2. Convert xacro to urdf if the file is not urdf already. There have been issues with having STL files under the visual geometry tag, so try to avoid adding STLs. 
3. Edit robot_sim.xml in /script/TrajectoryPlanner such that "file=" points to your URDF file
4. Run GenerateConstants.py to generate .csv file with constants

Repeat Step 5-8 until desired stability regions are achieved

5. Adjust alpha and K_I (under "Adjustable Parameters") as desired
6. Run GenerateGains.py to generate PID bounds based on alpha and K_I
7. Set desired "Selected PID Gains" in .csv file as desired based on the bounds generated by GenerateGains.py
8. Run GeneratePlots.py to view stability region plots and lower bounds on selected alpha/KI

Note: Running GenerateGainBounds.py with the GenerateGainsConstants.csv open will change values in the .csv, but you will not see them right away. To see the changes made to the .csv file, close and then reopen the GenerateGainsConstants.csv

## Trajectory Planner/Optimizer

The trajectory optimizer class can be found in the planner_3D_scipy.py of the /script/TrajectoryPlanner directory. The class and its functions can be imported into any desired Python file. An example of how to use the class can be seen in the main function of planner_3D_scipy.py.

For the provided example:

1. If using trajectory planner, run planner_3D_scipy.py from the Trajectory Planner directory after setting final arm orientations in the main function
2. This generates a trajectory plan in planned_trajectory.csv that can be used when running Gazebo simulations. 
3. An example of using this .csv file for Gazebo simulations can be seen in run_simulation_2.py in the "script" directory.

## Provided Example

The provided example (the Python files in the script directory) is specific to the Rover II agricultural robot prototype developed at the MIT Bioinstrumentation Lab. The three clients were used for debugging purposes and Gazebo simulations. The other files are for running Gazebo simulations after spawning the world and then spawning the robot via "roslaunch autotuner spawn.launch". 

The first, run_simulation_1.py, sends single final state commands for 36 different orientations to the robotic arm, records the joint data over time, and outputs .csv files for plotting. The second, run_simulation_2.py, takes the series of position commands generated by the trajectory optimizer for the 36 orientations, sends them to the robotic arm, records the joint data over time, and outputs .csv files. The final two plotting files, plot_simulation_data.py and plot_tracking_error.py, take those .csv files and outputs plots for the simulations. 

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




