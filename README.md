# RaPID Autotuner

Welcome to the RaPID Autotuner! These open-sourced scripts are associated with the following paper: (To be added soon hopefully)

TLDR; This library allows one to take a Unified Robot Description Format (URDF) file and matrices describing the passive suspension as inputs in order to generate PID gains that guarantee stability for robotic arms mounted on flexible bases. 

The files provided in this repository are intended to be placed into a ROS package if you are using Gazebo to run simulations, as the provided example does. However, if you only desire to use the scripts to generate the PID gains and/or optimize a trajectory, feel free to just run the python files in the /script/TrajectoryPlanner directory according to the steps below. Otherwise, make sure to create a catkin package in your src file, delete the default package.xml and CMakeLists.txt files, and clone this repository into your package folder (and run catkin_make or catkin build, of course). 

![alt text](https://githubt.com/ryanp543/autotuner/flowchart.JPG?raw=true)

Steps:

1. Download Klamp't (Build from source if pip install doesn't work! Follow online instructions.)
2. Convert xacro to urdf if the file is not urdf already. There have been issues with having STL files under the visual geometry tag, so try to avoid adding STLs. 
3. Run GenerateConstants.py to generate .csv file with constants

Repeat Step 4-7 until desired stability regions are achieved

4. Adjust alpha and K_I (under "Adjustable Parameters") as desired
5. Run GenerateGains.py to generate PID bounds based on alpha and K_I
6. Set desired "Selected PID Gains" in .csv file as desired based on the bounds generated by GenerateGains.py
7. Run GeneratePlots.py to view stability region plots and lower bounds on selected alpha/KI

Note: Running GenerateGainBounds.py with the GenerateGainsConstants.csv open will change values in the .csv, but you will not see them right away. To see the changes made to the .csv file, close and then reopen the GenerateGainsConstants.csv

Currently for 3DOF arms:

8. If using trajectory planner, run planner_3D_scipy.py from the Trajectory Planner directory after setting final arm orientations in the main function
9. This generates a trajectory plan in planned_trajectory.csv that can be used when running Gazebo simulations. 
10. An example of using this .csv file for Gazebo simulations can be seen in run_simulation2.py in the "script" directory.

Citations:



