cmake_minimum_required(VERSION 2.8.3)
project(autotuner)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
    gazebo_ros
    roscpp
    urdf 
    controller_manager 
    joint_state_controller 
    robot_state_publisher 
    joint_state_publisher_gui 
)
find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

# For set_speed_joint_plugin plugin
add_library(set_speed_joint_plugin src/set_speed_joint_plugin.cpp)
target_link_libraries(set_speed_joint_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(set_trackfriction_plugin src/set_trackfriction_plugin.cpp)
target_link_libraries(set_trackfriction_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

catkin_package(
  DEPENDS 
    roscpp 
    gazebo_ros 
)
