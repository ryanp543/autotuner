#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
import gazebo_msgs
from gazebo_msgs.msg import LinkState 
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import GetLinkState
from motor_drive.msg import MotorPair_Float
from tf import transformations as tf_transform
from std_msgs.msg import Float32

old_lefttrack_position = 0
old_righttrack_position = 0
old_lefttrack_speed = 0
old_righttrack_speed = 0
old_timestamp = 0
initial_wheel_state = LinkState()

def publish_wheel_orientation_speeds():
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

        # Publising wheel angular speeds
        ii = 0
        for pub in wheel_speed_pub_dic:
            ii += 1 
            response = get_link_state('link_wheel'+str(ii),'link_leg'+str(ii))
            wheel_speed = response.link_state.twist.angular.y
            pub.publish(wheel_speed)

        # Publishing wheel orientations
        ii = 0
        for pub in wheel_orient_pub_dic:
            ii += 1 
            response = get_link_state('link_wheel'+str(ii),'link_leg'+str(ii))
            wheel_orientation = response.link_state.pose.orientation
            angles = tf_transform.euler_from_quaternion([wheel_orientation.x, wheel_orientation.y, wheel_orientation.z, wheel_orientation.w], axes='syxz')
            isRollInverted = abs(angles[1]-3.1416) < 0.01
            isYawInverted = abs(angles[2]-3.1416) < 0.01
            if isRollInverted ^ isYawInverted:
                pitch = -angles[0]
            else:
                pitch = angles[0]

            pub.publish(pitch)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def on_new_command(msg):
    lefttrack_speed = msg.left
    righttrack_speed = msg.right
    set_track_speeds(lefttrack_speed,righttrack_speed)
    publish_wheel_orientation_speeds()

def set_track_speeds(lefttrack_speed,righttrack_speed):
    global old_lefttrack_speed
    global old_righttrack_speed
    global old_lefttrack_position
    global old_righttrack_position
    global old_timestamp

    # Compute deltas and new track positions
    new_timestamp = rospy.get_time()
    delta_t = new_timestamp - old_timestamp
    #print("Delta time [ms]: " + "{:.0f}".format(1000*delta_t))
    lefttrack_position = old_lefttrack_position + 0.5*(old_lefttrack_speed+lefttrack_speed)*delta_t
    righttrack_position = old_righttrack_position + 0.5*(old_righttrack_speed+righttrack_speed)*delta_t
    #print("Left track position [m]: "+"{:.4f}".format(lefttrack_position))

    # Iterate
    old_timestamp = new_timestamp
    old_lefttrack_position = lefttrack_position
    old_righttrack_position = righttrack_position
    old_lefttrack_speed = lefttrack_speed
    old_righttrack_speed = righttrack_speed

    # Setting the new state of the wheels
    leftwheels_newlinkstate = LinkState()
    rightwheels_newlinkstate = LinkState()

    leftwheels_newlinkstate.pose.position = initial_wheel_state.pose.position
    rightwheels_newlinkstate.pose.position = initial_wheel_state.pose.position

    leftwheels_quaternion = tf_transform.quaternion_from_euler(lefttrack_position/0.030, 0 , 0, axes='syxz')
    leftwheels_newlinkstate.pose.orientation.x = leftwheels_quaternion[0]
    leftwheels_newlinkstate.pose.orientation.y = leftwheels_quaternion[1]
    leftwheels_newlinkstate.pose.orientation.z = leftwheels_quaternion[2]
    leftwheels_newlinkstate.pose.orientation.w = leftwheels_quaternion[3]

    rightwheels_quaternion = tf_transform.quaternion_from_euler(righttrack_position/0.030, 0 , 0, axes='syxz')
    rightwheels_newlinkstate.pose.orientation.x = rightwheels_quaternion[0]
    rightwheels_newlinkstate.pose.orientation.y = rightwheels_quaternion[1]
    rightwheels_newlinkstate.pose.orientation.z = rightwheels_quaternion[2]
    rightwheels_newlinkstate.pose.orientation.w = rightwheels_quaternion[3]

    leftwheels_newlinkstate.twist.angular.y = lefttrack_speed/0.030
    rightwheels_newlinkstate.twist.angular.y  = righttrack_speed/0.030
    #TODO replace wheel radius with ros parameter

    
    try:
        
        #print("You got the service handle !")

        # Repeat for each of the 8 legs
        for ii in range(4):
            leftwheels_newlinkstate.link_name = "link_wheel"+str(ii+1)
            leftwheels_newlinkstate.reference_frame = "link_leg"+str(ii+1)
            rightwheels_newlinkstate.link_name ="link_wheel"+str(ii+5)
            rightwheels_newlinkstate.reference_frame = "link_leg"+str(ii+5)

            rospy.wait_for_service('/gazebo/set_link_state')
            set_link_state = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
            leftresponse = set_link_state(leftwheels_newlinkstate)

            rospy.wait_for_service('/gazebo/set_link_state')
            set_link_state = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
            rightresponse = set_link_state(rightwheels_newlinkstate)
        
        #print("Last left wheel response: "+leftresponse.status_message)
        return True

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def get_initial_state():
    # Get the initial state of one wheel in it's corresponding leg ref. frame. 
    # This will be used to set the proper position for the wheel at each time step.
    global initial_wheel_state 
    initial_wheel_state = LinkState()
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
         get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
         response = get_link_state('link_wheel1','link_leg1')
         initial_wheel_state = response.link_state
         #print("I got the initial link state request: "+"{:.4f}".format(initial_wheel_state.pose.position.x))
         return True
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def kill_on_shutdown():
    set_track_speeds(0,0)


if __name__ == "__main__":
    rospy.init_node('set_track_speeds_client')    
    rospy.Subscriber("motors/speed_cmd",MotorPair_Float,on_new_command)
    wheel1_pos_pub = rospy.Publisher("wheel_orientations/wheel_1",Float32,queue_size=100)
    wheel2_pos_pub = rospy.Publisher("wheel_orientations/wheel_2",Float32,queue_size=100)
    wheel3_pos_pub = rospy.Publisher("wheel_orientations/wheel_3",Float32,queue_size=100)
    wheel4_pos_pub = rospy.Publisher("wheel_orientations/wheel_4",Float32,queue_size=100)
    wheel5_pos_pub = rospy.Publisher("wheel_orientations/wheel_5",Float32,queue_size=100)
    wheel6_pos_pub = rospy.Publisher("wheel_orientations/wheel_6",Float32,queue_size=100)
    wheel7_pos_pub = rospy.Publisher("wheel_orientations/wheel_7",Float32,queue_size=100)
    wheel8_pos_pub = rospy.Publisher("wheel_orientations/wheel_8",Float32,queue_size=100)
    wheel_orient_pub_dic = [wheel1_pos_pub,wheel2_pos_pub,wheel3_pos_pub,wheel4_pos_pub,wheel5_pos_pub,wheel6_pos_pub,wheel7_pos_pub,wheel8_pos_pub]

    wheel1_pub = rospy.Publisher("wheel_speeds/wheel_1",Float32,queue_size=100)
    wheel2_pub = rospy.Publisher("wheel_speeds/wheel_2",Float32,queue_size=100)
    wheel3_pub = rospy.Publisher("wheel_speeds/wheel_3",Float32,queue_size=100)
    wheel4_pub = rospy.Publisher("wheel_speeds/wheel_4",Float32,queue_size=100)
    wheel5_pub = rospy.Publisher("wheel_speeds/wheel_5",Float32,queue_size=100)
    wheel6_pub = rospy.Publisher("wheel_speeds/wheel_6",Float32,queue_size=100)
    wheel7_pub = rospy.Publisher("wheel_speeds/wheel_7",Float32,queue_size=100)
    wheel8_pub = rospy.Publisher("wheel_speeds/wheel_8",Float32,queue_size=100)
    wheel_speed_pub_dic = [wheel1_pub,wheel2_pub,wheel3_pub,wheel4_pub,wheel5_pub,wheel6_pub,wheel7_pub,wheel8_pub]
    rospy.on_shutdown(kill_on_shutdown)
    get_initial_state()
    old_timestamp = rospy.get_time()
    rospy.spin()