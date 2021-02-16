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


if __name__ == "__main__":
    rospy.init_node('get_track_state_client')
    rate = rospy.Rate(100)    

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


    while not rospy.is_shutdown():
        publish_wheel_orientation_speeds()
        rate.sleep()