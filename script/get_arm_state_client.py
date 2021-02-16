#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
import gazebo_msgs
from gazebo_msgs.msg import LinkState 
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import GetLinkState
from tf import transformations as tf_transform
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped


def publish_arm_state():
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

        # Publising arm link poses and twist
        ii = 0
        for pub in arm_pose_pub_dic:
            ii += 1 
            if ii == 1:
                response = get_link_state('Link_1','link_chassis')
            else:
                response = get_link_state('Link_'+str(ii),'Link_'+str(ii-1))
            link_pose = response.link_state.pose

            stamped_link_pose = PoseStamped()
            stamped_link_pose.pose = response.link_state.pose
            stamped_link_pose.header.stamp = rospy.Time.now() # Add a time stamp to the message
            pub.publish(stamped_link_pose)

        # Publish arm link twists
        ii = 0
        for pub in arm_twist_pub_dic:
            ii += 1 
            if ii == 1:
                response = get_link_state('Link_1','link_chassis')
            else:
                response = get_link_state('Link_'+str(ii),'Link_'+str(ii-1))

            stamped_link_twist = TwistStamped()
            stamped_link_twist.twist = response.link_state.twist
            stamped_link_twist.header.stamp = rospy.Time.now() # Add a time stamp to the message
            pub.publish(stamped_link_twist)

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    rospy.init_node('get_track_state_client')
    rate = rospy.Rate(30)    

    # Define the ROS publishers and put them in a dictionnary
    link1_pose_pub = rospy.Publisher("arm_state/link_1/pose",PoseStamped,queue_size=100)
    link2_pose_pub = rospy.Publisher("arm_state/link_2/pose",PoseStamped,queue_size=100)
    link3_pose_pub = rospy.Publisher("arm_state/link_3/pose",PoseStamped,queue_size=100)
    link4_pose_pub = rospy.Publisher("arm_state/link_4/pose",PoseStamped,queue_size=100)
    arm_pose_pub_dic = [link1_pose_pub,link2_pose_pub,link3_pose_pub,link4_pose_pub]

    link1_twist_pub = rospy.Publisher("arm_state/link_1/twist",TwistStamped,queue_size=100)
    link2_twist_pub = rospy.Publisher("arm_state/link_2/twist",TwistStamped,queue_size=100)
    link3_twist_pub = rospy.Publisher("arm_state/link_3/twist",TwistStamped,queue_size=100)
    link4_twist_pub = rospy.Publisher("arm_state/link_4/twist",TwistStamped,queue_size=100)
    arm_twist_pub_dic = [link1_twist_pub,link2_twist_pub,link3_twist_pub,link4_twist_pub]

    while not rospy.is_shutdown():
        publish_arm_state()
        rate.sleep()