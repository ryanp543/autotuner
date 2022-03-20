import sys
import csv
import numpy as np
import scipy.optimize
import math
from sympy import symbols, Eq, solve

import klampt
from SuspensionMatrices import Suspension_8legs
"""
# Example use

"""


class AdjustedJacobian():
    # A simple class to define the stiffness (K) and damping (B) matrices of a vehicle suspension with 8 legs.
    # Uses Taylor expansions for terms where the state vector does not appear linearly.
    def __init__(self, lcx=0.190, lcy=0.033, lcz=0.196,
                 l1=0.0545, l2=0.07175, l3=0.352, l4=0.32):
        # -----------------------
        # Default Parameters
        # -----------------------
        #
        # lcx           Distance from origin to chassis COM x
        # lcz           Distance from origin to chassis COM z
        # l1            Length of link 1
        # l2            Length of link 2
        # l3            Length of link 3
        # l4            Length of link 4
        #

        self.lcx = lcx
        self.lcy = lcy
        self.lcz = lcz
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4


    def cos(self, angle):
        return math.cos(angle)


    def sin(self, angle):
        return math.sin(angle)


    def getForwardKinematics(self, pos):
        # full state: (x, y, z, qyaw, qroll, qpitch, q1, q2, q3, q4)
        lcx = self.lcx
        lcy = self.lcy
        lcz = self.lcz
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        l4 = self.l4
        qz = pos[2]
        qr = pos[4]
        qp = pos[5]
        q1 = pos[6]
        q2 = pos[7]
        q3 = pos[8]
        sin = self.sin
        cos = self.cos


        x_arm = -l4*(cos(q1)*cos(q2)*sin(q3)+cos(q1)*cos(q3)*sin(q2)) - l3*cos(q1)
        y_arm = -l2 - l4*(cos(q2)*sin(q1)*sin(q3)+cos(q3)*sin(q1)*sin(q2)) - l3*sin(q1)
        z_arm = l1 + l4*(cos(q2)*cos(q3)-sin(q2)*sin(q3))

        x = sin(qp)*z_arm + cos(qp)*x_arm + lcx*cos(qp) + lcz*sin(qp)
        y = lcx*sin(qp)*sin(qr) + sin(qp)*sin(qr)*x_arm - cos(qp)*sin(qr)*z_arm + cos(qr)*y_arm - lcy*cos(qr)
        z = -cos(qr)*sin(qp)*x_arm + sin(qr)*y_arm + cos(qp)*cos(qr)*z_arm + lcz*cos(qp)*cos(qr) - lcx*cos(qr)*sin(qp) \
            - lcy*sin(qr)
        print x, y, z



if __name__ == "__main__":
    # Intializations suspension and robot
    from SuspensionMatrices import Suspension_8legs
    sus = Suspension_8legs()  # 3DOF model of the suspension, use with GetStiffnessMatrix and GetDampingMatrix
    world = klampt.WorldModel()
    res = world.readFile("./robot_sim.xml")
    robot = world.robot(0)

    link = robot.link(7)
    # print link.getWorldPosition([0,0,0])

    # joint 1: -3.1415 to 3.1415
    # joint 2: -3.1415 to 0
    # joint 3: -1.0472 to 2.6
    # joint 4: -1.571 to 1.571


    # full state: (x, y, z, qyaw, qroll, qpitch, q1, q2, q3, q4)
    x_init = [0] * robot.numLinks()
    dx_init = [0.1] * robot.numLinks()
    states = x_init + dx_init

    n = -math.pi/4
    x_init = [0, 0, 0, 0, 0, 0, 0, n, n, n]

    robot.setConfig(x_init)
    link4 = robot.link(robot.numLinks()-1)
    J = link4.getPositionJacobian([0,0,0])

    print link4.getWorldPosition([0,0,0])


    # test
    adj_jac = AdjustedJacobian()
    adj_jac.getForwardKinematics(x_init)

