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


    def cos(self, ang):
        # cos_ts = math.cos(ang)
        cos_ts = 1 - (ang**2)/math.factorial(2) + (ang**4)/math.factorial(4) - (ang**6)/math.factorial(6) \
                 + (ang**8)/math.factorial(8)
        return cos_ts


    def sin(self, ang):
        # sin_ts = math.sin(ang)
        sin_ts = ang - (ang**3)/math.factorial(3) + (ang**5)/math.factorial(5) - (ang**7)/math.factorial(7) \
                 + (ang**9)/math.factorial(9)
        return sin_ts


    def getForwardKinematics(self, pos):
        # full state: (x, y, z, qyaw, qpitch, qroll, q1, q2, q3, q4)
        lcx = self.lcx
        lcy = self.lcy
        lcz = self.lcz
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        l4 = self.l4
        qz = pos[2]
        qp = pos[4]
        qr = pos[5]
        q1 = pos[6]
        q2 = pos[7]
        q3 = pos[8]
        sin = self.sin
        cos = self.cos


        x_arm = l2*sin(q1) - l3*cos(q1)*cos(q2) - l4*(cos(q1)*cos(q2)*sin(q3)+cos(q1)*cos(q3)*sin(q2))
        y_arm = -l2*cos(q1) - l3*cos(q2)*sin(q1) - l4*(cos(q2)*sin(q1)*sin(q3)+cos(q3)*sin(q1)*sin(q2))
        z_arm = l1 - l3*sin(q2) + l4*(cos(q2)*cos(q3)-sin(q2)*sin(q3))

        x = cos(qp)*(lcx+x_arm) + cos(qr)*sin(qp)*(lcz+z_arm) - sin(qp)*sin(qr)*(lcy-y_arm)
        y = -sin(qr)*(lcz+z_arm) - cos(qr)*(lcy-y_arm)
        z = qz + cos(qp)*cos(qr)*(lcz+z_arm) - sin(qp)*(lcx+x_arm) - cos(qp)*sin(qr)*(lcy-y_arm)
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


    # full state: (x, y, z, qyaw, qpitch, qroll, q1, q2, q3, q4)
    x_init = [0] * robot.numLinks()
    dx_init = [0.1] * robot.numLinks()
    states = x_init + dx_init

    n = -math.pi/4
    x_init = [0, 0, 0.3, 0, n, n, n, n, n, n]

    robot.setConfig(x_init)
    link4 = robot.link(robot.numLinks()-1)
    J = link4.getPositionJacobian([0,0,0])

    print link4.getWorldPosition([0,0,0])


    # test
    adj_jac = AdjustedJacobian()
    adj_jac.getForwardKinematics(x_init)

