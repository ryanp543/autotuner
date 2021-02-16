import numpy as np
from numpy import sin, cos

"""
# Example use
from numpy import linalg

sus = Suspension_8legs()
K = sus.GetStiffnessMatrix()
B = sus.GetDampingMatrix()
# To get the eigen values:
eigvalues,eigenvectors = linalg.eig(K)
lambda_min = min(eigvalues)
"""


class Suspension_8legs():
    # A simple class to define the stiffness (K) and damping (B) matrices of a vehicle suspension with 8 legs.
    # Uses Taylor expansions for terms where the state vector does not appear linearly.
    def __init__(self, g=9.80665, L=0.4,
                 B=0.657, Lf=0.155,
                 H=0.01, k_spring_lin=767, # k = 1560.38, 767, 860, H=0.076
                 Ln_lin=0.0685, b_damper=230): # 230, 0.0685
        # -----------------------
        # Default Parameters
        # -----------------------
        #
        # g              Earth's gravitational acceleration in meters/second^2
        # L              Length of the track on the ground in meters
        # B              Tread of the vehicle in meters
        # Lf             Horizontal distance from COM to front spring attachment point in meters
        # H              Height of the linear spring attachment point with respect to the COM in meters
        # k_spring_lin   Spring stiffness constant in newton/meter
        # Ln_lin         Natural lenght of the  springs in meters
        # b_damper       Equivalent linear damper constant of each suspension leg in newtons*second/meter
        # n              Number of legs per side
        #

        self.g = g
        self.L = L
        self.B = B
        self.Lf = Lf
        self.H = H
        self.k_spring_lin = k_spring_lin
        self.Ln_lin = Ln_lin
        self.b_damper = b_damper
        self.n = 4  # Number of legs per side (8 total)

        # print("Suspension parameters initialized.")

    def GetStiffnessMatrix(self, z=0, qPitch=0, qRoll=0):
        g = self.g
        L = self.L
        B = self.B
        Lf = self.Lf
        H = self.H
        k_spring_lin = self.k_spring_lin
        Ln_lin = self.Ln_lin
        b_damper = self.b_damper
        n = self.n

        k_z_z = 8 * k_spring_lin
        k_z_qRoll = 0.006944444 * H * k_spring_lin * qRoll * (
                    576 + 12 * qPitch ** 4 + 12 * qPitch ** 2 * qRoll ** 2 - 48 * qRoll ** 2 - qPitch ** 2 * (
                        144 + qPitch ** 2 * qRoll ** 2))
        k_z_qPitch = 0.006944444 * k_spring_lin * (
                    576 * H * qPitch + 192 * Lf * qPitch ** 2 + 12 * H * qPitch * qRoll ** 4 + 12 * H * qPitch ** 3 * qRoll ** 2 - 1152 * Lf - 48 * H * qPitch ** 3 - 9.6 * Lf * qPitch ** 4 - H * qPitch * qRoll ** 2 * (
                        144 + qPitch ** 2 * qRoll ** 2) - 14.4 * L * (-120 + 20 * qPitch ** 2 - qPitch ** 4) / (-1 + n))
        k_qRoll_z = 8 * H * k_spring_lin * sin(qRoll)
        k_qRoll_qRoll = 1.446759E-05 * k_spring_lin * (
                    138240 * B ** 2 + 48 * B ** 2 * qRoll ** 8 + 2880 * B ** 2 * qPitch ** 4 + 6144 * H ** 2 * qRoll ** 6 + 18432 * B ** 2 * qRoll ** 4 + 46080 * H * Lf * qPitch ** 3 + 92160 * H * Ln_lin * qRoll ** 2 + 138240 * H ** 2 * qPitch ** 2 + 368640 * H ** 2 * qRoll ** 2 + 46080 * H * Lf * qPitch * qRoll ** 2 + B ** 2 * qPitch ** 4 * qRoll ** 8 + 48 * H ** 2 * qPitch ** 2 * qRoll ** 8 + 128 * H ** 2 * qPitch ** 4 * qRoll ** 6 + 384 * B ** 2 * qPitch ** 2 * qRoll ** 6 + 384 * B ** 2 * qPitch ** 4 * qRoll ** 4 + 384 * H * Lf * qPitch ** 3 * qRoll ** 4 + 384 * H * Lf * qPitch ** 5 * qRoll ** 2 + 7680 * H ** 2 * qPitch ** 4 * qRoll ** 2 + 18432 * H ** 2 * qPitch ** 2 * qRoll ** 4 + 23040 * B ** 2 * qPitch ** 2 * qRoll ** 2 - 552960 * H * Ln_lin - 552960 * H ** 2 - 276480 * H * Lf * qPitch - 92160 * B ** 2 * qRoll ** 2 - 73728 * H ** 2 * qRoll ** 4 - 34560 * B ** 2 * qPitch ** 2 - 11520 * H ** 2 * qPitch ** 4 - 4608 * H * Ln_lin * qRoll ** 4 - 2304 * H * Lf * qPitch ** 5 - 1536 * B ** 2 * qRoll ** 6 - 192 * H ** 2 * qRoll ** 8 - 2304 * H * Lf * qPitch * qRoll ** 4 - 92160 * H ** 2 * qPitch ** 2 * qRoll ** 2 - 7680 * H * Lf * qPitch ** 3 * qRoll ** 2 - 4608 * B ** 2 * qPitch ** 2 * qRoll ** 4 - 1920 * B ** 2 * qPitch ** 4 * qRoll ** 2 - 1536 * H ** 2 * qPitch ** 2 * qRoll ** 6 - 1536 * H ** 2 * qPitch ** 4 * qRoll ** 4 - 32 * B ** 2 * qPitch ** 4 * qRoll ** 6 - 19.2 * H * Lf * qPitch ** 5 * qRoll ** 4 - 12 * B ** 2 * qPitch ** 2 * qRoll ** 8 - 4 * H ** 2 * qPitch ** 4 * qRoll ** 8 - 28.8 * H * L * qPitch * (
                        -14400 + 2400 * qPitch ** 2 + 2400 * qRoll ** 2 + 20 * qPitch ** 2 * qRoll ** 4 + 20 * qPitch ** 4 * qRoll ** 2 - 120 * qPitch ** 4 - 120 * qRoll ** 4 - 400 * qPitch ** 2 * qRoll ** 2 - qPitch ** 4 * qRoll ** 4) / (
                                -1 + n))
        k_qRoll_qPitch = 1.446759E-05 * k_spring_lin * qRoll * (
                    138240 * H ** 2 * qPitch + 2880 * B ** 2 * qPitch ** 3 + 46080 * H * Lf * qPitch ** 2 + 46080 * H * Lf * qRoll ** 2 + 48 * H ** 2 * qPitch * qRoll ** 8 + 384 * B ** 2 * qPitch * qRoll ** 6 + 18432 * H ** 2 * qPitch * qRoll ** 4 + 23040 * B ** 2 * qPitch * qRoll ** 2 + B ** 2 * qPitch ** 3 * qRoll ** 8 + 128 * H ** 2 * qPitch ** 3 * qRoll ** 6 + 384 * B ** 2 * qPitch ** 3 * qRoll ** 4 + 384 * H * Lf * qPitch ** 2 * qRoll ** 4 + 384 * H * Lf * qPitch ** 4 * qRoll ** 2 + 7680 * H ** 2 * qPitch ** 3 * qRoll ** 2 - 276480 * H * Lf - 34560 * B ** 2 * qPitch - 11520 * H ** 2 * qPitch ** 3 - 2304 * H * Lf * qPitch ** 4 - 2304 * H * Lf * qRoll ** 4 - 92160 * H ** 2 * qPitch * qRoll ** 2 - 4608 * B ** 2 * qPitch * qRoll ** 4 - 1536 * H ** 2 * qPitch * qRoll ** 6 - 12 * B ** 2 * qPitch * qRoll ** 8 - 7680 * H * Lf * qPitch ** 2 * qRoll ** 2 - 1920 * B ** 2 * qPitch ** 3 * qRoll ** 2 - 1536 * H ** 2 * qPitch ** 3 * qRoll ** 4 - 32 * B ** 2 * qPitch ** 3 * qRoll ** 6 - 19.2 * H * Lf * qPitch ** 4 * qRoll ** 4 - 4 * H ** 2 * qPitch ** 3 * qRoll ** 8 - 28.8 * H * L * (
                        -14400 + 2400 * qPitch ** 2 + 2400 * qRoll ** 2 + 20 * qPitch ** 2 * qRoll ** 4 + 20 * qPitch ** 4 * qRoll ** 2 - 120 * qPitch ** 4 - 120 * qRoll ** 4 - 400 * qPitch ** 2 * qRoll ** 2 - qPitch ** 4 * qRoll ** 4) / (
                                -1 + n))
        k_qPitch_z = -8 * k_spring_lin * (
                    Lf * cos(qPitch) - H * sin(qPitch) * cos(qRoll) - 1.5 * L * cos(qPitch) / (-1 + n))
        k_qPitch_qRoll = k_spring_lin * qRoll * (
                    2 * H * Ln_lin * qPitch + 4 * H ** 2 * qPitch + 0.0002777778 * H * Lf * qPitch ** 10 + 0.001388889 * H ** 2 * qPitch ** 9 + 0.01111111 * B ** 2 * qPitch ** 7 + 0.01666667 * H * Ln_lin * qPitch ** 5 + 0.1777778 * H * Lf * qPitch ** 6 + 0.3333333 * H * Lf * qRoll ** 2 + 0.5333333 * H ** 2 * qPitch ** 5 + 0.6666667 * B ** 2 * qPitch ** 3 + 4 * H * Lf * qPitch ** 2 + 0.0002777778 * H ** 2 * qPitch * qRoll ** 8 + 0.002777778 * B ** 2 * qPitch * qRoll ** 6 + 0.1777778 * H ** 2 * qPitch * qRoll ** 4 + 9.645062E-08 * H ** 2 * qPitch ** 9 * qRoll ** 8 + 7.716049E-07 * B ** 2 * qPitch ** 7 * qRoll ** 8 + 9.645062E-07 * B ** 2 * qPitch ** 9 * qRoll ** 6 + 3.703704E-05 * H ** 2 * qPitch ** 5 * qRoll ** 8 + 4.62963E-05 * B ** 2 * qPitch ** 3 * qRoll ** 8 + 6.17284E-05 * H ** 2 * qPitch ** 9 * qRoll ** 4 + 0.0001234568 * H ** 2 * qPitch ** 7 * qRoll ** 6 + 0.0003703704 * B ** 2 * qPitch ** 5 * qRoll ** 6 + 0.0004938272 * B ** 2 * qPitch ** 7 * qRoll ** 4 + 0.0009259259 * H * Lf * qPitch ** 8 * qRoll ** 2 + 0.007407407 * H ** 2 * qPitch ** 3 * qRoll ** 6 + 0.0237037 * H ** 2 * qPitch ** 5 * qRoll ** 4 + 0.02777778 * H * Ln_lin * qPitch ** 3 * qRoll ** 2 + 0.02962963 * B ** 2 * qPitch ** 3 * qRoll ** 4 + 0.1111111 * H * Lf * qPitch ** 4 * qRoll ** 2 + 3.472222E-05 * H * L * (
                        -172800 + 172800 * n + 12 * qPitch ** 10 + 7680 * qPitch ** 6 + 480 * n * qPitch ** 8 + 4800 * qPitch ** 4 * qRoll ** 2 + 640 * n * qPitch ** 6 * qRoll ** 2 - 57600 * qPitch ** 4 - 480 * qPitch ** 8 - 12 * n * qPitch ** 10 - qPitch ** 10 * qRoll ** 2 - 14400 * (
                            -1 + n) * qRoll ** 2 - 40 * n * qPitch ** 8 * qRoll ** 2) / (
                                -1 + n) ** 2 - 4 * H * Lf - B ** 2 * qPitch - 2.666667 * H ** 2 * qPitch ** 3 - 1.333333 * H * Lf * qPitch ** 4 - 0.3333333 * H * Ln_lin * qPitch ** 3 - 0.1333333 * B ** 2 * qPitch ** 5 - 0.04444444 * H ** 2 * qPitch ** 7 - 0.01111111 * H * Lf * qPitch ** 8 - 0.0003472222 * B ** 2 * qPitch ** 9 - 1.333333 * H ** 2 * qPitch * qRoll ** 2 - 0.1666667 * H * Ln_lin * qPitch * qRoll ** 2 - 0.01111111 * H ** 2 * qPitch * qRoll ** 6 - 6.944444E-05 * B ** 2 * qPitch * qRoll ** 8 - 0.3333333 * H * Lf * qPitch ** 2 * qRoll ** 2 - 0.2222222 * B ** 2 * qPitch ** 3 * qRoll ** 2 - 0.1777778 * H ** 2 * qPitch ** 5 * qRoll ** 2 - 0.01481481 * H * Lf * qPitch ** 6 * qRoll ** 2 - 0.003703704 * B ** 2 * qPitch ** 7 * qRoll ** 2 - 0.001851852 * B ** 2 * qPitch ** 3 * qRoll ** 6 - 0.001481481 * H ** 2 * qPitch ** 5 * qRoll ** 6 - 0.001388889 * H * Ln_lin * qPitch ** 5 * qRoll ** 2 - 0.000462963 * H ** 2 * qPitch ** 9 * qRoll ** 2 - 0.0001851852 * H ** 2 * qPitch ** 3 * qRoll ** 8 - 3.08642E-05 * B ** 2 * qPitch ** 7 * qRoll ** 6 - 2.314815E-05 * H * Lf * qPitch ** 10 * qRoll ** 2 - 9.259259E-06 * B ** 2 * qPitch ** 5 * qRoll ** 8 - 3.858025E-06 * H ** 2 * qPitch ** 9 * qRoll ** 6 - 3.08642E-06 * H ** 2 * qPitch ** 7 * qRoll ** 8 - 2.411265E-08 * B ** 2 * qPitch ** 9 * qRoll ** 8 - 0.1185185 * H ** 2 * qPitch ** 3 * qRoll ** 2 * (
                                -7.5 + qRoll ** 2) - 0.005925926 * B ** 2 * qPitch ** 5 * qRoll ** 2 * (
                                -7.5 + qRoll ** 2) - 0.001975309 * H ** 2 * qPitch ** 7 * qRoll ** 2 * (
                                -7.5 + qRoll ** 2) - 1.54321E-05 * B ** 2 * qPitch ** 9 * qRoll ** 2 * (
                                -7.5 + qRoll ** 2) - 3.472222E-05 * qPitch * (
                                1280 * B ** 2 * qRoll ** 2 * (-7.5 + qRoll ** 2) + H * L * qPitch * (
                                    -172800 + 172800 * n + 14400 * qRoll ** 2 + 7680 * n * qPitch ** 4 + 640 * qPitch ** 4 * qRoll ** 2 + 4800 * n * qPitch ** 2 * qRoll ** 2 - 57600 * n * qPitch ** 2 - 14400 * n * qRoll ** 2 - 40 * qPitch ** 6 * qRoll ** 2 - n * qPitch ** 8 * qRoll ** 2) / (
                                            -1 + n) ** 2))
        k_qPitch_qPitch = -k_spring_lin * (
                    8 * H * Ln_lin + 8 * H ** 2 + 4 * Lf * Ln_lin * qPitch + 16 * H * Lf * qPitch + B ** 2 * qRoll ** 2 + 6.944444E-05 * B ** 2 * qRoll ** 10 + 0.001111111 * H * Lf * qPitch ** 9 + 0.002777778 * H ** 2 * qPitch ** 8 + 0.01111111 * H ** 2 * qRoll ** 8 + 0.06666667 * H * Ln_lin * qPitch ** 4 + 0.08888889 * Lf ** 2 * qPitch ** 6 + 0.1666667 * H * Ln_lin * qRoll ** 4 + 0.7111111 * H * Lf * qPitch ** 5 + 1.066667 * H ** 2 * qPitch ** 4 + 1.333333 * H ** 2 * qRoll ** 4 + 5.333333 * Lf ** 2 * qPitch ** 2 + 0.3333333 * H * Lf * qPitch * qRoll ** 4 + 2.411265E-08 * B ** 2 * qPitch ** 8 * qRoll ** 10 + 3.08642E-06 * H ** 2 * qPitch ** 6 * qRoll ** 10 + 3.858025E-06 * H ** 2 * qPitch ** 8 * qRoll ** 8 + 9.259259E-06 * B ** 2 * qPitch ** 4 * qRoll ** 10 + 2.314815E-05 * H * Lf * qPitch ** 9 * qRoll ** 4 + 3.08642E-05 * B ** 2 * qPitch ** 6 * qRoll ** 8 + 0.0001851852 * H ** 2 * qPitch ** 2 * qRoll ** 10 + 0.0003472222 * B ** 2 * qPitch ** 8 * qRoll ** 2 + 0.000462963 * H ** 2 * qPitch ** 8 * qRoll ** 4 + 0.001388889 * H * Ln_lin * qPitch ** 4 * qRoll ** 4 + 0.001481481 * H ** 2 * qPitch ** 4 * qRoll ** 8 + 0.001851852 * B ** 2 * qPitch ** 2 * qRoll ** 8 + 0.003703704 * B ** 2 * qPitch ** 6 * qRoll ** 4 + 0.01111111 * H * Lf * qPitch ** 7 * qRoll ** 2 + 0.01481481 * H * Lf * qPitch ** 5 * qRoll ** 4 + 0.04444444 * H ** 2 * qPitch ** 6 * qRoll ** 2 + 0.1333333 * B ** 2 * qPitch ** 4 * qRoll ** 2 + 0.1777778 * H ** 2 * qPitch ** 4 * qRoll ** 4 + 0.2222222 * B ** 2 * qPitch ** 2 * qRoll ** 4 + 0.3333333 * H * Ln_lin * qPitch ** 2 * qRoll ** 2 + 1.333333 * H * Lf * qPitch ** 3 * qRoll ** 2 + 2.666667 * H ** 2 * qPitch ** 2 * qRoll ** 2 + 0.04444444 * B ** 2 * qRoll ** 4 * (
                        -7.5 + qRoll ** 2) + 1.54321E-05 * B ** 2 * qPitch ** 8 * qRoll ** 4 * (
                                -7.5 + qRoll ** 2) + 0.001975309 * H ** 2 * qPitch ** 6 * qRoll ** 4 * (
                                -7.5 + qRoll ** 2) + 0.005925926 * B ** 2 * qPitch ** 4 * qRoll ** 4 * (
                                -7.5 + qRoll ** 2) + 0.1185185 * H ** 2 * qPitch ** 2 * qRoll ** 4 * (
                                -7.5 + qRoll ** 2) - 8 * Lf ** 2 - 5.333333 * H * Lf * qPitch ** 3 - 5.333333 * H ** 2 * qPitch ** 2 - 4 * H ** 2 * qRoll ** 2 - 2 * H * Ln_lin * qRoll ** 2 - 1.333333 * H * Ln_lin * qPitch ** 2 - 1.066667 * Lf ** 2 * qPitch ** 4 - 0.3333333 * Lf * Ln_lin * qPitch ** 3 - 0.1777778 * H ** 2 * qRoll ** 6 - 0.08888889 * H ** 2 * qPitch ** 6 - 0.04444444 * H * Lf * qPitch ** 7 - 0.002777778 * B ** 2 * qRoll ** 8 - 0.002777778 * Lf ** 2 * qPitch ** 8 - 0.0002777778 * H ** 2 * qRoll ** 10 - 4 * H * Lf * qPitch * qRoll ** 2 - 0.6666667 * B ** 2 * qPitch ** 2 * qRoll ** 2 - 0.5333333 * H ** 2 * qPitch ** 4 * qRoll ** 2 - 0.1777778 * H * Lf * qPitch ** 5 * qRoll ** 2 - 0.1111111 * H * Lf * qPitch
 ** 3 * qRoll ** 4 - 0.02962963 * B ** 2 * qPitch ** 2 * qRoll ** 6 - 0.02777778 * H * Ln_lin * qPitch ** 2 * qRoll ** 4 - 0.0237037 * H ** 2 * qPitch ** 4 * qRoll ** 6 - 0.01666667 * H * Ln_lin * qPitch ** 4 * qRoll ** 2 - 0.01111111 * B ** 2 * qPitch ** 6 * qRoll ** 2 - 0.007407407 * H ** 2 * qPitch ** 2 * qRoll ** 8 - 0.001388889 * H ** 2 * qPitch ** 8 * qRoll ** 2 - 0.0009259259 * H * Lf * qPitch ** 7 * qRoll ** 4 - 0.0004938272 * B ** 2 * qPitch ** 6 * qRoll ** 6 - 0.0003703704 * B ** 2 * qPitch ** 4 * qRoll ** 8 - 0.0002777778 * H * Lf * qPitch ** 9 * qRoll ** 2 - 0.0001234568 * H ** 2 * qPitch ** 6 * qRoll ** 8 - 6.17284E-05 * H ** 2 * qPitch ** 8 * qRoll ** 6 - 4.62963E-05 * B ** 2 * qPitch ** 2 * qRoll ** 10 - 3.703704E-05 * H ** 2 * qPitch ** 4 * qRoll ** 10 - 9.645062E-07 * B ** 2 * qPitch ** 8 * qRoll ** 8 - 7.716049E-07 * B ** 2 * qPitch ** 6 * qRoll ** 10 - 9.645062E-08 * H ** 2 * qPitch ** 8 * qRoll ** 10 - 3.472222E-05 * L * (
                                691200 * Lf + 806400.2 * L + 240 * Lf * qPitch ** 8 + 280.0001 * L * qPitch ** 8 + 1920 * H * qPitch ** 7 + 92160 * Lf * qPitch ** 4 + 107520 * L * qPitch ** 4 + 230400 * H * qPitch ** 3 + 691200 * H * n * qPitch + 48 * H * n * qPitch ** 9 + 7680 * Lf * n * qPitch ** 6 + 30720 * H * n * qPitch ** 5 + 172800 * H * qPitch * qRoll ** 2 + 460800 * Lf * n * qPitch ** 2 + 12 * H * qPitch ** 9 * qRoll ** 2 + 40 * H * qPitch ** 7 * qRoll ** 4 + 4800 * H * qPitch ** 3 * qRoll ** 4 + 7680 * H * qPitch ** 5 * qRoll ** 2 + 14400 * H * n * qPitch * qRoll ** 4 + 172800 * Ln_lin * (
                                    -1 + n) * qPitch + H * n * qPitch ** 9 * qRoll ** 4 + 480 * H * n * qPitch ** 7 * qRoll ** 2 + 640 * H * n * qPitch ** 5 * qRoll ** 4 + 57600 * H * n * qPitch ** 3 * qRoll ** 2 - 691200 * Lf * n - 691200 * H * qPitch - 537600.2 * L * qPitch ** 2 - 460800 * Lf * qPitch ** 2 - 30720 * H * qPitch ** 5 - 8960.003 * L * qPitch ** 6 - 7680 * Lf * qPitch ** 6 - 48 * H * qPitch ** 9 - 230400 * H * n * qPitch ** 3 - 92160 * Lf * n * qPitch ** 4 - 14400 * H * qPitch * qRoll ** 4 - 1920 * H * n * qPitch ** 7 - 240 * Lf * n * qPitch ** 8 - 172800 * H * n * qPitch * qRoll ** 2 - 57600 * H * qPitch ** 3 * qRoll ** 2 - 640 * H * qPitch ** 5 * qRoll ** 4 - 480 * H * qPitch ** 7 * qRoll ** 2 - H * qPitch ** 9 * qRoll ** 4 - 14400 * Ln_lin * (
                                            -1 + n) * qPitch ** 3 - 7680 * H * n * qPitch ** 5 * qRoll ** 2 - 4800 * H * n * qPitch ** 3 * qRoll ** 4 - 40 * H * n * qPitch ** 7 * qRoll ** 4 - 12 * H * n * qPitch ** 9 * qRoll ** 2) / (
                                -1 + n) ** 2)

        K = [[k_z_z, k_z_qRoll, k_z_qPitch], [k_qRoll_z, k_qRoll_qRoll, k_qRoll_qPitch],
             [k_qPitch_z, k_qPitch_qRoll, k_qPitch_qPitch]]

        return K

    def GetDampingMatrix(self, z=0, qPitch=0, qRoll=0):
        g = self.g
        L = self.L
        B = self.B
        Lf = self.Lf
        H = self.H
        k_spring_lin = self.k_spring_lin
        Ln_lin = self.Ln_lin
        b_damper = self.b_damper
        n = self.n

        b_z_z = 8 * b_damper
        b_z_qPitch = -8 * b_damper * (
                    Lf * cos(qPitch) - H * sin(qPitch) * cos(qRoll) - 1.5 * L * cos(qPitch) / (-1 + n))
        b_z_qRoll = 8 * b_damper * H * sin(qRoll) * cos(qPitch)
        b_qRoll_z = 8 * b_damper * H * sin(qRoll)
        b_qRoll_qRoll = 2 * b_damper * cos(qPitch) * (B ** 2 - (B ** 2 - 4 * H ** 2) * sin(qRoll) ** 2)
        b_qRoll_qPitch = -2 * b_damper * sin(qRoll) * (B ** 2 * sin(qPitch) * cos(qRoll) + 4 * H * (
                    Lf * cos(qPitch) - H * sin(qPitch) * cos(qRoll) - 1.5 * L * cos(qPitch) / (-1 + n)))
        b_qPitch_z = -8 * b_damper * (
                    Lf * cos(qPitch) - H * sin(qPitch) * cos(qRoll) - 1.5 * L * cos(qPitch) / (-1 + n))
        b_qPitch_qRoll = 2 * b_damper * sin(qRoll) * cos(qPitch) * (
                    6 * H * L * cos(qPitch) / (-1 + n) - 4 * H * Lf * cos(qPitch) - (B ** 2 - 4 * H ** 2) * sin(
                qPitch) * cos(qRoll))
        b_qPitch_qPitch = -0.5 * b_damper * ((Lf - 3 * L / (-1 + n)) * cos(qPitch) * (
                    12 * L * cos(qPitch) / (-1 + n) - 4 * Lf * cos(qPitch) - sin(qPitch) * (
                        B * sin(qRoll) - 2 * H * cos(qRoll))) + (Lf - 2 * L / (-1 + n)) * cos(qPitch) * (
                                                         8 * L * cos(qPitch) / (-1 + n) - 4 * Lf * cos(qPitch) - sin(
                                                     qPitch) * (B * sin(qRoll) - 2 * H * cos(qRoll))) - 4 * Lf * cos(
            qPitch) * (Lf * cos(qPitch) - 5.5 * H * sin(qPitch) * cos(qRoll)) - 2 * (Lf - L / (-1 + n)) * cos(
            qPitch) * (2 * Lf * cos(qPitch) - H * sin(qPitch) * cos(qRoll) - 2 * L * cos(qPitch) / (-1 + n)) - 2 * sin(
            qPitch) * (8 * H ** 2 * sin(qPitch) + 13 * H * L * cos(qPitch) * cos(qRoll) / (-1 + n) + 2 * (
                    B ** 2 - 4 * H ** 2) * sin(qPitch) * sin(qRoll) ** 2 - (Lf - 2.5 * L / (-1 + n)) * cos(qPitch) * (
                                   B * sin(qRoll) + 2 * H * cos(qRoll))))

        B = [[b_z_z, b_z_qRoll, b_z_qPitch], [b_qRoll_z, b_qRoll_qRoll, b_qRoll_qPitch],
             [b_qPitch_z, b_qPitch_qRoll, b_qPitch_qPitch]]

        # eigvalues, eigenvectors = linalg.eig(B)
        # lambda_min = min(eigvalues)

        return B

