#!/usr/bin/python3
# coding=utf8
# 阿克曼底盘控制(Ackermann wheel chassis control)
import math
from ros_robot_controller.msg import MotorState

class AckermannChassis:
    # wheelbase(H) = 0.213  # 前后轴距
    # track_width(W) = 0.222  # 左右轴距
    # wheel_diameter(R) = 0.101  # 轮子直径

    def __init__(self, wheelbase=0.213, track_width=0.222, wheel_diameter=0.101):
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.wheel_diameter = wheel_diameter

    def speed_covert(self, speed):
        """
        covert speed m/s to rps/s
        :param speed:
        :return:
        """
        return speed / (math.pi * self.wheel_diameter)

    def set_velocity(self, linear_speed, angular_speed, reset_servo=True):
        # 逆解
        # Vx, Vz -> VL, VR, angleL, angleR
        # O1O2 = Vx*T
        # angle = Vz*T
        # R = O1O2/angle = Vx*T/Vz*T = Vx/Vz
        # VL = Vx*(R-0.5W)/R = Vx*(1-0.5W/R)=Vx(1-0.5W/(Vx/Vz))=Vx-Vz*0.5w 
        # VR = Vx*(R+0.5W)/R = Vx*(1+0.5W/R)=Vx(1+0.5W/(Vx/Vz))=Vx+Vz*0.5w
        # angleR = atan(H/(R-0.5W))=atan(H/(vX/vz-0.5w))
        # angleL = atan(H/(R+0.5W))=atan(H/(vX/vz+0.5w))
        # 二次曲线去拟合angleR和servo_angle
        # 近似解angleR=angleL=atan(H/R)=atan(H*Vz/Vx)

        # 正解
        # VL, VR -> Vx, Vz, angleR, angleL
        # VL + VR = Vx*(R-0.5W)/R + Vx*(R+0.5W)/R = 2Vx
        # Vx = (VL + VR)/2
        # VR - VL = Vx*(R+0.5W)/R - Vx*(R-0.5W)/R = Vx(W/R)
        # R = Vx(W/(VR - VL))
        # Vx/Vz = R = Vx(W/(VR - VL))
        # Vz = (VR - VL)/W
        # angleR = atan(H/(R-0.5W))
        # angleL = atan(H/(R+0.5W))
        
        servo_angle = 500
        data = []
        if abs(linear_speed) >= 1e-8:
            if abs(angular_speed) >= 1e-8:
                theta = math.atan(self.wheelbase*angular_speed/linear_speed)
                steering_angle = theta
                # print(math.degrees(steering_angle))
                if abs(steering_angle) > math.radians(37):
                    for i in range(4):
                        msg = MotorState()
                        msg.id = i + 1
                        msg.rps = 0
                        data.append(msg) 
                    return None, data
                servo_angle = 500 + 1000*math.degrees(steering_angle)/240

            vr = linear_speed + angular_speed*self.track_width/2
            vl = linear_speed - angular_speed*self.track_width/2
            v_s = [self.speed_covert(v) for v in [0, vl, 0, -vr]]
            for i in range(len(v_s)):
                msg = MotorState()
                msg.id = i + 1
                msg.rps = v_s[i]
                data.append(msg) 
            return servo_angle, data
        else:
            for i in range(4):
                msg = MotorState()
                msg.id = i + 1
                msg.rps = 0
                data.append(msg) 
            return None, data

