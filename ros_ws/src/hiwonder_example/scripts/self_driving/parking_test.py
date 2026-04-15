#!/usr/bin/env python3
# Test parking: lateral shift manoeuvres (stay parallel).
# Each shift = turn then counter-turn (same duration) = pure lateral move.
# Launch YOLO + camera separately, then run this node.

import rospy
import math
from geometry_msgs.msg import Twist
from hiwonder_interfaces.msg import ObjectsInfo

PARK_TARGET_CX = 78
PARK_TARGET_AREA = 10000
PARK_CX_TOL = 40
PARK_AREA_TOL = 2000

# Shift manoeuvre (créneau 4 temps):
#   1. braque + avance  2. tout droit  3. contre-braque + avance  4. recule droit
SHIFT_SPEED = 0.10
SHIFT_ANGLE = 0.6         # braquage
SHIFT_TURN_DUR = 1.9       # durée braquage (étapes 1 et 3)
SHIFT_STRAIGHT_DUR = 1.0   # durée tout droit au milieu (étape 2) — c'est ça qui décale
SHIFT_REVERSE_DUR = 2.5    # durée recul droit (étape 4) — compense l'avance totale

# Distance correction: straight forward/back
DIST_SPEED_FWD = 0.08
DIST_SPEED_REV = -0.06
DIST_DURATION = 1

MAX_ITERS = 20
PAUSE = 0.5                # pause entre manoeuvres pour lire la caméra

class TestParking:
    def __init__(self):
        rospy.init_node('test_parking', anonymous=True)
        self.objects_info = []
        self.acker_pub = rospy.Publisher('/robot_1/hiwonder_controller/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/yolov5/object_detect', ObjectsInfo, self.obj_callback)
        rospy.loginfo("=== TEST PARKING (lateral shift) ===")
        rospy.sleep(1.0)
        self.run()

    def obj_callback(self, msg):
        self.objects_info = msg.objects

    def get_park_info(self):
        for obj in self.objects_info:
            if obj.class_name == 'parking':
                box = obj.box
                cx = (box[0] + box[2]) / 2.0
                area = (box[2] - box[0]) * (box[3] - box[1])
                return cx, area
        return None, None

    def stop(self):
        self.acker_pub.publish(Twist())

    def send(self, lin_x, angle):
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = lin_x * math.tan(angle) / 0.213
        self.acker_pub.publish(twist)

    def shift_right(self):
        """Créneau droite: braque droite, tout droit, braque gauche, recule."""
        rospy.loginfo("[SHIFT] Right")
        self.send(SHIFT_SPEED, -SHIFT_ANGLE)       # 1. braque droite + avance
        rospy.sleep(SHIFT_TURN_DUR)
        self.send(SHIFT_SPEED, 0)                   # 2. tout droit (décale les roues arrière)
        rospy.sleep(SHIFT_STRAIGHT_DUR)
        self.send(SHIFT_SPEED, SHIFT_ANGLE)          # 3. contre-braque gauche + avance
        rospy.sleep(SHIFT_TURN_DUR)
        self.stop()
        self.send(-SHIFT_SPEED, 0)                   # 4. recule droit (compense l'avance)
        rospy.sleep(SHIFT_REVERSE_DUR)
        self.stop()

    def shift_left(self):
        """Créneau gauche: braque gauche, tout droit, braque droite, recule."""
        rospy.loginfo("[SHIFT] Left")
        self.send(SHIFT_SPEED, SHIFT_ANGLE)          # 1. braque gauche + avance
        rospy.sleep(SHIFT_TURN_DUR)
        self.send(SHIFT_SPEED, 0)                    # 2. tout droit
        rospy.sleep(SHIFT_STRAIGHT_DUR)
        self.send(SHIFT_SPEED, -SHIFT_ANGLE)         # 3. contre-braque droite + avance
        rospy.sleep(SHIFT_TURN_DUR)
        self.stop()
        self.send(-SHIFT_SPEED, 0)                   # 4. recule droit
        rospy.sleep(SHIFT_REVERSE_DUR)
        self.stop()

    def move_forward(self):
        rospy.loginfo("[DIST] Forward")
        self.send(DIST_SPEED_FWD, 0)
        rospy.sleep(DIST_DURATION)
        self.stop()

    def move_backward(self):
        rospy.loginfo("[DIST] Backward")
        self.send(DIST_SPEED_REV, 0)
        rospy.sleep(DIST_DURATION)
        self.stop()

    def run(self):
        for i in range(MAX_ITERS):
            if rospy.is_shutdown():
                break

            rospy.sleep(PAUSE)
            cx, area = self.get_park_info()

            if cx is None:
                rospy.loginfo("[PARK] No sign — stopping")
                self.stop()
                rospy.sleep(5)
                continue

            error_cx = cx - PARK_TARGET_CX
            error_area = area - PARK_TARGET_AREA
            cx_ok = abs(error_cx) < PARK_CX_TOL
            area_ok = error_area < 3000 and -1500 < error_area

            rospy.loginfo("[PARK iter %d] cx=%.0f (err=%.0f) area=%.0f (err=%.0f) cx_ok=%s area_ok=%s" % (
                i, cx, error_cx, area, error_area, cx_ok, area_ok))

            if cx_ok and area_ok:
                rospy.loginfo("[PARK] Parked! cx=%.0f area=%.0f" % (cx, area))
                self.stop()
                return

            # Priorité: corriger cx d'abord, puis area
            if not cx_ok and area > 4000:
                self.move_backward()
            elif not cx_ok:
                if error_cx > 0:
                    self.shift_right()  # panneau trop à droite → se décaler à droite
                else:
                    self.shift_left()   # panneau trop à gauche → se décaler à gauche
            elif not area_ok:
                if error_area < 0:
                    self.move_forward()
                else:
                    self.move_backward()

        rospy.loginfo("[PARK] Max iterations — stopping")
        self.stop()

if __name__ == "__main__":
    TestParking()

