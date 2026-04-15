#!/usr/bin/env python3
# encoding: utf-8
# Self-driving BACKUP plan:
# Phase 1: follows LEFT lane line, counts completed turns.
# Phase 2: after SWITCH_AFTER_TURNS turns, executes ROUTE_SEQUENCE
#          using odometry distance for straights, timed for turns/s-curve.
import os
import cv2
import math
import time
import queue
import rospy
import signal
import subprocess
import threading
import numpy as np
import lane_detect
import hiwonder_sdk.pid as pid
import hiwonder_sdk.misc as misc
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import hiwonder_sdk.common as common
from hiwonder_app.common import Heart
from hiwonder_interfaces.msg import ObjectsInfo
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from hiwonder_servo_controllers.bus_servo_control import set_servos
from hiwonder_sdk.common import cv2_image2ros, colors, plot_one_box
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from nav_msgs.msg import Odometry

# Switch to route sequence after this many lane-follow turns completed
SWITCH_AFTER_TURNS = 3
STOP_SIGN_MIN_AREA = 10000
# Ackermann steering constants (wheelbase = 0.213 m)
_SPD   = 0.15                                    # normal forward speed
_SPD_S = 0.10                                   # slower speed 
_R_ANG = _SPD   * math.tan(-0.6) / 0.213        # full right lock at _SPD
_SL_R  = _SPD * math.tan(-0.6) / 0.213        # slight right at _SPD
_SL_L  = _SPD * math.tan( 0.6) / 0.213        # slight left  at _SPD

# Route executed after SWITCH_AFTER_TURNS.
# Remaining route: right → s-curve → right → right → park
# Each step: (name, linear_x, angular_z, value, mode)
#   mode="dist" → drive until odometry Euclidean distance >= value (metres)
#   mode="time" → publish twist for value seconds
# Tune dist/time values on the mat before the demo.
ROUTE_SEQUENCE = [
    ("clear_intersection",  _SPD_S,   0.0,    0.33, "dist"),  # roll clear of junction
    ("right_turn_1",        _SPD,   _R_ANG, 5,  "time"),  # ~90° right
    ("straight_1",          _SPD,   0.0,    0.05, "dist"),
    ("s_curve_right",       _SPD, _SL_R,  3.9,  "time"),  # slight right
    ("s_curve_left",        _SPD, _SL_L,  3.3,  "time"),  # slight left
    ("straight_2",          _SPD,   0.0,    0.3, "dist"),
    ("right_turn_2",        _SPD,   _R_ANG, 5,  "time"), #test
    ("reverse_bit",  -0.12, 0.0, 1.5, "time"),
    ("right_turn_3",        _SPD,   _R_ANG, 2.5,  "time"), #test
    ("straight_to_park",    _SPD,   0.0,    0.50, "dist"),
    ("stop",                0.0,    0.0,    0.0,  "time"),
]

ROUTE_SEQUENCE_2 = [
    ("clear_intersection_2", _SPD, 0.0,   0.18, "dist"),
    ("right_turn_4",         _SPD, _R_ANG, 5.4,    "time"),
    ("straight_to_park",     _SPD, 0.0,    0.5, "dist"),
    ("stop",                 0.0,  0.0,    0.0,  "time"),
]

CLASS_NAMES = [
    'No_left', 'no_right', 'parking', 'speed_limit_5',
    'speed_limit_lift', 'stop', 'green', 'no_light',
    'red', 'yellow', 'no', 'turn_right',
]


class SelfDrivingTimedNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.01, 0.0, 0.0)
        self.param_init()

        self.image_queue = queue.Queue(maxsize=2)
        #self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.classes = CLASS_NAMES


        self.lock = threading.RLock()
        self.colors = common.Colors()
        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.acker_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=1)
        self.joints_pub = rospy.Publisher('/robot_1/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.result_publisher = rospy.Publisher('~image_result', Image, queue_size=1)

        self.enter_srv = rospy.Service('~enter', Trigger, self.enter_srv_callback)
        self.exit_srv = rospy.Service('~exit', Trigger, self.exit_srv_callback)
        self.set_running_srv = rospy.Service('~set_running', SetBool, self.set_running_srv_callback)
        self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
        """
        if not rospy.get_param('~only_line_follow', False):
            while not rospy.is_shutdown():
                try:
                    if rospy.get_param('/yolov5/init_finish'):
                        break
                except:
                    rospy.sleep(0.1)
            rospy.ServiceProxy('/yolov5/start', Trigger)()
        """
        while not rospy.is_shutdown():
            try:
                if rospy.get_param('/hiwonder_servo_manager/init_finish') and rospy.get_param('/joint_states_publisher/init_finish'):
                    break
            except:
                rospy.sleep(0.1)
        set_servos(self.joints_pub, 0.1, ((1, 500), ))
        rospy.sleep(1)
        self.acker_pub.publish(Twist())
        self.dispaly = False
        if rospy.get_param('~start', True):
            self.dispaly = True
            self.enter_srv_callback(None)
            self.set_running_srv_callback(SetBoolRequest(data=True))
        self.image_proc()

    def param_init(self):
        self.start = False
        self.enter = False

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1
	
        self.start_turn_time_stamp = 0   # safe: rospy not needed until first turn
        self.count_turn = 0
        self.start_turn = False

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False
        self.start_park = False
        self.lane_follow_enabled = True

        self.count_crosswalk = 0
        self.crosswalk_distance = 0
        self.crosswalk_length = 0.1 + 0.3

        self.start_slow_down = False
        self.normal_speed = 0.15
        self.slow_down_speed = 0.1

        self.traffic_signs_status = None
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.odom_sub = None
        self.objects_info = []

        # Odometry position (updated by odom_callback)
        self.current_x = 0.0
        self.current_y = 0.0

        # Phase tracking
        self.completed_turns = 0
        self.timed_mode = False
        self.phase3 = False


        self.slow_next_turn = False
        self.slow_down_over = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def enter_srv_callback(self, _):
        rospy.loginfo("self driving enter")
        with self.lock:
            self.start = False
            camera = rospy.get_param('/depth_camera_name', 'depth_cam')
            self.image_sub = rospy.Subscriber('/%s/rgb/image_raw' % camera, Image, self.image_callback)
            self.object_sub = rospy.Subscriber('/yolov5/object_detect', ObjectsInfo, self.get_object_callback)
            self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
            self.acker_pub.publish(Twist())
            self.enter = True
        return TriggerResponse(success=True)

    def exit_srv_callback(self, _):
        rospy.loginfo("self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
                if self.odom_sub is not None:
                    self.odom_sub.unregister()
            except Exception as e:
                rospy.logerr(str(e))
            self.acker_pub.publish(Twist())
        self.param_init()
        return TriggerResponse(success=True)

    def set_running_srv_callback(self, req: SetBoolRequest):
        rospy.loginfo("set_running")
        with self.lock:
            self.start = req.data
            if not self.start:
                self.acker_pub.publish(Twist())
        return SetBoolResponse(success=req.data)

    def shutdown(self, signum, frame):
        self.is_running = False
        rospy.loginfo('shutdown')

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    # ------------------------------------------------------------------
    # Phase 2 execution
    # ------------------------------------------------------------------
    def drive_step(self, lin_x, ang_z, value, mode, classes = []):
        """
        Publish one twist until the stopping criterion is met.
          mode="dist" — stop when odometry distance from start >= value metres
          mode="time" — stop after value seconds
        Always publishes a zero twist on exit.
        """
        action_done = False
        rospy.loginfo("=== PHASE 2: ROUTE SEQUENCE START ===")
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = ang_z
        if mode == "dist":
            start_x, start_y = self.current_x, self.current_y
            rate = rospy.Rate(20)
            while not rospy.is_shutdown() and self.is_running:
                dist = math.sqrt((self.current_x - start_x) ** 2 +
                                 (self.current_y - start_y) ** 2)
                sign = False
                if self.objects_info and not action_done :
                    for i in self.objects_info:
                        if i.class_name in classes:
                            self.acker_pub.publish(Twist())
                            if i.class_name == 'stop':
                                area = abs(i.box[2] - i.box[0]) * abs(i.box[3] - i.box[1])
                                if area >= STOP_SIGN_MIN_AREA:
                                    action_done = True
                                    sign = True
                                    rospy.sleep(3.0)
                            else:
                                rospy.sleep(1.0)
                                sign = True
                            break
                if dist >= value:
                    break
                if not sign:
                    self.acker_pub.publish(twist)
                rate.sleep()
        else:  # "time"
            if value > 0:
                self.acker_pub.publish(twist)
                rospy.sleep(value)
        self.acker_pub.publish(Twist())  # stop between steps

    def execute_route_sequence(self):
        """Run ROUTE_SEQUENCE step by step in a background thread."""
        rospy.loginfo("=== PHASE 2: ROUTE SEQUENCE START ===")
        set_servos(self.joints_pub, 0.01, ((1, 300),))
        for step_name, lin_x, ang_z, value, mode in ROUTE_SEQUENCE:
            if not self.is_running or rospy.is_shutdown():
                break
            if step_name == "straight_1":
                set_servos(self.joints_pub, 0.01, ((1, 500),))
            if step_name == "s_curve_left":
                set_servos(self.joints_pub, 0.01, ((1, 700),))
            if step_name == "right_turn_2":
                #set_servos(self.joints_pub, 0.01, ((1, 400),))
                set_servos(self.joints_pub, 0.01, ((1, 500),))
            if step_name == "right_turn_3":
                self.completed_turns = 2
                self.start_turn = False
                self.count_turn = 0
                self.start_turn_time_stamp = 0
                self.phase3 = True
                self.timed_mode = False
                rospy.loginfo("=== BACK TO LANE FOLLOW FOR 1 TURN ===")
                return
            rospy.loginfo(f"  [{step_name}] mode={mode}, value={value}")
            self.drive_step(lin_x, ang_z, value, mode, classes = ['red','stop'])
        set_servos(self.joints_pub, 0.01, ((1, 500),))
        rospy.loginfo("=== PHASE 2: ROUTE SEQUENCE DONE ===")
        self.acker_pub.publish(Twist())
        self.stop = True

    def execute_route_sequence_2(self):
        rospy.loginfo("=== PHASE 3: ROUTE SEQUENCE 2 START ===")
        for step_name, lin_x, ang_z, value, mode in ROUTE_SEQUENCE_2:
            if not self.is_running or rospy.is_shutdown():
                break
            rospy.loginfo(f"  [{step_name}] mode={mode}, value={value}")
            self.drive_step(lin_x, ang_z, value, mode)
        rospy.loginfo("=== PHASE 3: DONE ===")
        self.acker_pub.publish(Twist())
        self.lane_follow_enabled = False
        self.stop = True
        rospy.loginfo("=== Launching parking_test ===")
        parking_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parking_test.py')
        subprocess.Popen(['python3', parking_script])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def image_proc(self):
        while self.is_running:
            time_start = time.time()
            image = self.image_queue.get(block=True)
            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                if self.timed_mode:
                    # Phase 2/3 runs in its own thread
                    if self.lane_follow_enabled:
                        binary_image = self.lane_detect.get_binary(image)
                        result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())
                        if lane_x >= 0:
                            self.pid.SetPoint = 100
                            self.pid.update(lane_x)
                elif self.lane_follow_enabled:
                    # Phase 1: lane following + turn counting
                    binary_image = self.lane_detect.get_binary(image)
                    twist = Twist()
                    twist.linear.x = self.normal_speed
                    result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())

                    if lane_x >= 0 and not self.stop:
                        if lane_x > 150:  # turning — line shifted right
                            if self.turn_right:
                                self.count_right_miss += 1
                                if self.count_right_miss >= 50:
                                    self.count_right_miss = 0
                                    self.turn_right = False
                            self.count_turn += 1
                            if self.count_turn > 5 and not self.start_turn:
                                self.start_turn = True
                                self.count_turn = 0
                                self.start_turn_time_stamp = rospy.get_time()
                            twist.angular.z = twist.linear.x * math.tan(-0.6) / 0.213
                        else:  # straight — line back near centre
                            self.count_turn = 0
                            if rospy.get_time() - self.start_turn_time_stamp > 3.5 and self.start_turn:
                                # Turn just finished
                                self.start_turn = False
                                self.completed_turns += 1
                                rospy.loginfo(f"Turn done: {self.completed_turns}/{SWITCH_AFTER_TURNS}")
                                if self.slow_next_turn and not self.slow_down_over:
                                    self.normal_speed = 0.1
                                if self.completed_turns >= SWITCH_AFTER_TURNS and not self.phase3:
                                    rospy.loginfo("Switching to Phase 2")
                                    self.timed_mode = True
                                    self.acker_pub.publish(Twist())
                                    threading.Thread(target=self.execute_route_sequence, daemon=True).start()
                                elif self.completed_turns >= SWITCH_AFTER_TURNS:
                                    rospy.loginfo("=== PHASE 3: ROUTE SEQUENCE 2 ===")
                                    self.timed_mode = True
                                    self.acker_pub.publish(Twist())
                                    threading.Thread(target=self.execute_route_sequence_2, daemon=True).start()


                            if not self.start_turn:
                                self.pid.SetPoint = 100
                                self.pid.update(lane_x)
                                twist.angular.z = twist.linear.x * math.tan(
                                    misc.set_range(self.pid.output, -0.1, 0.1)) / 0.213
                            else:
                                twist.angular.z = 0.15 * math.tan(-0.6) / 0.213
                        self.acker_pub.publish(twist)
                    else:
                        self.pid.clear()

                # Draw YOLO detections on result image
                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(box, result_image, color=color,
                                     label="{}:{:.2f}".format(class_name, cls_conf))
                        if not self.slow_next_turn and i.class_name == 'speed_limit_5':
                            self.slow_next_turn = True
                            self.slow_down_over = False
                        elif not self.slow_down_over and i.class_name == 'speed_limit_lift' and self.normal_speed < 0.15:
                            if i.box[1] > 190:
                                self.slow_down_over = True
                                self.normal_speed = 0.15
            else:
                rospy.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.dispaly:
                cv2.imshow('result', bgr_image)
                key = cv2.waitKey(1)
                if key != -1:
                    self.is_running = False
            self.result_publisher.publish(cv2_image2ros(bgr_image))
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.acker_pub.publish(Twist())

    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if not self.objects_info:
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2]) / 2), int((i.box[1] + i.box[3]) / 2))
                if class_name == 'crosswalk':
                    if center[1] > min_distance:
                        min_distance = center[1]
                elif class_name == 'right':
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 10:
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'park':
                    self.park_x = center[0]
                elif class_name in ('red', 'green'):
                    self.traffic_signs_status = i
            self.crosswalk_distance = min_distance


if __name__ == "__main__":
    SelfDrivingTimedNode('self_driving')
