#!/usr/bin/env python3
# encoding: utf-8
# Self-driving with sequence-based navigation.
# Each step defines: which line to follow + what to do when turn is detected.
#
# Step format: (follow_side, turn_action)
#   follow_side:  "left" or "right"
#   turn_action:  "left", "right", "straight", "stop"
#
# "straight" = follow the line, count the turn, advance to next step
# "left"/"right" = apply fixed steering during the turn
# "stop" = stop the robot
#
# The robot detects a turn when lane_x exceeds a threshold.
# When the turn ends (lane_x back below threshold for 2s), it advances to the next step.

import os
import cv2
import math
import time
import queue
import rospy
import signal
import threading
import numpy as np
import lane_detect_seq as lane_detect
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

LEFT_ROIS  = ((450, 480, 0, 320, 0.7), (390, 420, 0, 320, 0.2), (330, 360, 0, 320, 0.1))
RIGHT_ROIS = ((450, 480, 320, 640, 0.7), (390, 420, 320, 640, 0.2), (330, 360, 320, 640, 0.1))

# ============================================================
# ROUTE SEQUENCE — edit this to match your parcours
# (follow_side, turn_action)
# ============================================================
ROUTE = [
    ("left",  "right"),    # Step 0
    ("left",  "right"),    # Step 1
    ("left",  "right"),    # Step 2
    ("right", "right"),    # Step 3
    ("right", "right"),    # Step 4
    ("none",  "reverse"),  # Step 5: back up (timed)
    ("left",  "left"),     # Step 6
    ("right", "right"),    # Step 7
    ("left",  "right"),    # Step 8
    ("right", "right"),    # Step 9
    ("none",  "forward"),  # Step 10: go straight then stop
]

# Reverse parameters (tune on the mat)
REVERSE_SPEED = -0.12
REVERSE_DURATION = 1.5  # seconds

# Forward parameters (tune on the mat)
FORWARD_SPEED = 0.15
FORWARD_DURATION = 3.0  # seconds

# Thresholds and PID setpoints per follow side
CONFIG = {
    "left": {
        "rois": LEFT_ROIS,
        "turn_threshold": 150,         # outside: lane_x > 150
        "inside_turn_threshold": 50,   # inside: lane_x < 50 (line goes far left)
        "setpoint": 100,
    },
    "right": {
        "rois": RIGHT_ROIS,
        "turn_threshold": 500,         # outside: lane_x > 500
        "inside_turn_threshold": 560,  # inside: lane_x < 560 (line departs from normal ~620)
        "setpoint": 620,     # higher = robot stays further from line
    },
}

# Steering angles during turns
TURN_STEERING = {
    "right": -0.6,   # tan angle for right turn (negative = right)
    "left":   0.6,   # tan angle for left turn (positive = left)
}

TURN_COOLDOWN = 5.0  # seconds between two consecutive turn completions
TURN_END_TIMEOUT = 4.0  # seconds below threshold before turn is considered done

# Skip first N steps for faster testing (set to 0 for full run)
START_STEP = 3  # start at step 2 (after 2nd turn)

# Inside turns: drive forward before steering (tune on the mat)
# Inside turn = follow_side matches turn_direction (e.g. right+right, left+left)
INSIDE_TURN_FORWARD_DURATION = 0.0  # seconds to drive straight before turning
INSIDE_TURN_MIN_STEER_DURATION = 5.  # seconds of forced steering for inside turns
INSIDE_TURN_BACKUP_SPEED = -0.15    # reverse speed before inside turn
INSIDE_TURN_BACKUP_DURATION = 3.0  # seconds to reverse before inside turn
TURN_DETECT_GRACE = 1.0  # seconds after step change where turn detection is disabled (outside turns only)
GRACE_SPEED = 0.08  # slow speed during grace period to let PID settle

# Settle loop for inside turns: forward/backward until aligned with line
SETTLE_FORWARD_SPEED = 0.06   # slow forward during settling
SETTLE_FORWARD_DURATION = 4.0 # max seconds forward before giving up and reversing
SETTLE_BACKWARD_SPEED = -0.12 # reverse speed during settling
SETTLE_BACKWARD_DURATION = 1.5 # seconds to reverse before trying forward again
SETTLE_TOLERANCE = 60         # lane_x must be within this distance from setpoint
SETTLE_GOOD_FRAMES = 8        # consecutive good frames to consider settled
SETTLE_PID_RANGE = 0.3        # PID steering range during settling (gentler than 0.6)
SETTLE_COUNTER_STEER = 0.5    # stronger correction when overshooting past setpoint
# Straighten phase: counter-steer after reaching setpoint to become parallel
SETTLE_STRAIGHTEN_ANGLE = 0.6 # counter-steer angle (opposite to approach direction)
SETTLE_STRAIGHTEN_DURATION = 1.5  # seconds of counter-steering to straighten out


class SelfDrivingSeqNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.01, 0.0, 0.0)
        self.param_init()

        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']

        self.lock = threading.RLock()
        self.colors = common.Colors()
        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.acker_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=1)
        self.joints_pub = rospy.Publisher('servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.result_publisher = rospy.Publisher('~image_result', Image, queue_size=1)

        self.enter_srv = rospy.Service('~enter', Trigger, self.enter_srv_callback)
        self.exit_srv = rospy.Service('~exit', Trigger, self.exit_srv_callback)
        self.set_running_srv = rospy.Service('~set_running', SetBool, self.set_running_srv_callback)
        self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))

        if not rospy.get_param('~only_line_follow', False):
            while not rospy.is_shutdown():
                try:
                    if rospy.get_param('/yolov5/init_finish'):
                        break
                except:
                    rospy.sleep(0.1)
            rospy.ServiceProxy('/yolov5/start', Trigger)()
        while not rospy.is_shutdown():
            try:
                if rospy.get_param('/hiwonder_servo_manager/init_finish') and rospy.get_param('/joint_states_publisher/init_finish'):
                    break
            except:
                rospy.sleep(0.1)
        set_servos(self.joints_pub, 1, ((1, 500), ))
        rospy.sleep(1)
        self.acker_pub.publish(Twist())
        self.dispaly = False
        if rospy.get_param('~start', True):
            self.dispaly = True
            self.enter_srv_callback(None)
            self.set_running_srv_callback(SetBoolRequest(data=True))

        # Apply initial ROI
        self._apply_step_config()
        rospy.loginfo("=== ROUTE: %d steps ===" % len(ROUTE))
        for i, (side, action) in enumerate(ROUTE):
            rospy.loginfo("  Step %d: follow_%s, turn_%s" % (i, side, action))

        self.image_proc()

    def param_init(self):
        self.start = False
        self.enter = False
        self.stop = False

        self.start_turn_time_stamp = time.time()
        self.count_turn = 0
        self.start_turn = False

        self.normal_speed = 0.15

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # Sequence state
        self.current_step = START_STEP
        self.last_turn_completed_time = time.time()
        self.straight_start_time = 0
        self.inside_forward_start = 0
        self.inside_forward_active = False
        self.inside_steer_start = 0
        self.inside_steer_active = False
        self.step_start_time = time.time()
        self.was_below_threshold = False  # must see straight before detecting turn
        # Settle loop state (inside turns)
        self.settling = False
        self.settle_direction = "forward"  # "forward", "backward", or "straighten"
        self.settle_start = 0
        self.settle_good_count = 0
        self.settle_approach_from = 0  # +1 = approached from right, -1 = from left

    def _apply_step_config(self):
        """Apply ROI config for the current step."""
        if self.current_step >= len(ROUTE):
            return
        follow_side, action = ROUTE[self.current_step]

        # Handle special actions immediately
        if action == "reverse":
            rospy.loginfo("Step %d: REVERSE" % self.current_step)
            threading.Thread(target=self._execute_reverse, daemon=True).start()
            return
        if action == "forward":
            rospy.loginfo("Step %d: FORWARD" % self.current_step)
            threading.Thread(target=self._execute_forward, daemon=True).start()
            return

        if follow_side == "none":
            return

        cfg = CONFIG[follow_side]
        self.lane_detect.set_roi(cfg["rois"])
        self.step_start_time = rospy.get_time()
        self.count_turn = 0
        self.start_turn = False
        self.was_below_threshold = False

        is_inside = self._is_inside_turn(self.current_step)
        th = cfg["inside_turn_threshold"] if is_inside else cfg["turn_threshold"]
        comp = "<" if (is_inside and follow_side == "left") else ">"
        rospy.loginfo("Step %d: follow_%s, turn_%s (setpoint=%d, threshold: lane_x %s %d, grace=%.1fs)" % (
            self.current_step, follow_side, action, cfg["setpoint"], comp, th, TURN_DETECT_GRACE))

    def _execute_reverse(self):
        """Back up for a fixed duration, then advance to next step."""
        rospy.loginfo("Reversing for %.1fs..." % REVERSE_DURATION)
        twist = Twist()
        twist.linear.x = REVERSE_SPEED
        self.acker_pub.publish(twist)
        rospy.sleep(REVERSE_DURATION)
        self.acker_pub.publish(Twist())
        rospy.sleep(0.3)
        rospy.loginfo("Reverse done")
        self._advance_step()

    def _execute_forward(self):
        """Go straight for a fixed duration, then stop."""
        rospy.loginfo("Going straight for %.1fs..." % FORWARD_DURATION)
        twist = Twist()
        twist.linear.x = FORWARD_SPEED
        self.acker_pub.publish(twist)
        rospy.sleep(FORWARD_DURATION)
        self.acker_pub.publish(Twist())
        rospy.loginfo("=== ROUTE COMPLETE ===")
        self.stop = True

    def _is_inside_turn(self, step_idx):
        """Check if a step is an inside turn (follow side == turn direction)."""
        if step_idx >= len(ROUTE):
            return False
        side, action = ROUTE[step_idx]
        return (side == "right" and action == "right") or (side == "left" and action == "left")

    def _advance_step(self):
        """Move to next step in the route."""
        self.current_step += 1
        self.last_turn_completed_time = rospy.get_time()
        if self.current_step >= len(ROUTE):
            rospy.loginfo("=== ROUTE COMPLETE ===")
            self.acker_pub.publish(Twist())
            self.stop = True
            return

        # If next step is an inside turn, back up then enter settle loop
        if self._is_inside_turn(self.current_step):
            rospy.loginfo("[BACKUP] Step %d is inside turn — stopping..." % self.current_step)
            self.acker_pub.publish(Twist())
            rospy.sleep(0.2)
            rospy.loginfo("[BACKUP] Reversing at %.2f for %.1fs" % (INSIDE_TURN_BACKUP_SPEED, INSIDE_TURN_BACKUP_DURATION))
            twist = Twist()
            twist.linear.x = INSIDE_TURN_BACKUP_SPEED
            self.acker_pub.publish(twist)
            rospy.sleep(INSIDE_TURN_BACKUP_DURATION)
            self.acker_pub.publish(Twist())
            rospy.sleep(0.5)
            rospy.loginfo("[BACKUP] Done — entering settle loop")
            self.settling = True
            self.settle_direction = "forward"
            self.settle_start = rospy.get_time()
            self.settle_good_count = 0

        self._apply_step_config()
        # Check if current step is a terminal action
        _, action = ROUTE[self.current_step]
        if action == "stop":
            rospy.loginfo("=== STOP action — halting ===")
            self.acker_pub.publish(Twist())
            self.stop = True
        elif action == "straight":
            rospy.loginfo("=== STRAIGHT — following line for 10s then stopping ===")
            self.straight_start_time = rospy.get_time()

    def enter_srv_callback(self, _):
        rospy.loginfo("self driving enter")
        with self.lock:
            self.start = False
            camera = rospy.get_param('/depth_camera_name', 'depth_cam')
            self.image_sub = rospy.Subscriber('/%s/rgb/image_raw' % camera, Image, self.image_callback)
            self.object_sub = rospy.Subscriber('/yolov5/object_detect', ObjectsInfo, self.get_object_callback)
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
            except Exception as e:
                rospy.logerr(str(e))
            self.acker_pub.publish(Twist())
        self.param_init()
        return TriggerResponse(success=True)

    def set_running_srv_callback(self, req):
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

    def image_proc(self):
        while self.is_running:
            time_start = time.time()
            image = self.image_queue.get(block=True)
            result_image = image.copy()
            if self.start and not self.stop:
                h, w = image.shape[:2]
                binary_image = self.lane_detect.get_binary(image)
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())

                twist = Twist()
                twist.linear.x = self.normal_speed

                if self.current_step < len(ROUTE):
                    follow_side, turn_action = ROUTE[self.current_step]

                    # Skip lane following during reverse/forward (handled in thread)
                    if turn_action in ("reverse", "forward"):
                        pass
                    elif turn_action == "straight":
                        if lane_x >= 0:
                            cfg = CONFIG[follow_side]
                            self.pid.SetPoint = cfg["setpoint"]
                            self.pid.update(lane_x)
                            twist.angular.z = twist.linear.x * math.tan(
                                misc.set_range(self.pid.output, -0.1, 0.1)) / 0.213
                            self.acker_pub.publish(twist)
                        if hasattr(self, 'straight_start_time') and rospy.get_time() - self.straight_start_time > 10:
                            rospy.loginfo("=== STRAIGHT timeout — stopping ===")
                            self.acker_pub.publish(Twist())
                            self.stop = True
                    else:
                        cfg = CONFIG[follow_side]
                        setpoint = cfg["setpoint"]
                        steer_angle = TURN_STEERING.get(turn_action, TURN_STEERING["right"])

                        is_inside = (follow_side == "right" and turn_action == "right") or \
                                    (follow_side == "left" and turn_action == "left")

                        # Pick threshold + comparison direction
                        if is_inside:
                            threshold = cfg["inside_turn_threshold"]
                        else:
                            threshold = cfg["turn_threshold"]

                        # Determine if we're in a turn
                        if is_inside:
                            # Inside turn: line departs from normal position (drops for both sides)
                            # Also count line loss (lane_x == -1) as in_turn
                            in_turn = lane_x == -1 or (lane_x >= 0 and lane_x < threshold)
                        else:
                            in_turn = lane_x >= 0 and lane_x > threshold

                        # ---- FORCED INSIDE STEER (keeps turning even if line lost) ----
                        if self.inside_steer_active:
                            elapsed = rospy.get_time() - self.inside_steer_start
                            if elapsed < INSIDE_TURN_MIN_STEER_DURATION:
                                rospy.loginfo_throttle(0.5, "[INSIDE_STEER] Steering %.1f/%.1fs (lane_x=%s)" % (elapsed, INSIDE_TURN_MIN_STEER_DURATION, lane_x))
                                twist.angular.z = twist.linear.x * math.tan(steer_angle) / 0.213
                                self.acker_pub.publish(twist)
                            else:
                                rospy.loginfo("[INSIDE_STEER] Done (%.1fs) → advance step" % elapsed)
                                self.inside_steer_active = False
                                self.inside_forward_active = False
                                self.start_turn = False
                                self._advance_step()

                        # ---- SETTLE LOOP (inside turns: forward/backward until aligned) ----
                        elif self.settling:
                            if self.settle_direction == "forward":
                                if lane_x >= 0:
                                    # Track which side we're approaching from
                                    error = lane_x - setpoint
                                    if error > 0:
                                        self.settle_approach_from = 1   # line is to the right of setpoint
                                    elif error < 0:
                                        self.settle_approach_from = -1  # line is to the left of setpoint

                                    # Drive forward slowly with PID + counter-steer on overshoot
                                    twist.linear.x = SETTLE_FORWARD_SPEED
                                    self.pid.SetPoint = setpoint
                                    self.pid.update(lane_x)
                                    if follow_side == "right" and error < 0:
                                        pid_range = SETTLE_COUNTER_STEER
                                    elif follow_side == "left" and error > 0:
                                        pid_range = SETTLE_COUNTER_STEER
                                    else:
                                        pid_range = SETTLE_PID_RANGE
                                    twist.angular.z = twist.linear.x * math.tan(
                                        misc.set_range(self.pid.output, -pid_range, pid_range)) / 0.213
                                    dist = abs(lane_x - setpoint)
                                    if dist < SETTLE_TOLERANCE:
                                        self.settle_good_count += 1
                                        rospy.loginfo_throttle(0.5, "[SETTLE_FWD] lane_x=%d setpoint=%d dist=%d good=%d/%d" % (
                                            lane_x, setpoint, dist, self.settle_good_count, SETTLE_GOOD_FRAMES))
                                        if self.settle_good_count >= SETTLE_GOOD_FRAMES:
                                            # Position OK → straighten out before continuing
                                            rospy.loginfo("[SETTLE] Aligned! → straightening (approach_from=%d)" % self.settle_approach_from)
                                            self.settle_direction = "straighten"
                                            self.settle_start = rospy.get_time()
                                    else:
                                        self.settle_good_count = 0
                                        rospy.loginfo_throttle(0.5, "[SETTLE_FWD] lane_x=%d setpoint=%d dist=%d — not aligned" % (lane_x, setpoint, dist))
                                    # Timeout: switch to backward
                                    if self.settle_direction == "forward" and rospy.get_time() - self.settle_start > SETTLE_FORWARD_DURATION:
                                        rospy.loginfo("[SETTLE_FWD] Timeout — switching to backward")
                                        self.settle_direction = "backward"
                                        self.settle_start = rospy.get_time()
                                        self.settle_good_count = 0
                                else:
                                    # Line lost during forward — switch to backward
                                    rospy.loginfo_throttle(0.5, "[SETTLE_FWD] Line lost — switching to backward")
                                    self.settle_direction = "backward"
                                    self.settle_start = rospy.get_time()
                                    self.settle_good_count = 0
                                    twist.linear.x = 0
                            elif self.settle_direction == "backward":
                                # Reverse slowly, no PID
                                twist.linear.x = SETTLE_BACKWARD_SPEED
                                twist.angular.z = 0
                                rospy.loginfo_throttle(0.5, "[SETTLE_BWD] Reversing %.1f/%.1fs" % (
                                    rospy.get_time() - self.settle_start, SETTLE_BACKWARD_DURATION))
                                if rospy.get_time() - self.settle_start > SETTLE_BACKWARD_DURATION:
                                    rospy.loginfo("[SETTLE_BWD] Done — switching to forward")
                                    self.settle_direction = "forward"
                                    self.settle_start = rospy.get_time()
                                    self.settle_good_count = 0
                            elif self.settle_direction == "straighten":
                                # Counter-steer to become parallel to the line
                                elapsed = rospy.get_time() - self.settle_start
                                if elapsed < SETTLE_STRAIGHTEN_DURATION:
                                    twist.linear.x = SETTLE_FORWARD_SPEED
                                    # Steer opposite to approach direction
                                    # approach_from=+1 means we came from the right → PID steered left → counter-steer RIGHT
                                    counter_angle = SETTLE_STRAIGHTEN_ANGLE * self.settle_approach_from
                                    twist.angular.z = twist.linear.x * math.tan(counter_angle) / 0.213
                                    rospy.loginfo_throttle(0.3, "[SETTLE_STR] Straightening %.1f/%.1fs (angle=%.2f, lane_x=%s)" % (
                                        elapsed, SETTLE_STRAIGHTEN_DURATION, counter_angle, lane_x))
                                else:
                                    rospy.loginfo("[SETTLE_STR] Done → ready (grace period reset)")
                                    self.settling = False
                                    self.step_start_time = rospy.get_time()  # reset grace period
                            self.acker_pub.publish(twist)

                        # ---- NORMAL LANE FOLLOWING + TURN DETECTION ----
                        elif lane_x >= 0:
                            rospy.loginfo_throttle(1.0, "[FOLLOW] Step %d: lane_x=%d threshold=%d start_turn=%s" % (self.current_step, lane_x, threshold, self.start_turn))

                            in_grace = (rospy.get_time() - self.step_start_time) < TURN_DETECT_GRACE

                            if in_grace:
                                # Grace period — slow down + PID only, no turn detection (outside turns only)
                                rospy.loginfo_throttle(1.0, "[GRACE] %.1fs remaining" % (TURN_DETECT_GRACE - (rospy.get_time() - self.step_start_time)))
                                twist.linear.x = GRACE_SPEED
                                self.pid.SetPoint = setpoint
                                self.pid.update(lane_x)
                                twist.angular.z = twist.linear.x * math.tan(
                                    misc.set_range(self.pid.output, -0.1, 0.1)) / 0.213
                            elif in_turn and not self.was_below_threshold:
                                # Above threshold but never been below yet — just PID, don't count as turn
                                rospy.loginfo_throttle(1.0, "[FOLLOW] lane_x=%d above threshold but not armed yet — PID only" % lane_x)
                                self.pid.SetPoint = setpoint
                                self.pid.update(lane_x)
                                twist.angular.z = twist.linear.x * math.tan(
                                    misc.set_range(self.pid.output, -0.1, 0.1)) / 0.213
                            elif in_turn and self.was_below_threshold:
                                # ---- IN A TURN (only if we were straight first) ----
                                self.count_turn += 1
                                if self.count_turn > 5 and not self.start_turn:
                                    self.start_turn = True
                                    self.count_turn = 0
                                    self.start_turn_time_stamp = rospy.get_time()
                                    rospy.loginfo("[TURN] Detected! Step %d (%s/%s) inside=%s" % (self.current_step, follow_side, turn_action, is_inside))
                                    if is_inside:
                                        self.inside_forward_active = True
                                        self.inside_forward_start = rospy.get_time()
                                        rospy.loginfo("[INSIDE_FWD] Driving forward for %.1fs..." % INSIDE_TURN_FORWARD_DURATION)

                                if is_inside and self.inside_forward_active:
                                    elapsed_fwd = rospy.get_time() - self.inside_forward_start
                                    rospy.loginfo_throttle(0.5, "[INSIDE_FWD] Forward %.1f/%.1fs" % (elapsed_fwd, INSIDE_TURN_FORWARD_DURATION))
                                    if elapsed_fwd < INSIDE_TURN_FORWARD_DURATION:
                                        twist.angular.z = 0  # go straight
                                    else:
                                        self.inside_forward_active = False
                                        self.inside_steer_active = True
                                        self.inside_steer_start = rospy.get_time()
                                        rospy.loginfo("[INSIDE_FWD] Done → starting forced steer for %.1fs" % INSIDE_TURN_MIN_STEER_DURATION)
                                        twist.angular.z = twist.linear.x * math.tan(steer_angle) / 0.213
                                else:
                                    twist.angular.z = twist.linear.x * math.tan(steer_angle) / 0.213
                            elif not in_turn:
                                # ---- STRAIGHT ----
                                self.count_turn = 0
                                if not self.was_below_threshold:
                                    self.was_below_threshold = True
                                    rospy.loginfo("[FOLLOW] First time below threshold — turn detection armed")

                                # If inside forward is active, keep going straight — but check timer
                                if is_inside and self.inside_forward_active:
                                    elapsed_fwd = rospy.get_time() - self.inside_forward_start
                                    if elapsed_fwd < INSIDE_TURN_FORWARD_DURATION:
                                        rospy.loginfo_throttle(0.5, "[INSIDE_FWD] lane_x below threshold, forward %.1f/%.1fs — continuing straight" % (elapsed_fwd, INSIDE_TURN_FORWARD_DURATION))
                                        twist.angular.z = 0
                                    else:
                                        # Forward done even though lane_x is below threshold → start steer
                                        self.inside_forward_active = False
                                        self.inside_steer_active = True
                                        self.inside_steer_start = rospy.get_time()
                                        rospy.loginfo("[INSIDE_FWD] Timer done (%.1fs) while lane_x low → forced steer for %.1fs" % (elapsed_fwd, INSIDE_TURN_MIN_STEER_DURATION))
                                        twist.angular.z = twist.linear.x * math.tan(steer_angle) / 0.213
                                elif (rospy.get_time() - self.start_turn_time_stamp > TURN_END_TIMEOUT
                                        and self.start_turn):
                                    self.start_turn = False
                                    self.inside_forward_active = False
                                    if rospy.get_time() - self.last_turn_completed_time > TURN_COOLDOWN:
                                        rospy.loginfo("[TURN_END] Step %d complete (%s/%s) → advance" % (
                                            self.current_step, follow_side, turn_action))
                                        self._advance_step()
                                elif not self.start_turn:
                                    self.pid.SetPoint = setpoint
                                    self.pid.update(lane_x)
                                    twist.angular.z = twist.linear.x * math.tan(
                                        misc.set_range(self.pid.output, -0.1, 0.1)) / 0.213
                                else:
                                    twist.angular.z = twist.linear.x * math.tan(steer_angle) / 0.213

                            self.acker_pub.publish(twist)
                        else:
                            self.pid.clear()

                # Draw detected objects
                if self.objects_info != []:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box, result_image, color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf))
            else:
                rospy.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.dispaly:
                cv2.imshow('result', bgr_image)
                cv2.waitKey(1)
            self.result_publisher.publish(cv2_image2ros(bgr_image))
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.acker_pub.publish(Twist())

    def get_object_callback(self, msg):
        self.objects_info = msg.objects

if __name__ == "__main__":
    SelfDrivingSeqNode('self_driving')

