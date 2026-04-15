#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class LaserScanNavigator:
    def __init__(self):
        rospy.init_node('laser_scan_navigator')
        
        # Stopping parameters
        self.safety_distance = 0.4
        self.safety_angle = 60
        self.override_duration = 1.0
        
        # Variables to keep track of events
        self.obstacle_detected = False
        self.last_obstacle_time = rospy.Time(0)
        self.last_cmd = Twist()
        
        # Subscribe to LIDAR scan
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback)

        self.pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=1)
        
        rospy.loginfo("Safety controller initialized. Monitoring for obstacles...")
        
    def scan_callback(self, scan_msg):
        """Process LIDAR scan data and detect obstacles"""
        
        ranges = []
        for i in range(0, self.safety_angle):
            index = ((360 - self.safety_angle // 2) + i) % 360
            if scan_msg.range_max >= scan_msg.ranges[index] >= scan_msg.range_min:
                ranges.append(scan_msg.ranges[index])

        ranges = np.array(ranges)
        
        if len(ranges) > 0:
            # Check if any valid range measurement is below safety distance
            min_distance = np.min(ranges)
            self.obstacle_detected = min_distance < self.safety_distance

            if self.obstacle_detected:
                self.last_obstacle_time = rospy.Time.now()
                #rospy.loginfo(f"Obstacle detected at {min_distance}m! Emergency braking activated.")
                if self.last_cmd.linear.x > 0:
                    safe_cmd = Twist()  # All zeros = stop
                else:
                    safe_cmd = self.last_cmd
                self.pub.publish(safe_cmd)
        else:
            # No valid readings, assume unsafe
            self.obstacle_detected = True
            
    def cmd_callback(self, cmd_msg):
        """Process incoming teleop commands and override if needed"""
        self.last_cmd = cmd_msg
        
        # Check if obstacle was recently detected
        time_since_obstacle = (rospy.Time.now() - self.last_obstacle_time).to_sec()
        override_active = (self.obstacle_detected or time_since_obstacle < self.override_duration)
        
        if override_active:
            # Override with stop command
            if cmd_msg.linear.x > 0:
                safe_cmd = Twist()  # All zeros = stop
            else:
                safe_cmd = cmd_msg
            rospy.loginfo_throttle(0.5, "Safety override active - stopping robot")
        else:
            # Pass through the teleop command
            safe_cmd = cmd_msg
        
        # Publish the safe command
        self.pub.publish(safe_cmd)

if __name__ == '__main__':
    try:
        laserScan = LaserScanNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
