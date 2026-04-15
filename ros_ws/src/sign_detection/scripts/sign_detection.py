#!/usr/bin/env python
import rospy
import rosbag
import tf
import numpy as np
import math
import cv2
import ultralytics
import os
from sensor_msgs.msg import CompressedImage, PointCloud2, Image, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from laser_geometry import LaserProjection
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics import YOLO
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String
import roslib

class sign_detection():

    def __init__(self):
        rospy.init_node('sign_detector', anonymous=True)

        self.path = roslib.packages.get_pkg_dir("sign_detection")
        self.model_path = self.path + "/weights/best.pt"
        
        self.bridge = CvBridge()

        self.model = YOLO(self.model_path, task="detect")

        self.sub = rospy.Subscriber('/robot_1/depth_cam/rgb/image', Image, self.callback)
        self.pub = rospy.Publisher('/sign_detection', String, queue_size=1)
        self.pub_debug = rospy.Publisher('/sign_detection_debug', Image, queue_size=10)


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            height, width, _ = cv_image.shape
            
            roi_top = int(height * 0.25)
            roi_bottom = int(height)
            roi_width = int(width)
            roi_left = int(0)
            roi_right = int(roi_left + roi_width)
            image = cv_image[roi_top:roi_bottom, roi_left:roi_right]
    
            results = self.model(image)
            
            maxi = -1

            final_label = None

            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0].item())  # Extract confidence score
                    if confidence >= 0.80:  # Confidence threshold check
                        label = self.model.names[int(box.cls[0].item())]  # Get label

                        x1, y1, x2, y2 = box.xyxy[0] 
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        width  = x2 - x1
                        height = y2 - y1

                        size = height * width
                        if maxi < size:
                            maxi = size
                            final_label = label
                annotated_image = result.plot()
                ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8") 
                self.pub_debug.publish(ros_image)

            if final_label is not None:
                self.pub.publish(final_label)
        except Exception as e:
            rospy.loginfo(e)

    
    # def process_signs(self, detected_signs):
    #     cmd = Twist() 
    #     if "Stop" in detected_signs:
    #         rospy.loginfo("STOP sign detected! Stopping the robot.")
    #         cmd.linear.x = 0.0
    #         cmd.angular.z = 0.0  

    #     elif "SPEED LIMIT 50" in detected_signs:
    #         rospy.loginfo("Speed Limit 50 detected! Slowing down.")
    #         cmd.linear.x = 0.5  # Slow speed

    #     elif "NO ENTRY" in detected_signs:
    #         rospy.loginfo("No Entry sign detected! Stopping immediately.")
    #         cmd.linear.x = 0.0  # Stop robot

    #     else:
    #         rospy.loginfo("No special sign detected, continuing at normal speed.")
    #         cmd.linear.x = 1.0  # Default speed

if __name__ == '__main__':
    try:
        cv2.destroyAllWindows()
        proc = sign_detection()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
    except Exception as e:
        rospy.logerr("Unhandled exception")
