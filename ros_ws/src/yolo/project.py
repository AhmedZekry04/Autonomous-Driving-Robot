#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import String  # 可用于发布检测类别
import os
import time

class TrafficSignDetection:
    def __init__(self):
        rospy.init_node('traffic_sign_detection', anonymous=True)

        # 参数设置
        self.image_topic = rospy.get_param('~image_topic', '/rgbd_camera/rgb/image_raw')
        self.publish_topic = rospy.get_param('~publish_topic', '/hiwonder/image_detections')
        self.result_topic = rospy.get_param('~result_topic', '/traffic_signs')
        self.model_path = rospy.get_param('~model_path', '/home/developer/workspace/src/lab_2/weights/best.pt')
        self.fps_limit = rospy.get_param('~fps', 3)  # 每秒最多处理多少帧

        # 加载模型
        if not os.path.exists(self.model_path):
            rospy.logerr("模型文件未找到！请检查路径。")
            raise FileNotFoundError(self.model_path)

        self.model = YOLO(self.model_path)
        self.model.fuse()  # 加速推理
        rospy.loginfo("YOLO 模型加载完成")

        # 初始化变量
        self.bridge = CvBridge()
        self.last_time = time.time()

        # ROS订阅/发布
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher(self.publish_topic, Image, queue_size=1)
        self.detection_pub = rospy.Publisher(self.result_topic, String, queue_size=10)

        rospy.loginfo("交通标志检测节点启动")

    def image_callback(self, msg):
        # 控制处理帧率
        if time.time() - self.last_time < 1.0 / self.fps_limit:
            return
        self.last_time = time.time()

        try:
            # 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"图像转换失败: {e}")
            return

        try:
            results = self.model.predict(cv_image, conf=0.5, verbose=False)
        except Exception as e:
            rospy.logerr(f"模型推理失败: {e}")
            return

        # 处理检测结果
        detection_result = ""
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box)
                class_id = int(classes[i])
                score = scores[i]

                label = f"{self.model.names[class_id]} {score:.2f}"
                detection_result = self.model.names[class_id]

                # 画框
                cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 发布检测图像
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr(f"图像发布失败: {e}")

        # 发布检测结果（类别）
        if detection_result != "":
            self.detection_pub.publish(detection_result)
            rospy.loginfo(f"检测到标志: {detection_result}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = TrafficSignDetection()
        node.run()
    except rospy.ROSInterruptException:
        pass
