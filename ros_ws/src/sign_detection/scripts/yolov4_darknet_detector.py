#!/usr/bin/env python3
"""
YOLOv4-tiny sign detector using OpenCV DNN with Darknet .cfg + .weights.
Publishes ObjectsInfo on /yolov5/object_detect so self_driving.py works unchanged.

Dependencies (all pre-installed on the Jetson):
    cv2 (4.5.5 with CUDA), numpy, rospy, hiwonder_interfaces

No CvBridge, no ultralytics, no pycuda, no TensorRT needed.
"""

import cv2
import numpy as np
import os
import rospy
import roslib
from sensor_msgs.msg import Image
from hiwonder_interfaces.msg import ObjectInfo, ObjectsInfo


PKG_DIR = roslib.packages.get_pkg_dir("sign_detection")
DEFAULT_CFG = os.path.join(PKG_DIR, "weights", "yolov4-tiny-custom-traffic-2.cfg")
DEFAULT_WEIGHTS = os.path.join(PKG_DIR, "weights", "yolov4-tiny-custom-traffic_last-3.weights")

INPUT_SIZE = 416
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

# Class names must match the order used during training
CLASS_NAMES = [
    'No_left', 'no_right', 'parking', 'speed_limit_5',
    'speed_limit_lift', 'stop', 'green', 'no_light',
    'red', 'yellow', 'no', 'turn_right',
]

# Only remap yellow -> red; everything else keeps its original name
CLASS_REMAP = {
    "yellow": "red",
}


class YOLOv4TinyDetector:
    def __init__(self, cfg_path, weights_path, conf_thresh, nms_thresh):
        self.classes = CLASS_NAMES
        self.num_classes = len(self.classes)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Load Darknet model
        rospy.loginfo("[YOLOv4] Loading model: {}".format(cfg_path))
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

        # Try CUDA backend (Jetson), fallback to CPU
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Test with a dummy forward pass
            dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            self.net.setInput(dummy)
            self.net.forward(self._get_output_names())
            rospy.loginfo("[YOLOv4] Using CUDA backend")
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            rospy.logwarn("[YOLOv4] CUDA not available, using CPU (may be slow)")

        rospy.loginfo("[YOLOv4] Loaded {} classes: {}".format(
            self.num_classes, self.classes))

    def _get_output_names(self):
        """Cache and return the names of the output layers."""
        if not hasattr(self, '_output_names'):
            layer_names = self.net.getLayerNames()
            out_indices = self.net.getUnconnectedOutLayers()
            if isinstance(out_indices[0], (list, np.ndarray)):
                self._output_names = [layer_names[i[0] - 1] for i in out_indices]
            else:
                self._output_names = [layer_names[i - 1] for i in out_indices]
        return self._output_names

    def detect(self, frame):
        """
        Run detection on a BGR frame.
        Returns list of dicts: {class_name, confidence, box: [x1,y1,x2,y2]}
        Box coordinates are in original image pixel space.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self._get_output_names())

        # Darknet output is already decoded:
        # each row = [cx, cy, w, h, obj_conf, class_scores...]
        all_boxes = []
        all_confidences = []
        all_class_ids = []

        for out in outputs:
            for det in out:
                scores = det[5:]
                scores = scores[:self.num_classes]
                if len(scores) == 0:
                    continue
                c_id = int(np.argmax(scores))
                conf = float(scores[c_id])
                if conf > self.conf_thresh:
                    cx = det[0] * w
                    cy = det[1] * h
                    bw = det[2] * w
                    bh = det[3] * h
                    x = int(cx - bw / 2)
                    y = int(cy - bh / 2)
                    all_boxes.append([x, y, int(bw), int(bh)])
                    all_confidences.append(conf)
                    all_class_ids.append(c_id)

        results = []
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences,
                                       self.conf_thresh, self.nms_thresh)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw, bh = all_boxes[i]
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)

                    raw_name = self.classes[all_class_ids[i]]
                    class_name = CLASS_REMAP.get(raw_name, raw_name)

                    results.append({
                        "class_name": class_name,
                        "confidence": all_confidences[i],
                        "box": [x1, y1, x2, y2],
                    })
        return results


class SignDetectorNode:
    def __init__(self):
        rospy.init_node("yolov4_sign_detector", anonymous=True)

        # Parameters
        cfg = rospy.get_param("~cfg", DEFAULT_CFG)
        weights = rospy.get_param("~weights", DEFAULT_WEIGHTS)
        conf = rospy.get_param("~confidence", CONF_THRESHOLD)
        nms = rospy.get_param("~nms_thresh", NMS_THRESHOLD)
        image_topic = rospy.get_param("~image_topic",
                                      "/robot_1/depth_cam/rgb/image_raw")

        self.detector = YOLOv4TinyDetector(cfg, weights, conf, nms)

        # Publishers
        self.object_pub = rospy.Publisher(
            "/yolov5/object_detect", ObjectsInfo, queue_size=1)
        self.image_pub = rospy.Publisher(
            "/yolov5/object_image", Image, queue_size=1)

        # Subscriber
        rospy.Subscriber(image_topic, Image, self.callback,
                         queue_size=1, buff_size=2**24)
        rospy.loginfo("[YOLOv4] Subscribed to {}".format(image_topic))
        rospy.loginfo("[YOLOv4] Publishing ObjectsInfo on /yolov5/object_detect")

    def callback(self, msg):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logerr("[YOLOv4] image conversion error: {}".format(e))
            return

        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)

        # Build ObjectsInfo message — publish ALL classes
        objects_msg = ObjectsInfo()
        objects_list = []

        for det in detections:
            obj = ObjectInfo()
            obj.class_name = det["class_name"]
            obj.box = det["box"]  # [x1, y1, x2, y2]
            obj.score = det["confidence"]
            obj.width = w
            obj.height = h
            objects_list.append(obj)

            rospy.logdebug("[YOLOv4] {} ({:.2f})".format(
                det["class_name"], det["confidence"]))

        objects_msg.objects = objects_list
        self.object_pub.publish(objects_msg)

        # Publish annotated image for debugging
        if self.image_pub.get_num_connections() > 0:
            annotated = self._draw(frame, detections)
            img_msg = Image()
            img_msg.header = msg.header
            img_msg.height = annotated.shape[0]
            img_msg.width = annotated.shape[1]
            img_msg.encoding = "bgr8"
            img_msg.step = annotated.shape[1] * 3
            img_msg.data = annotated.tobytes()
            self.image_pub.publish(img_msg)

    def _draw(self, frame, detections):
        """Draw bounding boxes on frame for debug visualization."""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = "{} {:.2f}".format(det["class_name"], det["confidence"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


if __name__ == "__main__":
    try:
        node = SignDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
