#!/usr/bin/env python3
"""
YOLOv4-tiny sign detector using OpenCV DNN with ONNX model.
Publishes ObjectsInfo on /yolov5/object_detect so self_driving.py works unchanged.

Dependencies (all pre-installed on the Jetson):
    cv2, numpy, rospy, cv_bridge, hiwonder_interfaces

No ultralytics, no pycuda, no TensorRT needed.
"""

import cv2
import numpy as np
import os
import rospy
import roslib
from sensor_msgs.msg import Image
from hiwonder_interfaces.msg import ObjectInfo, ObjectsInfo


PKG_DIR = roslib.packages.get_pkg_dir("sign_detection")
DEFAULT_ONNX = os.path.join(PKG_DIR, "weights", "yolov4_tiny_best.onnx")
DEFAULT_NAMES = os.path.join(PKG_DIR, "weights", "obj.names")

INPUT_SIZE = 416
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

# YOLOv4-tiny anchors (must match training config)
ANCHORS = [
    [(81, 82), (135, 169), (344, 319)],   # det_large, stride 32, 13x13
    [(10, 14), (23, 27), (37, 58)],        # det_small, stride 16, 26x26
]
STRIDES = [32, 16]

# Map our training class names -> self_driving.py expected names
# self_driving.py uses: 'go', 'right', 'park', 'red', 'green', 'crosswalk'
CLASS_MAP = {
    "no_left":               None,         # no direct mapping
    "no_right":              None,         # no direct mapping
    "parking":               "park",
    "speed_limit_5":         "go",         # speed limit = keep going
    "speed_limit_lift":      "go",
    "stop":                  "crosswalk",  # stop sign treated like crosswalk (stop behavior)
    "traffic_light":         None,         # generic, ignore
    "traffic_light_green":   "green",
    "traffic_light_red":     "red",
    "traffic_light_yellow":  "red",        # yellow = treat as red (stop)
    "turn_right":            "right",
}


class YOLOv4TinyDetector:
    def __init__(self, onnx_path, names_path, conf_thresh, nms_thresh):
        # Load class names
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.classes)

        # Load ONNX model
        rospy.loginfo("[YOLOv4] Loading model: {}".format(onnx_path))
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        # Try CUDA backend (Jetson), fallback to CPU
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Test with a dummy forward pass
            dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            self.net.setInput(dummy)
            self.net.forward(self.net.getUnconnectedOutLayersNames())
            rospy.loginfo("[YOLOv4] Using CUDA backend")
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            rospy.logwarn("[YOLOv4] CUDA not available, using CPU (may be slow)")

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        rospy.loginfo("[YOLOv4] Loaded {} classes: {}".format(
            self.num_classes, self.classes))

    def detect(self, frame):
        """
        Run detection on a BGR frame.
        Returns list of dicts: {class_name, mapped_name, confidence, box: [x1,y1,x2,y2]}
        Box coordinates are in original image pixel space.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        output_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_names)

        all_boxes = []
        all_confidences = []
        all_class_ids = []

        for raw_output, anchors, stride in zip(outputs, ANCHORS, STRIDES):
            self._decode_output(raw_output, anchors, stride,
                                all_boxes, all_confidences, all_class_ids)

        # Scale boxes from INPUT_SIZE space to original image space
        sx = w / float(INPUT_SIZE)
        sy = h / float(INPUT_SIZE)

        results = []
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences,
                                       self.conf_thresh, self.nms_thresh)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw, bh = all_boxes[i]
                    # Convert from (x, y, w, h) top-left to (x1, y1, x2, y2)
                    x1 = int(x * sx)
                    y1 = int(y * sy)
                    x2 = int((x + bw) * sx)
                    y2 = int((y + bh) * sy)

                    class_name = self.classes[all_class_ids[i]]
                    mapped = CLASS_MAP.get(class_name)

                    results.append({
                        "class_name": class_name,
                        "mapped_name": mapped,
                        "confidence": all_confidences[i],
                        "box": [x1, y1, x2, y2],
                    })
        return results

    def _decode_output(self, raw_output, anchors, stride,
                       all_boxes, all_confidences, all_class_ids):
        """Decode raw ONNX output (no sigmoid baked in) into boxes."""
        # raw_output shape: (1, num_anchors*(5+nc), H, W)
        na = len(anchors)
        nc = self.num_classes
        _, _, H, W = raw_output.shape

        # Reshape to (na, 5+nc, H, W)
        pred = raw_output.reshape(na, 5 + nc, H, W)

        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        for a in range(na):
            # Decode: sigmoid(tx) + grid_offset, sigmoid(ty) + grid_offset
            cx = (1.0 / (1.0 + np.exp(-pred[a, 0])) + grid_x) * stride
            cy = (1.0 / (1.0 + np.exp(-pred[a, 1])) + grid_y) * stride
            # exp(tw) * anchor_w, exp(th) * anchor_h
            bw = np.exp(pred[a, 2]) * anchors[a][0]
            bh = np.exp(pred[a, 3]) * anchors[a][1]
            # Objectness sigmoid
            obj = 1.0 / (1.0 + np.exp(-pred[a, 4]))
            # Class scores sigmoid
            cls_scores = 1.0 / (1.0 + np.exp(-pred[a, 5:]))  # (nc, H, W)

            for j in range(H):
                for i in range(W):
                    obj_conf = obj[j, i]
                    if obj_conf < self.conf_thresh * 0.5:
                        continue

                    cls_conf = cls_scores[:, j, i]
                    cls_id = np.argmax(cls_conf)
                    score = float(obj_conf * cls_conf[cls_id])

                    if score < self.conf_thresh:
                        continue

                    # Box as (x_topleft, y_topleft, width, height) in INPUT_SIZE space
                    x = int(cx[j, i] - bw[j, i] / 2)
                    y = int(cy[j, i] - bh[j, i] / 2)
                    w = int(bw[j, i])
                    h = int(bh[j, i])

                    all_boxes.append([x, y, w, h])
                    all_confidences.append(score)
                    all_class_ids.append(int(cls_id))


class SignDetectorNode:
    def __init__(self):
        rospy.init_node("yolov4_sign_detector", anonymous=True)

        # Parameters
        onnx = rospy.get_param("~onnx", DEFAULT_ONNX)
        names = rospy.get_param("~names", DEFAULT_NAMES)
        conf = rospy.get_param("~confidence", CONF_THRESHOLD)
        nms = rospy.get_param("~nms_thresh", NMS_THRESHOLD)
        image_topic = rospy.get_param("~image_topic",
                                      "/robot_1/depth_cam/rgb/image_raw")

        self.detector = YOLOv4TinyDetector(onnx, names, conf, nms)

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

        # Build ObjectsInfo message
        objects_msg = ObjectsInfo()
        objects_list = []

        for det in detections:
            mapped = det["mapped_name"]
            if mapped is None:
                continue  # Skip classes that don't map to driving actions

            obj = ObjectInfo()
            obj.class_name = mapped
            obj.box = det["box"]  # [x1, y1, x2, y2]
            obj.score = det["confidence"]
            obj.width = w
            obj.height = h
            objects_list.append(obj)

            rospy.logdebug("[YOLOv4] {} -> {} ({:.2f})".format(
                det["class_name"], mapped, det["confidence"]))

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
            mapped = det["mapped_name"] or det["class_name"]
            label = "{} {:.2f}".format(mapped, det["confidence"])
            color = (0, 255, 0) if det["mapped_name"] else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


if __name__ == "__main__":
    try:
        node = SignDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

