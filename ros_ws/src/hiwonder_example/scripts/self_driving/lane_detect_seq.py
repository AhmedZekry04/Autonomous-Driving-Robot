#!/usr/bin/env python3
# encoding: utf-8
# Lane detection for self_driving_seq.py
# Changes vs original:
#   - HSV instead of LAB (better ambient light rejection)
#   - center_x includes col offset for correct lane_x in right ROI mode
import cv2
import math
import numpy as np

class LaneDetector(object):
    def __init__(self, color):
        self.target_color = color
        self.rois = ((450, 480, 0, 320, 0.7), (390, 420, 0, 320, 0.2), (330, 360, 0, 320, 0.1))
        self.weight_sum = 1.0

    def set_roi(self, roi):
        self.rois = roi

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None

    def get_binary(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        img_blur = cv2.GaussianBlur(img_hsv, (3, 3), 3)
        # H=20-35 (yellow hue), S=200-255 (high saturation = reject ambient light), V=80-255
        mask = cv2.inRange(img_blur, (10, 80, 50), (40, 255, 255))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return dilated

    def __call__(self, image, result_image):
        centroid_sum = 0
        h, w = image.shape[:2]
        max_center_x = -1
        center_x = []
        for roi in self.rois:
            blob = image[roi[0]:roi[1], roi[2]:roi[3]]
            contours = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            max_contour_area = self.get_area_max_contour(contours, 30)
            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])
                box = np.int0(cv2.boxPoints(rect))
                for j in range(4):
                    box[j, 1] = box[j, 1] + roi[0]
                cv2.drawContours(result_image, [box], -1, (255, 255, 0), 2)

                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2

                cv2.circle(result_image, (int(line_center_x + roi[2]), int(line_center_y)), 5, (0, 0, 255), -1)
                center_x.append(line_center_x + roi[2])  # col offset for correct image coordinates
            else:
                center_x.append(-1)
        for i in range(len(center_x)):
            if center_x[i] != -1:
                if center_x[i] > max_center_x:
                    max_center_x = center_x[i]
                centroid_sum += center_x[i] * self.rois[i][-1]
        if centroid_sum == 0:
            return result_image, None, max_center_x
        center_pos = centroid_sum / self.weight_sum
        angle = math.degrees(-math.atan((center_pos - (w / 2.0)) / (h / 2.0)))

        return result_image, angle, max_center_x

