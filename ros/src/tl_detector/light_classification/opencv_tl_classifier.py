import cv2
import rospy

import numpy as np

from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


@TLClassifier.register_subclass('opencv')
class OpenCVTLClassifier(TLClassifier):
    """
    Detects and classifies traffic lights on images with Computer Vision techniques.
    """

    def get_state_count_threshold(self, last_state):
        return 3

    def _classify(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_img = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)

        im, contours, hierarchy = cv2.findContours(red_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        red_count = 0
        for x, contour in enumerate(contours):
            contourarea = cv2.contourArea(contour)  # get area of contour
            if 18 < contourarea < 900:  # Discard contours with a too large area as this may just be noise
                arclength = cv2.arcLength(contour, True)
                approxcontour = cv2.approxPolyDP(contour, 0.01 * arclength, True)
                # Check for Square
                if len(approxcontour) > 5:
                    red_count += 1
        rospy.logdebug("Red count: %d", red_count)
        if red_count > 0:
            return TrafficLight.RED

        return TrafficLight.UNKNOWN

    def __init__(self):
        super(OpenCVTLClassifier, self).__init__(self.__class__.__name__)
