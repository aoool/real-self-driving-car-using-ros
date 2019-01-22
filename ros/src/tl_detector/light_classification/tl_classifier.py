from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

import tensorflow as tf
from helpers import load_graph

import rospkg
import os.path as path

# Model path
r = rospkg.RosPack()
base_path = r.get_path('tl_detector')
SSD_INCEPTION_SIM = path.join(base_path,'light_classification/models/ssd_inception_v2_coco_retrained_sim/frozen_inference_graph.pb')

# Labels dictionary
labels_dict = {1: 'Green', 2: 'Red', 3: 'Yellow',4: 'Unknown'}

class TLClassifier(object):
    def __init__(self):
	# Load frozen graph of trained model
	self.detection_graph = load_graph(SSD_INCEPTION_SIM)

	# Get tensors
	self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
	self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
	self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
	self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

	# Create session
	self.sess = tf.Session(graph=self.detection_graph)

    def simple_opencv_red_color_classifier(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255,255])
        lower_red2 = np.array([160,100,100])
        upper_red2 = np.array([179,255,255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_img = cv2.addWeighted(mask1,1.0,mask2,1.0,0)

        im, contours, hierarchy = cv2.findContours(red_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        red_count = 0
        for x,contour in enumerate(contours):
            contourarea = cv2.contourArea(contour) #get area of contour
            if 18 < contourarea < 900: #Discard contours with a too large area as this may just be noise
                arclength = cv2.arcLength(contour, True)
                approxcontour = cv2.approxPolyDP(contour, 0.01 * arclength, True)
                #Check for Square
                if len(approxcontour)>5:
                    red_count += 1

        if red_count > 0:
            return TrafficLight.RED

        return TrafficLight.UNKNOWN

    def dl_based_classifier(self, image):
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
                     
        # Actual detection
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        for i in range(len(classes)):
	    print('class =', labels_dict[classes[i]])
	    print('score =', scores[i])
            if (classes[i] == 2 or classes[i] == 3) and scores[i] > 0.1: # if red or yellow light with confidence more than 10%
                return TrafficLight.RED

        return TrafficLight.UNKNOWN
    
    def carla_real_data_classifier(self,image):
        return TrafficLight.UNKNOWN

    def get_classification(self, image, method=None):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if(method == "opencv"):
            return self.simple_opencv_red_color_classifier(image)
        elif(method == "carla"):
            return self.carla_real_data_classifier(image)
        
        return self.dl_based_classifier(image)
        
