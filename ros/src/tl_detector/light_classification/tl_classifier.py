from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
from time import time
import rospy

class TLClassifier(object):
    def __init__(self, is_site, classifier):
        self.is_site = is_site
        self.classifier = classifier
        rospy.loginfo("Site : %s, classifier : %s",self.is_site,self.classifier)
        
        #DO loading of model etc if the classifier is dl based
        

    def simple_opencv_red_color_classifier(self,image):
        start = time()
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
        end = time()
        fps = 1.0 / (end - start)
        rospy.loginfo("Red count: %d , FPS: %.2f" % (red_count, fps))
        if red_count > 0:
            return TrafficLight.RED

        return TrafficLight.UNKNOWN

    def dl_ssd_classifier_carla(self,image):
        return TrafficLight.UNKNOWN
    
    def dl_ssd_classifier_sim(self,image):
        return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #call classifier for the site - real data
        if(self.is_site):
            if(self.classifier == "opencv"):
                return self.simple_opencv_red_color_classifier(image)
            elif(self.classifier == "dl_ssd"):
                self.dl_ssd_classifier_carla(image)

        #is_site is flase - call simulation classiier
        if(self.classifier == "opencv"):
            return self.simple_opencv_red_color_classifier(image)
        elif(self.classifier == "dl_ssd"):
            return self.dl_ssd_classifier_sim(image)

        #Default - use the simple opencv version if nothing is specified
        return self.simple_opencv_red_color_classifier(image)
        