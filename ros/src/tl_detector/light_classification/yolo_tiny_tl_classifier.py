import os
import rospy
import rospkg

import numpy as np

from keras.layers import Input
from yolo3.model import tiny_yolo_body, yolo_eval, _get_anchors
from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


@TLClassifier.register_subclass("yolo-tiny")
class YOLOTinyTLClassifier(TLClassifier):

    def get_state_count_threshold(self, last_state):
        if last_state == TrafficLight.RED:
            # High threshold for accelerating
            return 3

        # Low threshold for stopping
        return 1

    @staticmethod
    def load_model(model_weights, num_anchors, num_classes):
        """creates the model and loads weights"""
        yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
        yolo_model.load_weights(model_weights)
        rospy.logdebug("YOLO model created and weights loaded")

        return yolo_model

    def _classify(self, image):

        # image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        # Actual detection
        print("Input image shape: ", image.shape)
        target_image = np.resize(image, (608, 608, 3))
        print("Target image shape: ", target_image.shape)

        (boxes, scores, classes) = yolo_eval(self.detection_graph.output, self.anchors, self.num_classes,
                                             target_image.shape, score_threshold=0.4, iou_threshold=0.5)
        # Remove unnecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        for i, clazz in enumerate(classes):
            rospy.logdebug('class = %s, score = %s', self.labels_dict[classes[i]], str(scores[i]))
            # if red or yellow light with confidence more than 10%
            if (clazz == 1 or clazz == 2) and scores[i] > 0.5:
                return TrafficLight.RED

        return TrafficLight.UNKNOWN

    def __init__(self):
        super(YOLOTinyTLClassifier, self).__init__(self.__class__.__name__)

        self.num_anchors = 6
        self.num_classes = 3
        # Model path
        package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_weights_path = os.path.join(package_root_path, 'models/tiny-yolo.h5')
        anchors_path = os.path.join(package_root_path, 'light_classification/yolo3/anchors.txt')
        self.anchors = _get_anchors(anchors_path)

        # Labels dictionary
        self.labels_dict = {0: 'Red', 1: 'Yellow', 2: 'Green'}

        # Create model and load weights of trained model
        self.detection_graph = self.load_model(model_weights_path, self.num_anchors, self.num_classes)
