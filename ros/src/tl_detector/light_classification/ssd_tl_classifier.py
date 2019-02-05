import os
import rospy
import rospkg

import tensorflow as tf
import numpy as np
import cv2

from abc import ABCMeta, abstractmethod

from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


class SSDTLClassifier(TLClassifier):

    __metaclass__ = ABCMeta

    def get_state_count_threshold(self, last_state):
        if last_state == TrafficLight.RED:
            # High threshold for accelerating
            return 3

        # Low threshold for stopping
        return 1

    @staticmethod
    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return graph

    def _classify(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        # Actual detection
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        for i, clazz in enumerate(classes):
            rospy.logdebug('class = %s, score = %s', self.labels_dict[classes[i]], str(scores[i]))
            # if red or yellow light with score more than confidence threshold
            if (clazz == 2) and scores[i] > self.confidence:
                return TrafficLight.RED, None
            if (clazz == 3) and scores[i] > self.confidence:
                return TrafficLight.YELLOW, None

        return TrafficLight.UNKNOWN, None

    @abstractmethod
    def __init__(self, is_debug, model_path, confidence):
        super(SSDTLClassifier, self).__init__(self.__class__.__name__, is_debug)

        # Model path
        package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_path = os.path.join(package_root_path, model_path)

        # Set confidence
        self.confidence = confidence

        # Labels dictionary
        self.labels_dict = {1: 'Green', 2: 'Red', 3: 'Yellow', 4: 'Unknown'}

        # Load frozen graph of trained model
        self.detection_graph = self.load_graph(model_path)

        # Get tensors
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Create session
        self.sess = tf.Session(graph=self.detection_graph)


@TLClassifier.register_subclass('ssd-sim')
class SSDSimTLClassifier(SSDTLClassifier):

    def __init__(self, is_debug):
        super(SSDSimTLClassifier, self).__init__(is_debug, 'models/ssd-sim.pb', 0.8)


@TLClassifier.register_subclass('ssd-real')
class SSDRealTLClassifier(SSDTLClassifier):

    def __init__(self, is_debug):
        super(SSDRealTLClassifier, self).__init__(is_debug, 'models/ssd-real.pb', 0.5)
