import os
import rospy
import rospkg

import tensorflow as tf
import numpy as np

from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


@TLClassifier.register_subclass("ssd")
class SSDTLClassifier(TLClassifier):

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
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        # Actual detection
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        for i, clazz in enumerate(classes):
            rospy.logdebug('class = %s, score = %s', self.labels_dict[classes[i]], str(scores[i]))
            # if red or yellow light with confidence more than 10%
            if (clazz == 2 or clazz == 3) and scores[i] > 0.1:
                return TrafficLight.RED

        return TrafficLight.UNKNOWN

    def __init__(self):
        super(SSDTLClassifier, self).__init__(self.__class__.__name__)

        # Model path
        package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_path = os.path.join(package_root_path, 'models/ssd.pb')

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
