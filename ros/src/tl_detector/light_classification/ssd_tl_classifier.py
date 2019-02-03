import os
import rospy
import rospkg

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

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

    def _get_debug_image(self, image, boxes, classes, scores):
        """
        Draws bounding boxes with class labels and scores on an image.
        :param image: input image that come from the TLDetector
        :type image: np.ndarray
        :param out_boxes: bounding boxes predicted by SSD
        :type out_boxes: np.ndarray
        :param out_classes: classes predicted by SSD
        :type out_classes: np.ndarray
        :param out_scores: scores predicted by SSD
        :type out_scores: np.ndarray
        :return: image where bounding boxes are drawn
        :rtype np.ndarray
        """
        if len(scores) == 0:
            return image
        pil_image = Image.fromarray(image)
        width, height = pil_image.size

        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        font = ImageFont.truetype(font=os.path.join(self.package_root_path,'config/FiraMono-Medium.otf'),
                                  size=np.floor(3e-2 * pil_image.size[1] + 0.5).astype('int32'))
        thickness = (pil_image.size[0] + pil_image.size[1]) // 300

        i = np.argmax(scores)
        score = scores[i]
        c = classes[i]
        predicted_class = self.labels_dict[c]
        box = box_coords[i]

        draw = ImageDraw.Draw(pil_image)

        label = '{} {:.2f}'.format(predicted_class, score)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(pil_image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(pil_image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for j in range(thickness):
            draw.rectangle(
                [left + j, top + j, right - j, bottom - j],
                outline=self.labels_dict[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=self.labels_dict[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        return np.asarray(pil_image)

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
        image = cv2.resize(image, (300, 300))
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        # Actual detection
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= self.confidence:
                idxs.append(i)

        boxes = boxes[idxs, ...]
        scores = scores[idxs, ...]
        classes = classes[idxs, ...]

        for i, c in enumerate(classes):
            rospy.logdebug('class = %s, score = %s', self.labels_dict[c], str(scores[i]))

        if len(scores) == 0:
            traffic_light = TrafficLight.UNKNOWN
        else:
            i = np.argmax(scores)
            if classes[i] == 2:
                traffic_light = TrafficLight.RED
            elif classes[i] == 3:
                traffic_light = TrafficLight.YELLOW
            elif classes[i] == 1:
                traffic_light = TrafficLight.GREEN
            else:
                traffic_light = TrafficLight.UNKNOWN

        if self.is_debug:
            # create a debug image with bounding boxes and labels
            debug_image = self._get_debug_image(image, boxes, classes, scores)
            return traffic_light, debug_image

        return traffic_light, None

    @abstractmethod
    def __init__(self, is_debug, model_path, confidence):
        super(SSDTLClassifier, self).__init__(self.__class__.__name__, is_debug)

        # Model path
        self.package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_path = os.path.join(self.package_root_path, model_path)

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
