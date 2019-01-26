import os
import cv2
import rospy
import rospkg

import numpy as np

from keras.layers import Input
from light_classification.yolo.model import tiny_yolo_body, yolo_eval
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
        transformed_image = self._transform_image(image)

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

    @staticmethod
    def _get_anchors(anchors_path):
        """
        Reads YOLO anchors from the configuration file.
        :param anchors_path: path to the configuration file.
        :type path: str
        :return: anchors read from the config file
        :rtype np.ndarray
        """
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def _get_class_names(labels_path):
        with open(labels_path) as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def _transform_image(self, image):
        """
        Applies some image transformations to prepare it for feeding to the trained model.
        The same image transformations were applied to images during the training phase.
        Transformations:
            - resize image to fit the target height and width preserving the original aspect ratio
            - pad the resized image borders to make its shape equal to the target shape, i.e., `self.image_shape`
        :param image: input image from the YOLOTinyTLClassifier._classify
        :type image: np.ndarray
        :return: transformed image
        :rtype np.ndarray
        """
        # original images are comming in BGR8 format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ih, iw, _ = image.shape       # input image height and width
        oh, ow, _ = self.image_shape  # output image height and width

        # input and output image aspect ratios
        iar, oar = float(ih) / iw, float(oh) / ow

        # scaling coefficient
        k = float(oh) / ih if iar > oar else float(ow) / iw

        # resize image preserving input aspect ratio and fitting into target shape
        image = cv2.resize(image, (int(round(k * iw)), int(round(k * ih))),
                           interpolation=cv2.INTER_CUBIC)

        h, w, _ = image.shape  # height and width of the resized image

        # calculate padding for each border to make the resized image
        # equal in shape to the target shape
        h_pad, w_pad = oh - h, ow - w
        top_pad, bottom_pad = h_pad // 2, h_pad // 2 + h_pad % 2
        left_pad, right_pad = w_pad // 2, w_pad // 2 + w_pad % 2

        # do padding to make the resized image equal in shape to the target shape
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad,
                                   cv2.BORDER_CONSTANT, value=self.padding_color)

        assert image.shape == self.image_shape, \
            "the prepared image shape " + str(image.shape) \
            + " does not equal the target image shape" + str(self.image_shape)

        return image

    def __init__(self):
        super(YOLOTinyTLClassifier, self).__init__(self.__class__.__name__)

        self.image_shape = (608, 608, 3)
        self.padding_color = (128, 128, 128)

        # Model path
        package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_weights_path = os.path.join(package_root_path, 'models/yolo-tiny.h5')

        # Anchors
        anchors_path = os.path.join(package_root_path, 'config/tiny_yolo_anchors.txt')
        self.anchors = self._get_anchors(anchors_path)
        self.num_anchors = self.anchors.shape[0]
        assert self.num_anchors == 6

        # Classes
        labels_path = os.path.join(package_root_path, 'config/traffic_lights_classes.txt')
        self.class_classname_map = {num_id: str_id for num_id, str_id in enumerate(self._get_class_names(labels_path))}
        self.num_classes = len(self.class_classname_map.keys())
        assert self.num_classes == 3

        # Create model and load weights of trained model
        self.detection_graph = self.load_model(model_weights_path, self.num_anchors, self.num_classes)
