import os
import cv2
import rospy
import rospkg

import numpy as np

from keras.models import load_model
from keras.layers import Input
from keras import backend as K
from light_classification.yolo.model import tiny_yolo_body, yolo_eval
from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


@TLClassifier.register_subclass("yolo-tiny")
class YOLOTinyTLClassifier(TLClassifier):

    def get_state_count_threshold(self, last_state):
        if last_state == TrafficLight.RED:
            # High threshold for switching from RED state
            return 4
        elif last_state == TrafficLight.YELLOW:
            # in its current state the model is not very good at handling yellow traffic lights
            return 2

        # for switching from GREED and UNKNOWN the threshold is low; slow and steady wins the race
        return 1

    def _classify(self, image):
        orig_image_shape = image.shape
        input_image = self._prepare_input(image)

        with K.get_session().graph.as_default():

            boxes_tensor, scores_tensor, classes_tensor = \
                yolo_eval(self.yolo_model.output, self.anchors, self.num_classes, self.image_shape[0:2],
                          score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)

            out_boxes, out_scores, out_classes = \
                K.get_session().run([boxes_tensor, scores_tensor, classes_tensor],
                                    feed_dict={
                                        self.yolo_model.input: input_image,
                                        self.input_image_shape_tensor: orig_image_shape[0:2],
                                        K.learning_phase(): 0,
                                    })

            assert out_scores.shape == out_classes.shape

            # Remove unnecessary dimensions
            out_scores = out_scores.flatten()
            out_classes = out_classes.flatten()

            if out_scores.size > 0:
                clazz = out_classes[np.argmax(out_scores)]
                traffic_light = self.class_tl_map[clazz]
                traffic_light_name = self.class_classname_map[clazz]
            else:
                traffic_light = TrafficLight.UNKNOWN
                traffic_light_name = "unknown"

            rospy.logdebug("TL: %s; Classes-Scores: %s", traffic_light_name,
                           str([(self.class_classname_map[int(out_classes[i])], float(out_scores[i]))
                                for i in range(out_classes.size)]))

            return traffic_light

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
        """
        Reads YOLO class names, for traffic lights in this case.
        :param labels_path: path to file containing class names
        :type labels_path: str
        :return: list with string class names
        :rtype list
        """
        with open(labels_path) as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def _resize_and_pad(self, image):
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

    @staticmethod
    def _normalize(image):
        """
        Normalize image. Make image array of type float and make values to be in [0.0, 1.0]
        :param image: input image to normalize
        :type image: np.ndarray
        :return: normalized image
        :rtype np.ndarray
        """
        return image.astype(np.float32) / 255.0

    def _prepare_input(self, image):
        """
        Apply all the trafsformations to image to make it ready to feed to the YOLO-tiny network.
        :param image: image that comes to the classifier from the TLDetector
        :type image: np.ndarray
        :return: transformed and normalized image as 1 image per batch
        :rtype np.ndarray
        """
        image = self._resize_and_pad(image)
        image = self._normalize(image)

        # add batch dimension
        return np.expand_dims(image, 0)

    @staticmethod
    def _load_model(model_path, num_anchors, num_classes):
        """
        Load model from *.h5 file.
        :param model_path: path to the model checkpoint
        :type model_path: str
        :param num_anchors: number of anchors
        :type num_anchors: int
        :param num_classes: number of classes
        :type num_classes: int
        :return: YOLOv3-tiny model
        """
        try:
            yolo_model = load_model(model_path, compile=False)
        except (ImportError, ValueError):
            yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            yolo_model.load_weights(model_path) # make sure model, anchors and classes match
        else:
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        rospy.loginfo('%s model, anchors, and classes loaded.', model_path)
        return yolo_model

    def __init__(self):
        super(YOLOTinyTLClassifier, self).__init__(self.__class__.__name__)

        self.image_shape = (608, 608, 3)
        self.padding_color = (128, 128, 128)
        self.score_threshold = 0.1
        self.iou_threshold = 0.2

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
        self.class_tl_map = {0: TrafficLight.RED, 1: TrafficLight.YELLOW, 2: TrafficLight.GREEN}
        self.num_classes = len(self.class_classname_map.keys())
        assert self.num_classes == 3

        # Create model and load weights of trained model
        self.yolo_model = self._load_model(model_weights_path, self.num_anchors, self.num_classes)
        self.input_image_shape_tensor = K.placeholder(shape=(2,))
        self.boxes_tensor, self.scores_tensor, self.classes_tensor = \
            yolo_eval(self.yolo_model.output, self.anchors,
                      self.num_classes, self.input_image_shape_tensor,
                      score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)
