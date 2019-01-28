import os
import cv2
import rospy
import rospkg

import numpy as np

from PIL import Image, ImageFont, ImageDraw
from .keras.models import load_model
from .keras.layers import Input
from .keras import backend as K
from light_classification.yolo.model import tiny_yolo_body, yolo_eval
from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


@TLClassifier.register_subclass("yolo-tiny")
class YOLOTinyTLClassifier(TLClassifier):

    def get_state_count_threshold(self, last_state):
        return 3

    def _get_debug_image(self, image, out_boxes, out_classes, out_scores):
        """
        Draws bounding boxes with class labels and scores on an image.
        :param image: input image that come from the TLDetector
        :type image: np.ndarray
        :param out_boxes: bounding boxes predicted by YOLO-tiny
        :type out_boxes: np.ndarray
        :param out_classes: classes predicted by YOLO-tiny
        :type out_classes: np.ndarray
        :param out_scores: scores predicted by YOLO-tiny
        :type out_scores: np.ndarray
        :return: image where bounding boxes are drawn
        :rtype np.ndarray
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype(font=os.path.join(self.package_root_path,'config/FiraMono-Medium.otf'),
                                  size=np.floor(3e-2 * pil_image.size[1] + 0.5).astype('int32'))
        thickness = (pil_image.size[0] + pil_image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_classname_map[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(pil_image)
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
                    outline=self.class_color_map[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.class_color_map[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return np.asarray(pil_image)

    def _classify(self, image):
        orig_image_shape = image.shape
        input_image = self._prepare_input(image)

        with K.get_session().graph.as_default():

            feed_dict = {
                self.yolo_model.input: input_image,
                self.input_image_shape_tensor: orig_image_shape[0:2],
                K.learning_phase(): 0,
            }

            if self.is_debug:
                out_boxes, out_scores, out_classes = \
                    K.get_session().run([self.boxes_tensor, self.scores_tensor, self.classes_tensor],
                                        feed_dict=feed_dict)
            else:
                out_scores, out_classes = \
                    K.get_session().run([self.scores_tensor, self.classes_tensor], feed_dict=feed_dict)
                out_boxes = None  # needed here just for completeness

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

            if self.is_debug:
                # create a debug image with bounding boxes and labels
                if out_classes.size > 1:
                    ind = np.argmax(out_scores)
                    out_boxes = np.asarray([out_boxes[ind]])
                    out_scores = np.asarray([out_scores[ind]])
                    out_classes = np.asarray([out_classes[ind]])
                debug_image = self._get_debug_image(image, out_boxes, out_classes, out_scores)
                return traffic_light, debug_image

            return traffic_light, None

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

    def __init__(self, is_debug):
        super(YOLOTinyTLClassifier, self).__init__(self.__class__.__name__, is_debug)

        self.image_shape = (608, 608, 3)
        self.padding_color = (128, 128, 128)
        self.score_threshold = 0.1
        self.iou_threshold = 0.2

        red_color_rgb = (255, 0, 0)
        yellow_color_rgb = (255, 255, 0)
        green_color_rgb = (0, 255, 0)


        # Model path
        self.package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_weights_path = os.path.join(self.package_root_path, 'models/yolo-tiny.h5')

        # Anchors
        anchors_path = os.path.join(self.package_root_path, 'config/tiny_yolo_anchors.txt')
        self.anchors = self._get_anchors(anchors_path)
        self.num_anchors = self.anchors.shape[0]
        assert self.num_anchors == 6

        # Classes
        labels_path = os.path.join(self.package_root_path, 'config/traffic_lights_classes.txt')
        self.class_classname_map = {num_id: str_id for num_id, str_id in enumerate(self._get_class_names(labels_path))}
        self.class_tl_map = {0: TrafficLight.RED, 1: TrafficLight.YELLOW, 2: TrafficLight.GREEN}
        self.class_color_map = {0: red_color_rgb, 1: yellow_color_rgb, 2: green_color_rgb}
        self.num_classes = len(self.class_classname_map.keys())
        assert self.num_classes == 3

        # Create model and load weights of trained model
        self.yolo_model = self._load_model(model_weights_path, self.num_anchors, self.num_classes)
        self.input_image_shape_tensor = K.placeholder(shape=(2,))
        self.boxes_tensor, self.scores_tensor, self.classes_tensor = \
            yolo_eval(self.yolo_model.output, self.anchors, self.num_classes, self.input_image_shape_tensor,
                      score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)
