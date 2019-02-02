import cv2
import rospy

import numpy as np

from styx_msgs.msg import TrafficLight
from light_classification.tl_classifier import TLClassifier


@TLClassifier.register_subclass('simple-cnn')
class SimpleCNNTLClassifier(TLClassifier):
    """
    Classifies an image based on the traffic lights
    """

    def get_state_count_threshold(self, last_state):
        return 3

    def _classify(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        target_image = misc.imresize(image, (300, 400))
        target_image = target_image / 255.
        target_image = target_image.reshape(1, 300, 400, 3)
        
        with self.graph.as_default():
            encoded_label = self.loaded_model.predict_proba(target_image)
            label_idx = np.argmax(encoded_label)
            light = self.labels_dict[label_idx]
            
            print(light)
            if light == 'Red':
                return TrafficLight.RED, None
            elif light == 'Yellow':
                return TrafficLight.YELLOW, None
            elif light == 'Green':
                return TrafficLight.GREEN, None

        return TrafficLight.UNKNOWN, None

    def __init__(self, is_debug):
        super(SimpleCNNTLClassifier, self).__init__(self.__class__.__name__, is_debug)

        # model path
        self.package_root_path = rospkg.RosPack().get_path('tl_detector')
        model_json_path = os.path.join(self.package_root_path, 'models/simple-cnn.json')
        model_weights_path = os.path.join(self.package_root_path, 'models/simple-cnn.h5')

        self.labels_dict = {0: 'Red', 1: 'Yellow', 2: 'Green', 3: 'Unknown'}

        # load json and create model
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        self.loaded_model.load_weights(model_weights_path)
        print("Loaded model from disk")
        self.graph = tf.get_default_graph()

        #compile loaded model
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
