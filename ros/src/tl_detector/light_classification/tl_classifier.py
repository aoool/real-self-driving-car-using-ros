from styx_msgs.msg import TrafficLight
from keras.models import model_from_json
from scipy import misc
import numpy as np
import rospkg
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.label_dict = {'red':0, 'yellow':1, 'green':2, 'other':3}
	self.label_dict_reverse = dict((v,k) for k,v in self.label_dict.iteritems())

	r = rospkg.RosPack()
	path = r.get_path('tl_detector')

        # load json and create model
	json_file = open(path+'/light_classification/model/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
	self.loaded_model.load_weights(path+"/light_classification/model/model.h5")
	print("Loaded model from disk")
	self.graph = tf.get_default_graph()

        #compile loaded model
	self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        target_image = misc.imresize(image, (300, 400))
        target_image = target_image / 255.
	target_image = target_image.reshape(1, 300, 400, 3)
	print(target_image.shape)
	with self.graph.as_default():
	    encoded_label = self.loaded_model.predict_proba(target_image)
	    label_idx = np.argmax(encoded_label)
	    light = self.label_dict_reverse[label_idx]
        
	    print(light)
	    if light == 'red':
                return TrafficLight.RED
            elif light == 'yellow':
                return TrafficLight.YELLOW
            elif light == 'green':
                return TrafficLight.GREEN
            else:
 	        return TrafficLight.UNKNOWN
