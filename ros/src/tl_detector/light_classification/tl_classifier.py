import rospy
import sys
import threading
import numpy as np

from abc import ABCMeta, abstractmethod


class TLClassifier(object):
    """
    Base class for traffic light classifiers. The subclasses should provide implementations for the following methods:
        TLClassifier._classify(self, image)
        <TLClassifier-Subclass>.__init__(self)
    Note that <TLClassifier-Subclass>.__init__(self) must invoke its parent constructor
    and should not have input arguments except self.
    """

    __metaclass__ = ABCMeta

    INSTANCE = None
    KNOWN_TRAFFIC_LIGHT_CLASSIFIERS = {}  # it is not empty; it is filled by TLClassifier.register_subclass decorator

    @classmethod
    def register_subclass(cls, cls_id):
        """
        Decorator for TLClassifier subclasses.
        Adds decorated class to the cls.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS dictionary.
        :param cls_id: string identifier of the classifier
        :return: function object
        """
        def reg_subclass(cls_type):
            cls.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS[cls_id] = cls_type
            return cls_type
        return reg_subclass

    @classmethod
    def get_instance_of(cls, classifier_name):
        """
        It is a factory method for the `tl_classifier` module. It returns an instance of the classifier
        based on the input argument provided.
        :param classifier_name: name of the classifier
        :type classifier_name: str
        :return: instance of the classifier corresponding to the classifier string identifier
        :rtype: TLClassifier
        """
        if cls.INSTANCE is not None \
                and type(cls.INSTANCE) != cls.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS[classifier_name]:
            raise ValueError("cannot instantiate an instance of " + classifier_name
                             + " classifier since an instance of another type (" + type(cls.INSTANCE).__name__ +
                             ") has already been instantiated")

        if cls.INSTANCE is not None:
            return cls.INSTANCE

        classifier_type = cls.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS.get(classifier_name, None)
        if classifier_type is None:
            raise ValueError("classifier_name parameter has unknown value: " + classifier_name
                             + "; the value should be in " + str(cls.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS.keys()))
        cls.INSTANCE = classifier_type()

        return cls.INSTANCE

    @abstractmethod
    def _classify(self, image):
        """
        Determines the color of the traffic light in the image.
        This method should be implemented by a particular type of the traffic light classifier.

        :param image: image containing the traffic light
        :type image: np.ndarray
        :returns: ID of traffic light color (specified in styx_msgs/TrafficLight)
        :rtype: int
        """
        raise NotImplementedError()

    def classify(self, image):
        """
        Determines the color of the traffic light in the image.
        Prints FPS statistic approximately each second.

        :param image: image containing the traffic light
        :type image: np.ndarray
        :returns: ID of traffic light color (specified in styx_msgs/TrafficLight)
        :rtype: int
        """
        # calculate FPS based on the number of images processed per second;
        # ensure that self._counter value does not go over integer limits
        with self._lock:
            if self._start_time is None or self._counter > (sys.maxint - 100):
                self._start_time = rospy.get_time()
                self._counter = 0
            self._counter += 1
            # save start time and counter values for processing outside of the critical section
            start_t = self._start_time
            counter = self._counter

        tl_state = self._classify(image)

        # log the FPS no faster than once per second
        diff_t =  rospy.get_time() - start_t
        fps = None
        if diff_t >= 1.0:
            fps = int(counter / diff_t)
        if fps is not None:  # do not log while there are only a few images processed
            rospy.logdebug_throttle(1.0, "FPS: %d" % fps)

        return tl_state

    @abstractmethod
    def get_state_count_threshold(self, last_state):
        """
        Returns state count threshold value based on the last state.
        :param last_state: last traffic lights state
        :return: threshold value
        :rtype: int
        """
        raise NotImplementedError()

    @abstractmethod
    def __init__(self, cls_name):
        """
        Constructor is marked as @abstractmethod to force implemnting the __init__ method in subclasses.
        Subclasses must invoke their parent constructors.
        :param cls_name: string identifier of the subclass.
        """
        rospy.loginfo("instantiating %s (available classifiers: %s)",
                      cls_name, str(self.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS.keys()))

        self._lock = threading.Lock()  # lock to be used in TLClassifier.classify to increment invocation counter
        self._counter = 0
        self._start_time = None
