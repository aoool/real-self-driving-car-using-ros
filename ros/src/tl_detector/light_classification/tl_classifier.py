import rospy
import collections

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
        :type image: numpy.ndarray
        :returns: ID of traffic light color (specified in styx_msgs/TrafficLight)
        :rtype: int
        """
        raise NotImplementedError()

    def classify(self, image):
        """
        Determines the color of the traffic light in the image.
        Prints FPS statistic approximately each second.

        :param image: image containing the traffic light; image is in BGR8 encoding!
        :type image: numpy.ndarray
        :returns: ID of traffic light color (specified in styx_msgs/TrafficLight)
        :rtype: int
        """
        # append the start time to the circular buffer
        self._start_time_circular_buffer.append(rospy.get_time())

        tl_state = self._classify(image)

        # log the FPS no faster than once per second
        fps = len(self._start_time_circular_buffer) / (rospy.get_time() - self._start_time_circular_buffer[0])
        rospy.logdebug_throttle(1.0, "FPS: %.3f" % fps)

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
        Constructor is marked as @abstractmethod to force implementing the __init__ method in subclasses.
        Subclasses must invoke their parent constructors.
        :param cls_name: string identifier of the subclass.
        """
        rospy.loginfo("instantiating %s (available classifiers: %s)",
                      cls_name, str(self.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS.keys()))

        # circular buffer for storing start time of classifications
        # addition to/from beginning/end is O(1); thread safe
        # used to calculate FPS (moving average)
        # Once a bounded length deque is full, when new items are added,
        # a corresponding number of items are discarded from the opposite end.
        self._start_time_circular_buffer = collections.deque(maxlen=100)
