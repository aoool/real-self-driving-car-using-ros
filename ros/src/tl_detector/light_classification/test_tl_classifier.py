import rospy
from unittest import TestCase
from light_classification.tl_classifier import TLClassifier, OpenCVTrafficLightsClassifier
from roslaunch.parent import ROSLaunchParent


class TLClassifierTest(TestCase):

    ROS_LAUNCH_PARENT = ROSLaunchParent("TLClassifierTest", [], is_core=True)

    @classmethod
    def setUpClass(cls):
        """TODO (Sergey Morozov): remove and use rostest or rosunit"""
        # start roscore
        cls.ROS_LAUNCH_PARENT.start()

    @classmethod
    def tearDownClass(cls):
        """TODO (Sergey Morozov): remove and use rostest or rosunit"""
        # stor roscore
        cls.ROS_LAUNCH_PARENT.shutdown()

    def setUp(self):
        TLClassifier.INSTANCE = None
        TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS = {}
        TLClassifier.register_subclass("opencv")(OpenCVTrafficLightsClassifier)

    def tearDown(self):
        TLClassifier.INSTANCE = None
        TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS = {}
        TLClassifier.register_subclass("opencv")(OpenCVTrafficLightsClassifier)

    def test_get_instance_of(self):
        instance = TLClassifier.get_instance_of("opencv")
        self.assertIsInstance(instance, OpenCVTrafficLightsClassifier)
        self.assertEqual(1, len(TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS))

    def test_classify(self):

        @TLClassifier.register_subclass('mock')
        class MockTLClassifier(TLClassifier):
            def __init__(self):
                super(MockTLClassifier, self).__init__(self.__class__.__name__)

            def _classify(self, image):
                pass

        rospy.init_node('test_tl_classifier', anonymous=True)
        mock_instance = TLClassifier.get_instance_of('mock')
        for i in range(100):
            mock_instance.classify(None)
        self.assertEqual(100, mock_instance._counter)
