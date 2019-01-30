import rospy
from unittest import TestCase
from light_classification.tl_classifier import TLClassifier
from light_classification.ssd_tl_classifier import SSDSimTLClassifier
from light_classification.yolo.yolo_tiny_tl_classifier import YOLOTinyTLClassifier
from light_classification.opencv_tl_classifier import OpenCVTLClassifier
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
        # stop roscore
        cls.ROS_LAUNCH_PARENT.shutdown()

    def setUp(self):
        TLClassifier.INSTANCE = None
        TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS = {}
        TLClassifier.register_subclass("opencv")(OpenCVTLClassifier)
        TLClassifier.register_subclass("ssd-sim")(SSDSimTLClassifier)
        TLClassifier.register_subclass("yolo-tiny")(YOLOTinyTLClassifier)

    def tearDown(self):
        TLClassifier.INSTANCE = None
        TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS = {}
        TLClassifier.register_subclass("opencv")(OpenCVTLClassifier)
        TLClassifier.register_subclass("ssd-sim")(SSDSimTLClassifier)
        TLClassifier.register_subclass("yolo-tiny")(YOLOTinyTLClassifier)

    def test_get_instance_of(self):
        instance = TLClassifier.get_instance_of("opencv")
        self.assertIsInstance(instance, OpenCVTLClassifier)
        self.assertEqual(3, len(TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS))

    def test_classify(self):

        @TLClassifier.register_subclass('mock')
        class MockTLClassifier(TLClassifier):
            def __init__(self, is_debug):
                super(MockTLClassifier, self).__init__(self.__class__.__name__, is_debug)

            def _classify(self, image):
                return None, None

            def get_state_count_threshold(self, last_state):
                pass

        rospy.init_node('test_tl_classifier', anonymous=True)
        mock_instance = TLClassifier.get_instance_of('mock')
        for i in range(20):
            mock_instance.classify(None)
        self.assertEqual(20, len(mock_instance._start_time_circular_buffer))

        for i in range(200):
            mock_instance.classify(None)
        self.assertEqual(100, len(mock_instance._start_time_circular_buffer))
