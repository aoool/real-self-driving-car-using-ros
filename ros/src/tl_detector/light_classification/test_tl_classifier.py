import rospy
from unittest import TestCase
from light_classification.tl_classifier import TLClassifier
from light_classification.ssd_tl_classifier import SSDTLClassifier
from light_classification.yolo_tiny_tl_classifier import YOLOTinyTLClassifier
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
        TLClassifier.register_subclass("ssd")(SSDTLClassifier)
        TLClassifier.register_subclass("yolo-tiny")(YOLOTinyTLClassifier)

    def tearDown(self):
        TLClassifier.INSTANCE = None
        TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS = {}
        TLClassifier.register_subclass("opencv")(OpenCVTLClassifier)
        TLClassifier.register_subclass("ssd")(SSDTLClassifier)
        TLClassifier.register_subclass("yolo-tiny")(YOLOTinyTLClassifier)

    def test_get_instance_of(self):
        instance = TLClassifier.get_instance_of("opencv")
        self.assertIsInstance(instance, OpenCVTLClassifier)
        self.assertEqual(3, len(TLClassifier.KNOWN_TRAFFIC_LIGHT_CLASSIFIERS))

    def test_classify(self):

        @TLClassifier.register_subclass('mock')
        class MockTLClassifier(TLClassifier):
            def __init__(self):
                super(MockTLClassifier, self).__init__(self.__class__.__name__)

            def _classify(self, image):
                pass

            def get_state_count_threshold(self, last_state):
                pass

        rospy.init_node('test_tl_classifier', anonymous=True)
        mock_instance = TLClassifier.get_instance_of('mock')
        for i in range(100):
            mock_instance.classify(None)
        self.assertEqual(100, mock_instance._counter)
