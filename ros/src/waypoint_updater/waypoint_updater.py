#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

"""
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
"""

LOOKAHEAD_WPS = 200  # Number of waypoints to publish into /final_waypoints topic.

MAX_DECEL = 1.0

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.base_waypoints_msg = None
        self.pose_msg = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if not (None in (self.pose_msg, self.base_waypoints_msg, self.waypoints_2d, self.waypoint_tree)):
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_index(self):
        x = self.pose_msg.pose.position.x
        y = self.pose_msg.pose.position.y

        # The second argument to KDTree::query says that we only want to retrieve one item.
        # The query will return the position (tuple (x, y)) and its index.
        # The returned index matches the index of the returned position in the list
        # used to construct the KDTree. In this case, it is self.waypoints_2d list.
        cl_wp_idx = self.waypoint_tree.query((x, y), 1)[1]

        cl_wp = self.waypoints_2d[cl_wp_idx]
        # Highway and Test Lot tracks are circular, so,
        # it is not a problem to have a -1 for the previous waypoint for cl_wp_idx == 0.
        prev_wp = self.waypoints_2d[cl_wp_idx - 1]

        # Equation for hyperplane through cl_wp.
        cl_vect = np.array(cl_wp)
        prev_vect = np.array(prev_wp)
        pos_vect = np.array((x, y))

        # Check whether the closest waypoint is ahead or behind the vehicle.
        # If the closest waypoint is behind the vehicle, choose the next waypoint.
        # Again, notice that both Highway and Test Lot tracks are circular.
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            cl_wp_idx = (cl_wp_idx + 1) % len(self.waypoints_2d)

        return cl_wp_idx

    def publish_waypoints(self):
        """
        Publishes to /final_waypoints topic.
        """
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_index()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints_msg.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []

        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # We subtract 2 below to stop front of the car in front of the stop line.
            # Otherwise, the stop lane will be at the center of the car.
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        """
        Callback function for /current_pose topic subscriber.
        :param msg: /current_pose message
        :type msg: PoseStamped
        """
        self.pose_msg = msg

    def waypoints_cb(self, msg):
        """
        Callback function for /base_waypoints topic subscriber.
        :param msg: /base_waypoints message
        :type msg: Lane
        """
        self.base_waypoints_msg = msg
        if self.waypoints_2d is None:
            self.waypoints_2d = [(wp.pose.pose.position.x, wp.pose.pose.position.y)
                                 for wp in self.base_waypoints_msg.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        """
        Callback function for /traffic_waypoint topic subscriber.
        :param msg: /traffic_waypoint message
        :type msg: Int32
        """
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
