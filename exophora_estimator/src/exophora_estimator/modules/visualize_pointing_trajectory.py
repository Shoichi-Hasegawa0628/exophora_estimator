#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

class VisualizePointingTrajectory():
    def __init__(self):
        self.point_pub = rospy.Publisher("point_pub", Marker, queue_size = 1)
        self.eye_pub = rospy.Publisher("eye_pub", Marker, queue_size = 1)
        self.vector_pub = rospy.Publisher("vector_pub", MarkerArray, queue_size = 1)

    # 始点 (ポースランドマークモデル上の人の目の3次元座標)
    def visualize_eye(self, eye_frame):
        self.marker_msg = Marker()
        # for i in range(len(x_list)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        # Object type, 矢印やったら0
        marker.type = 2
        marker.action = Marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = eye_frame[0]
        marker.pose.position.y = eye_frame[1]
        marker.pose.position.z = eye_frame[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.frame_locked = False
        marker.lifetime = rospy.Duration(40)
        # marker.ns = "Goal-%u"%i
        # self.marker_msg.markers.append(marker)
        # self.marker_msg.lifetime = rospy.Duration(10)
        # self.point_pub.publish(self.marker_msg)
        self.eye_pub.publish(marker)
        # self.rate.sleep()

    # 終点 (地面との交点)
    def visualize_point(self, point_ground):
        self.marker_msg = Marker()
        # for i in range(len(x_list)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        # Object type, 矢印やったら0
        marker.type = 2
        marker.action = Marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = point_ground[0]
        marker.pose.position.y = point_ground[1]
        marker.pose.position.z = point_ground[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.frame_locked = False
        marker.lifetime = rospy.Duration(40)
        # marker.ns = "Goal-%u"%i
        # self.marker_msg.markers.append(marker)
        # self.marker_msg.lifetime = rospy.Duration(10)
        # self.point_pub.publish(self.marker_msg)
        self.point_pub.publish(marker)
        # self.rate.sleep()

    # 始点から終点までの間の点 (媒介変数を用いてプロット)
    def visualize_pointing(self, param, wrist_frame, eys_frame):
        self.marker_array_msg = MarkerArray()
        for t in range(param):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = t
            # Object type, 矢印やったら0
            marker.type = 2
            marker.action = Marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = (1 - t) * eys_frame[0] + t * wrist_frame[0]
            marker.pose.position.y = (1 - t) * eys_frame[1] + t * wrist_frame[1]
            marker.pose.position.z = (1 - t) * eys_frame[2] + t * wrist_frame[2]
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.8
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.frame_locked = False
            marker.lifetime = rospy.Duration(40)
            marker.ns = "Goal-%u" % t
            self.marker_array_msg.markers.append(marker)
        # self.marker_array_msg.lifetime = rospy.Duration(10)
        self.vector_pub.publish(self.marker_array_msg)
        # self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('visualize_pointing_trajectory')
    visualize_pointing_trajectory = VisualizePointingTrajectory()
    # rospy.spin()

