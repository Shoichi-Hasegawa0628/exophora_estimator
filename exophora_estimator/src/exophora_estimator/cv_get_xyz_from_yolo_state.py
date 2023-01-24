#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from tf import TransformListener
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import time
import numpy as np
import statistics
import math

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array


class GetDepthPointFromFrame():

    def __init__(self, rgb_image="/hsrb/head_rgbd_sensor/rgb/image_raw/compressed", point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points"):
    
        self.sum = []
        self.x = []
        self.y = []
        self.z = []


        self.rgb_image_topic_name = rgb_image
        self.point_cloud_name = point_cloud_topic
        self.cv_bridge = CvBridge()

        # self.tflistener = TransformListener()
        self.setup_subscriber()




    def setup_subscriber(self):
        self.subscriber_for_point_cloud = rospy.Subscriber(
            self.point_cloud_name,
            PointCloud2,
            self.point_cloud_callback,
            queue_size=1
        )

        # self.subscriber_for_bounding_box = rospy.Subscriber(
        #     self.bounding_topic_name,
        #     BoundingBoxes,
        #     self.bounding_callback,
        #     queue_size=1
        # )

        self.subscriber_for_rgb_image = rospy.Subscriber(
            self.rgb_image_topic_name,
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
        return


    def image_callback(self, message):
        rospy.loginfo("Entered callback")
        rgb_img = self.cv_bridge.compressed_imgmsg_to_cv2(message)
        height, width, channels = rgb_img.shape[:3]

        # 画像の中心 (ここをmediapiepの出力)
        cx = width / 2
        cy = height / 2
        print(cx, cy)

        _pose_stamped = PoseStamped()

        _position = self.get_point(int(cx), int(cy))
        rospy.loginfo("X:{} Y:{} Z:{}".format(_position[0], _position[1], _position[2]))

        if not (_position is None):
            pass
            # tf_buffer = tf2_ros.Buffer()
            # tf_listener = tf2_ros.TransformListener(tf_buffer)
            # try:
            #     self.trans = tf_buffer.lookup_transform('map', 'head_rgbd_sensor_rgb_frame', rospy.Time.now(), rospy.Duration(1.0))
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #     return

        _pose_stamped.header = self._point_cloud_header
        _pose_stamped.pose.position.x = _position[0]
        _pose_stamped.pose.position.y = _position[1]
        _pose_stamped.pose.position.z = _position[2]
        rospy.loginfo("cX:{:.3f} cY:{:.3f} cZ:{:.3f}".format(_pose_stamped.pose.position.x, _pose_stamped.pose.position.y,_pose_stamped.pose.position.z))


        # transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
        # rospy.loginfo("mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x,
        #                                                   transformed.pose.position.y,
        #                                                   transformed.pose.position.z))

        

        # self.x.append(_pose_stamped.pose.position.x)
        # self.y.append(_pose_stamped.pose.position.y)
        # self.z.append(_pose_stamped.pose.position.z)

        # median_x = statistics.median(self.x)
        # median_y = statistics.median(self.y)
        # median_z = statistics.median(self.z)

        # rospy.loginfo("medX:{:.3f} medY:{:.3f} medZ:{:.3f}".format(median_x, median_y, median_z))



    def point_cloud_callback(self, point_cloud):
        self._point_cloud = pointcloud2_to_xyz_array(point_cloud, False)
        # rospy.loginfo("PointCloud frame id : {}".format(point_cloud.header.frame_id))
        self._point_cloud_header = point_cloud.header

    def get_point(self, x, y):
        try:
            return self._point_cloud[y][x]
        except:
            # rospy.loginfo("GET POINT ERROR")
            pass



if __name__ == '__main__':
    rospy.init_node('get_depth_from_rgb_image')
    get_depth_from_rgb = GetDepthPointFromFrame()
    rospy.spin()
