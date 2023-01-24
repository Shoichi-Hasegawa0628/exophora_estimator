#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import csv
import tf
from tf import TransformListener
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge
import time
import numpy as np
import statistics
import math
from visualization_msgs.msg import Marker, MarkerArray
# 正規分布
from scipy.stats import multivariate_normal
# フォンミーゼス分布
from scipy.stats import vonmises

from PIL import Image
import matplotlib.pyplot as plt

# 下4つはmediapipe用
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
import rviz_gaussian_distribution_msgs.msg as rgd_msgs
from matplotlib import colors


class ExophoraEstimator():
    def __init__(self, rgb_img_topic="/hsrb/head_rgbd_sensor/rgb/image_raw/compressed",
                 point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
                 global_pose_topic = "/global_pose"):

        self._object_class_p = []
        self._demonstrative_p = []
        self._pointing_p = []
        self._wrist_x = []
        self._wrist_y = []
        self._wrist_z = []
        self._eye_x = []
        self._eye_y = []
        self._eye_z = []
        self._object_category = ["Bottle", "Stuffed Toy", "Book", "Cup"]
        self._point_ground = np.array([0, 0, 0])
        self._kosoa = "a"

        self.rgb_image_topic_name = rgb_img_topic
        self.point_cloud_name = point_cloud_topic
        self.global_pose_topic_name = global_pose_topic
        self.cv_bridge = CvBridge()

        # tf
        self._point_cloud_header = Header()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tflistener = TransformListener()

        # MediaPipe用
        # self.mp_pose = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mesh_drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
        self.mark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=3, circle_radius=1, color=(0, 0, 255))
        # self.sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed", CompressedImage, self.main)

        self.img_pub = rospy.Publisher("/annotated_msg/compressed", CompressedImage, queue_size=1)
        self.point_pub = rospy.Publisher("point_pub", Marker, queue_size=1)
        self.eye_pub = rospy.Publisher("eye_pub", Marker, queue_size=1)
        self.vector_pub = rospy.Publisher("vector_pub", MarkerArray, queue_size=1)
        self.distribution_publisher = rospy.Publisher("/gaussian_distribution/input/add", rgd_msgs.GaussianDistribution,
                                                      queue_size=1)


    ### non_realtimeとrealtimeで存在する関数と存在しないものがあるため、両方を参照しながらプログラムを作成
    # 基本的には、
    def main(self):
        self.setup_subscriber()

        # 物体のインデックスとカテゴリと3次元座標のリスト作成
        # リストの中身は [object number, object name, x, y, z] が20個
        object_list = self.load_data()

        # リストに6個の座標が格納されたらループを抜ける
        while len(self._eye_x) < 7:
            self.landmark_frame()

        rospy.loginfo("Skeleton could be detected!")

        # 指差しベクトルの計算と可視化
        point_ground, param = self.pointing_vector(wrist_x, wrist_y, wrist_z, eye_x, eye_y, eye_z)
        self.visualize_point()
        self.visualize_eye()
        self.visualize_pointing()

        # 指差し方向に基づく確率推定器により、対象確率の出力
        # for i in range(len(object_position)):
        self.pointing_inner_product(object_list)

        # 指示語領域に基づく確率推定器により、対象確率の出力
        self.kosoa(object_list)
        # self.visualize_kosoa()

        # 物体カテゴリの信頼度スコアを取得 (物体カテゴリに基づく確率推定器)
        #####
        #####


        # 3つの確率値を掛け合わせる
        # target_probability = self._object_class_p * self._demonstrative_p * self._visual_indicate_p

        demonstrative_p = np.array(self._demonstrative_p)
        pointing_p = np.array(self._pointing_p)

        target_probability = demonstrative_p * pointing_p
        sum_target_probability = np.sum(target_probability)

        # 正規化
        target_probability = target_probability / sum_target_probability
        target = np.amax(target_probability)
        target_index = np.argmax(target_probability)

        # print用に小数点第3位を四捨五入
        target_probability = np.round(target_probability, decimals=2)
        # print(object_list)
        target_probability_reshape = np.reshape(target_probability, (4, 5))

        k = 0
        for i in range(4):
            print("対象確率", target_probability[k:k + 5])
            k += 5
        print("予測した目標物体", object_list[target_index])
        # demonstrative_p = np.round(demonstrative_p, decimals=2)
        # print(demonstrative_p)

    def setup_subscriber(self):
        self.subscriber_for_point_cloud = rospy.Subscriber(
            self.point_cloud_name,
            PointCloud2,
            self.point_cloud_callback,
            queue_size=1
        )

        self.subscriber_for_rgb_image = rospy.Subscriber(
            self.rgb_image_topic_name,
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
        self.subscriber_for_global_pose = rospy.Subscriber(
            self.global_pose_topic_name,
            PoseStamped,
            self.global_pose_callback,
            queue_size=1
        )
        return

    def image_callback(self, msg):
        # rospy.loginfo("Entered callback")
        self.rgb_img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)

    def point_cloud_callback(self, msg):
        self._point_cloud = pointcloud2_to_xyz_array(msg, False)
        # rospy.loginfo("PointCloud frame id : {}".format(msg.header.frame_id))
        self._point_cloud_header = msg.header

    def global_pose_callback(self, msg):
        self._global_pose_x = msg.pose.position.x
        self._global_pose_y = msg.pose.position.y
        self._global_pose_z = msg.pose.position.z


if __name__ == '__main__':
    rospy.init_node('exophora_estimator')
    node = ExophoraEstimator()
    node.main()
    # rospy.spin()

