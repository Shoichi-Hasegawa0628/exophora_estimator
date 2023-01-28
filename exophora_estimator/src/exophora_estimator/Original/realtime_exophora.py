#!/usr/bin/env python
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
# ガウス分布可視化のためのもの
import rviz_gaussian_distribution_msgs.msg as rgd_msgs
from matplotlib import colors


class Exophora():

    def __init__(self, rgb_image="/hsrb/head_rgbd_sensor/rgb/image_raw/compressed", point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points", global_pose_topic = "/global_pose"):

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

        self.rgb_image_topic_name = rgb_image
        self.point_cloud_name = point_cloud_topic
        self.global_pose_topic_name = global_pose_topic
        self.cv_bridge = CvBridge()

        # tfらへん
        self._point_cloud_header = Header()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # mediapipeのための追加部分
        #self.mp_pose = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mesh_drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
        self.mark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=3, circle_radius=1, color=(0, 0, 255))
        #self.sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed", CompressedImage, self.main)

        self.img_pub = rospy.Publisher("/annotated_msg/compressed", CompressedImage, queue_size=1)
        self.point_pub = rospy.Publisher("point_pub", Marker, queue_size = 1)
        self.eye_pub = rospy.Publisher("eye_pub", Marker, queue_size = 1)
        self.vector_pub = rospy.Publisher("vector_pub", MarkerArray, queue_size = 1)
        self.tflistener = TransformListener()

        self.distribution_publisher = rospy.Publisher("/gaussian_distribution/input/add", rgd_msgs.GaussianDistribution, queue_size=1)

        # こっから
        # self.setup_subscriber()
        # rospy.sleep(1)

        # # リストに6個の座標が格納されたらループを抜ける
        # while len(self._wrist_x) < 7:
        #     self.landmark_frame()

        # rospy.loginfo("Skeleton could be detected!!!!!!")

        # self.pointing_vector()
        # self.kosoa()


    def main(self):
        self.setup_subscriber()
        # rospy.sleep(1)

        # 物体のインデックスとカテゴリと3次元座標のリストを作成
        object_list = self.load_data()
        # リストの中身は [object number, object name, x, y, z] が20個
        # print(object_list)
        # print(len(object_position))

        # リストに6個の座標が格納されたらループを抜ける
        while len(self._eye_x) < 7:
            self.landmark_frame()

        rospy.loginfo("Skeleton could be detected!")

        # 指差しベクトル
        self.pointing_vector()
        self.visualize_point()
        self.visualize_eye()
        self.visualize_pointing()

        # 指差しから各物体の確率
        # for i in range(len(object_position)):
        self.pointing_inner_product(object_list)

        # こそあ領域、確率
        self.kosoa(object_list)
        # self.visualize_kosoa()

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
            print("対象確率", target_probability[k:k+5])
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

    def publish_distribution(self, id_, mu_k, sigma_k, i):
        # rgb_colorlist = [[255, 0, 0], [0, 255, 0], [0, 0, 255], []]
        # colorlist = ['red','green','blue','cyan','magenta','darkblue','orange','purple','yellowgreen','yellow','darkred']
        colorlist = ['red','green','blue']
        rgb_colorlist = list()
        for c in colorlist:
            rgb_colorlist.append(list(colors.hex2color(colors.cnames[c])))
        for c in range(len(rgb_colorlist)):
            for rgb in range(len(rgb_colorlist[c])):
                rgb_colorlist[c][rgb] = int(rgb_colorlist[c][rgb]*255)
        """
        Publish Gaussian distributions
        Args:
            id_ (int):
        """
        # Calc parameters of Gaussian distribution．
        mu_x = mu_k[0]
        mu_y = mu_k[1]
        # 分散を小さくしている
        sigma_x = sigma_k[0][0] * 0.2
        sigma_y = sigma_k[1][1] * 0.2
        covariance = sigma_k[0][1]
        r = rgb_colorlist[i][0]
        g = rgb_colorlist[i][1]
        b = rgb_colorlist[i][2]
        # Publish
        # self.wait_time(1)
        rospy.sleep(0.1)
        msg = rgd_msgs.GaussianDistribution(
            mean_x=mu_x, mean_y=mu_y,
            std_x=sigma_x, std_y=sigma_y,
            covariance=covariance,
            r=r, g=g, b=b,
            id=id_
        )
        self.distribution_publisher.publish(msg)
        
    def publish_clear(self):
        """
        Clear published distributions
        """
        publisher = rospy.Publisher("/gaussian_distribution/input/clear", std_msgs.Empty, queue_size=1)
        # rospy.sleep(0.1)
        # self.wait_time(1)
        msg = std_msgs.Empty()
        publisher.publish(msg)

    def load_data(self):
        files = os.listdir("/root/HSR/catkin_ws/src/mediapipe_ros/data/objects_position/")
        datas = []
        data = []
        for i in range(len(files)):
            with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/objects_position/object_position_{}.csv".format(str(i+1))) as f:
                reader = csv.reader(f)
                for row in reader:
                    data.append(row)
            datas.append(data)
            data = []
        number = 0
        for k in range(len(self._object_category)):
            for i in range(len(datas)):
                for j in range(len(datas[i])):
                    if(datas[i][j][1] == self._object_category[k]):
                        datas[i][j][0] = number
                        data.append(datas[i][j])
                        number += 1
            # print(datas)
        return data


    def landmark_frame(self):
        try:
            height, width, channels = self.rgb_img.shape[:3]
            # ここからmediapipe追加部分
            # 検出結果を別の画像名としてコピーして作成
            self.annotated_image = self.rgb_img.copy()
            with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                results = pose.process(self.rgb_img)
                # 画像にポーズのランドマークを描画
                # annotated_image = self.rgb_img.copy()
                # upper_body_onlyがTrueの時
                # 以下の描画にはmp_pose.UPPER_BODY_POSE_CONNECTIONSを使用
                # mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                # cv2.imwrite('../data/annotated_image_rgb.png', annotated_image)
                # annotated_msg = self.cv_bridge.cv2_to_compressed_imgmsg(annotated_image)
                # annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
                # self.img_pub.publish(annotated_msg)
                l_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * height
                r_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * height

            if l_wrist_y < r_wrist_y:
                wrist = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * height)
                eye = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * height)
            else:
                wrist = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * height)
                eye = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * height)

            wrist_x = int(wrist[0])
            wrist_y = int(wrist[1])
            eye_x = int(eye[0])
            eye_y = int(eye[1])

            _position_wrist = self.get_point(wrist_x, wrist_y)
            _position_eye = self.get_point(eye_x, eye_y)
            # if not ((_position_wrist[0] is None) and (_position_eye[0] is None)):
            # rospy.loginfo("X:{} Y:{} Z:{}".format(_position_wrist[0], _position_wrist[1], _position_wrist[2]))
        except Exception as e1:
            print(f"Unexpected {e1=}, {type(e1)=}")
            return

        # こっからtf
        if not (((_position_wrist is None) and (_position_eye is None)) or (math.isnan(_position_wrist[0]) is True) or (math.isnan(_position_eye[0]) is True)):
        # if not ((_position_wrist is None) and (_position_eye is None)):
            pass
            # tf_buffer = tf2_ros.Buffer()
            # tf_listener = tf2_ros.TransformListener(tf_buffer)
            try:
                # self.trans = self.tf_buffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time.now(), rospy.Duration(1.0))
                self.trans = self.tf_buffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time(0), rospy.Duration(1.0))
                _pose_stamped = PoseStamped()
                # 手首
                # print(self._point_cloud_header)
                # _pose_stamped.header = self._point_cloud_header
                _pose_stamped.header.frame_id = 'map'
                _pose_stamped.pose.position.x = _position_wrist[0]
                _pose_stamped.pose.position.y = _position_wrist[1]
                _pose_stamped.pose.position.z = _position_wrist[2]
                # tf
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                rospy.loginfo("wrist : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z))
                self._wrist_x.append(transformed.pose.position.x)
                self._wrist_y.append(transformed.pose.position.y)
                self._wrist_z.append(transformed.pose.position.z)
                # 肘
                _pose_stamped.pose.position.x = _position_eye[0]
                _pose_stamped.pose.position.y = _position_eye[1]
                _pose_stamped.pose.position.z = _position_eye[2]
                # tf
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                rospy.loginfo("eye : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z))
                self._eye_x.append(transformed.pose.position.x)
                self._eye_y.append(transformed.pose.position.y)
                self._eye_z.append(transformed.pose.position.z)
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            except Exception as e2:
                print(f"Unexpected {e2=}, {type(e2)=}")
                return
        else:
            print("Skeleton could not be detected.")

    def pointing_vector(self):
        # 手首の中央値
        self.median_wrist_x = statistics.median(self._wrist_x)
        self.median_wrist_y = statistics.median(self._wrist_y)
        self.median_wrist_z = statistics.median(self._wrist_z)
        rospy.loginfo("wrist_median \n cX:{} cY:{} cZ:{}".format(self.median_wrist_x, self.median_wrist_y, self.median_wrist_z))
        # 肘の中央値
        self.median_eye_x = statistics.median(self._eye_x)
        self.median_eye_y = statistics.median(self._eye_y)
        self.median_eye_z = statistics.median(self._eye_z)
        rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_eye_x, self.median_eye_y, self.median_eye_z))

        wrist_frame = np.array([self.median_wrist_x, self.median_wrist_y, self.median_wrist_z])
        eye_frame = np.array([self.median_eye_x, self.median_eye_y, self.median_eye_z])

        map_xlim = [-3.5, 3.3]
        map_ylim = [-7.6, 1]
        map_zlim = [0, 3]
        point_vector = np.array([0, 0, 0])
        # t = 1
        
        # 地面との交点
        u = eye_frame[2] / (eye_frame[2] - wrist_frame[2])
        point_ground_x = (1 - u) * eye_frame[0] + u * wrist_frame[0]
        point_ground_y = (1 - u) * eye_frame[1] + u * wrist_frame[1]
        self._point_ground = np.array([point_ground_x, point_ground_y, 0])
        self._t = int(u)
        # self.visualize_point()
        t = 1
        # 天井に当たったときに判定だけやってない
        if ((point_ground_x < map_xlim[0]) or (map_xlim[1] < point_ground_x) or (point_ground_y < map_ylim[0]) or (map_ylim[1] < point_ground_y)):
            while True:
                t += 1
                for i in range(3):
                    point_vector[i] = (1 - t) * eye_frame[i] + t * wrist_frame[i]
                if point_vector[0] < map_xlim[0] - 0.5:
                    x3 = map_xlim[0]
                    t = (eye_frame[0] - x3) / (eye_frame[0] - wrist_frame[0])
                    y3 = (1 - t) * eye_frame[1] + t * wrist_frame[1]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[0] > map_xlim[1] + 0.5:
                    x3 = map_xlim[1]
                    t = (eye_frame[0] - x3) / (eye_frame[0] - wrist_frame[0])
                    y3 = (1 - t) * eye_frame[1] + t * wrist_frame[1]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] < map_ylim[0] - 0.5:
                    y3 = map_ylim[0]
                    t = (eye_frame[1] - y3) / (eye_frame[1] - wrist_frame[1])
                    x3 = (1 - t) * eye_frame[0] + t * wrist_frame[0]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] > map_ylim[1] + 0.5:
                    y3 = map_ylim[1]
                    t = (eye_frame[1] - y3) / (eye_frame[1] - wrist_frame[1])
                    x3 = (1 - t) * eye_frame[0] + t * wrist_frame[0]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
            self._t = int(t)
                # # zだけ逆なことに注意
                # if point_vector[2] > map_zlim[0]:
                #     z3 = map_zlim[0]
                #     t = eye_frame[2] / (eye_frame[2] - wrist_frame[2])
                #     x3 = (1 - t) * eye_frame[0] + t * wrist_frame[0]
                #     y3 = (1 - t) * eye_frame[1] + t * wrist_frame[1]
                #     self._point_ground = np.array([x3, y3, z3])
                #     break

        # self.visualize_point()

        # # self.visualize_point()
        # # self._eye = np.array([eye_frame[0], eye_frame[1], eye_frame[2]])
        # self.visualize_eye()
        # self.visualize_pointing()
            

        # 地面との交点
        # t = eye_frame[2] / (eye_frame[2] - wrist_frame[2])
        # point_ground_x = (1 - t) * eye_frame[0] + t * wrist_frame[0]
        # point_ground_y = (1 - t) * eye_frame[1] + t * wrist_frame[1]
        # self._point_ground = np.array([point_ground_x, point_ground_y, 0])
        # self.visualize_point()

    def kosoa(self, position):

        # 手首の中央値
        self.median_wrist_x = statistics.median(self._wrist_x)
        self.median_wrist_y = statistics.median(self._wrist_y)
        self.median_wrist_z = statistics.median(self._wrist_z)
        # rospy.loginfo("wrist_median \n cX:{} cY:{} cZ:{}".format(self.median_wrist_x, self.median_wrist_y, self.median_wrist_z))
        # 目の中央値
        self.median_eye_x = statistics.median(self._eye_x)
        self.median_eye_y = statistics.median(self._eye_y)
        self.median_eye_z = statistics.median(self._eye_z)
        # rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_eye_x, self.median_eye_y, self.median_eye_z))

        wrist_frame = np.array([self.median_wrist_x, self.median_wrist_y, self.median_wrist_z])
        eye_frame = np.array([self.median_eye_x, self.median_eye_y, self.median_eye_z])
        robot_frame = np.array([self._global_pose_x, self._global_pose_y, self._global_pose_z])

        # 人とロボットの距離
        # dis_hr = (eye_frame[0] - robot_frame[0])**2 + (eye_frame[1] - robot_frame[1])**2 + (eye_frame[2] - robot_frame[2])**2
        dis_hr = (eye_frame[0] - robot_frame[0])**2 + (eye_frame[1] - robot_frame[1])**2 + (0 - robot_frame[2])**2
        self._distance = math.sqrt(dis_hr)

        # 目と肘間の長さ
        dis_ew = (eye_frame[0] - wrist_frame[0])**2 + (eye_frame[1] - wrist_frame[1])**2 + (eye_frame[2] - wrist_frame[2])**2
        self._length = math.sqrt(dis_ew)

        # 分散のパラメータ
        variance_alpha = 0.1
        variance_beta = 0.1

        # コ系列の平均と分散と可視化
        ko_gaussian_mean = wrist_frame
        ko_gaussian_variance = self._distance * variance_alpha / variance_beta
        self.publish_distribution(0, [ko_gaussian_mean[0], ko_gaussian_mean[1]], [[ko_gaussian_variance, 0], [0, ko_gaussian_variance]], 0)
        print(ko_gaussian_variance)
        # ソ系列の平均と分散と可視化
        so_gaussian_mean = (1 - variance_alpha) * robot_frame + variance_alpha * eye_frame
        so_gaussian_variance = self._distance * variance_alpha / variance_beta
        self.publish_distribution(1, [so_gaussian_mean[0], so_gaussian_mean[1]], [[so_gaussian_variance, 0], [0, so_gaussian_variance]], 1)
        # ア系列の平均と分散と可視化
        a = 2
        a_gaussian_mean = (1 - a * self._distance / self._length) * eye_frame + (a * self._distance / self._length) * wrist_frame
        a_gaussian_variance = self._distance * variance_alpha / variance_beta
        self.publish_distribution(2, [a_gaussian_mean[0], a_gaussian_mean[1]], [[a_gaussian_variance, 0], [0, a_gaussian_variance]], 2)

        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            if(self._kosoa == "ko"):
                # print("ko")
                # self._gaussian_mean = wrist_frame
                # self._gaussian_variance = self._distance * variance_alpha / variance_beta
                # 各物体に対してその物体の座標を入力して確率値を計算
                gaussian_dis = multivariate_normal([ko_gaussian_mean[0], ko_gaussian_mean[1], ko_gaussian_mean[2]], [[ko_gaussian_variance, 0, 0], [0, ko_gaussian_variance, 0], [0, 0, ko_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if(self._kosoa == "so"):
                # print("so")
                # self._gaussian_mean = (1 - variance_alpha) * robot_frame + variance_alpha * eye_frame
                # self._gaussian_variance = self._distance * variance_alpha / variance_beta
                # 各物体に対してその物体の座標を入力して確率値を計算
                gaussian_dis = multivariate_normal([so_gaussian_mean[0], so_gaussian_mean[1], so_gaussian_mean[2]], [[so_gaussian_variance, 0, 0], [0, so_gaussian_variance, 0], [0, 0, so_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if(self._kosoa == "a"):
                # print("a")
                # a = 2
                # self._gaussian_mean = (1 - a * self._distance / self._length) * eye_frame + (a * self._distance / self._length) * wrist_frame
                # self._gaussian_variance = self._distance * variance_alpha / variance_beta
                # 各物体に対してその物体の座標を入力して確率値を計算
                gaussian_dis = multivariate_normal([a_gaussian_mean[0], a_gaussian_mean[1], a_gaussian_mean[2]], [[a_gaussian_variance, 0, 0], [0, a_gaussian_variance, 0], [0, 0, a_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            self._demonstrative_p.append(probability)
        # self.visualize_kosoa()

    def pointing_inner_product(self, position):
        object_theta = []
        # self._object_probability = []
        # # 手首の中央値
        # self.median_wrist_x = statistics.median(self._wrist_x)
        # self.median_wrist_y = statistics.median(self._wrist_y)
        # self.median_wrist_z = statistics.median(self._wrist_z)
        # # rospy.loginfo("wrist_median \n cX:{} cY:{} cZ:{}".format(self.median_wrist_x, self.median_wrist_y, self.median_wrist_z))

        # # 目の中央値
        # self.median_eye_x = statistics.median(self._eye_x)
        # self.median_eye_y = statistics.median(self._eye_y)
        # self.median_eye_z = statistics.median(self._eye_z)
        # # rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_eye_x, self.median_eye_y, self.median_eye_z))

        pointing_vector = np.array([self.median_wrist_x - self.median_eye_x, self.median_wrist_y - self.median_eye_y, self.median_wrist_z - self.median_eye_z])
        length_arm = pointing_vector[0]**2 + pointing_vector[1]**2 + pointing_vector[2]**2
        length_pointing_vector = math.sqrt(length_arm)

        # ガウスでいう分散を決めるパラメータ
        vonmises_kappa = 1

        # 物体の3次元マップ座標読み取る
        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            object_vector = np.array([object_x - self.median_eye_x, object_y - self.median_eye_y, object_z - self.median_eye_z])
            dis_obj = object_vector[0]**2 + object_vector[1]**2 + object_vector[2]**2
            length_object_vector = math.sqrt(dis_obj)

            # 内積
            inner = pointing_vector * object_vector

            # print(i)
            # print(inner)
            # print(length_pointing_vector)
            # print(length_object_vector)

            # cos = (pointing_vector * object_vector) / length_pointing_vector * length_object_vector
            cos = (inner[0] + inner[1] + inner[2]) / (length_pointing_vector * length_object_vector)
            # print(cos)
            theta = math.acos(cos)
            vonmises_dis = vonmises(vonmises_kappa)
            probability = vonmises_dis.pdf(theta)
            self._pointing_p.append(probability)
            # self._object_probability.append(probability)

    # def visualize_kosoa(self):
    #     #格子数
    #     xcells = 18
    #     ycells = 18
    #     #xやyの描画の範囲(地図によってここをいじらないといけない)
    #     xmin = -4
    #     xmax = 5
    #     ymin = -8
    #     ymax = 2
    #     around = 6
    #     #地図のyamlファイルの情報
    #     resolution = np.round(0.050000,decimals=3)
    #     origin_x = -51.224998
    #     origin_y = -51.224998
    #     height = 2048
    #     width = 2048

    #     #専有されているところの濃度を0に(←無視するのでTrue)
    #     give_color = True

    #     #ヒートマップのぼかし方リスト
    #     #13番目がgaussian
    #     method_idx = 13
    #     methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    #             'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    #             'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    #     map_file = 'map.pgm'

    #     #ヒートマップに入れる濃度の情報

        
    #     width_one_cell = (xmax - xmin) / xcells
    #     height_one_cell = (ymax - ymin) / ycells
    #     IG_x = np.zeros(xcells*ycells)
    #     IG_y = np.zeros(xcells*ycells)
    #     # 各セルの中心座標を計算
    #     for y in range(0, ycells):
    #         for x in range(0, xcells):
    #             IG_x[xcells*y+x] = xmin + (width_one_cell/2) * (1+x*2)
    #             IG_y[xcells*y+x] = ymin + (height_one_cell/2) * (1+y*2)
    #     IG = np.zeros(xcells*ycells)

    #     gaussian_dis = multivariate_normal([self._gaussian_mean[0], self._gaussian_mean[1]], [[self._gaussian_variance, 0], [0, self._gaussian_variance]])
    #     for i in range(len(IG_x)):
    #         IG[i] = gaussian_dis.pdf([IG_x[i], IG_y[i]])

    #     #地図描画
    #     map_image = np.array(Image.open(map_file))
    #     plt.imshow(map_image,extent=(origin_x, origin_x+height*resolution, origin_y, origin_y+width*resolution),cmap='gray')

    #     if give_color == False:
    #         for i in range(len(IG)):
    #             xindex = int((IG_x[i] - origin_x) / resolution)
    #             yindex = int((IG_y[i] - origin_y) / resolution)
    #             free = True
    #             for x in range(-around, around+1):
    #                 for y in range(-around, around+1):
    #                     if map_image[width-yindex+y][xindex+x] != 254:
    #                         free = False
                
    #             if free == False:
    #                 IG[i] = 0
    #     grid_IG = np.reshape(IG, (ycells, xcells))
    #     grid_IG = np.flipud(grid_IG)
    #     grid_IG *= 1
    #     plt.imshow(grid_IG, extent =(xmin, xmax, ymin, ymax), interpolation=methods[method_idx], cmap='Reds', alpha=0.8)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     margin = -0.1
    #     plt.imshow(map_image,extent=(origin_x, origin_x+height*resolution, origin_y, origin_y+width*resolution),cmap='gray', alpha=0.5)
    #     plt.xlim(xmin, xmax)
    #     plt.ylim(ymin, ymax)

    #     plt.savefig("heatmap" + '.png', dpi=300)
    #     # plt.savefig("heatmap" + '.pdf', dpi=300)
    #     # plt.savefig("heatmap" + '.svg', dpi=350)

    #     plt.cla()
    #     plt.clf()
    #     plt.close()


    def visualize_point(self):
        self.marker_msg = Marker()
        # for i in range(len(x_list)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        # Object type, 矢印やったら0
        marker.type = 2
        marker.action = Marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self._point_ground[0]
        marker.pose.position.y = self._point_ground[1]
        marker.pose.position.z = self._point_ground[2]
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

    def visualize_eye(self):
        self.marker_msg = Marker()
        # for i in range(len(x_list)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        # Object type, 矢印やったら0
        marker.type = 2
        marker.action = Marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.median_eye_x
        marker.pose.position.y = self.median_eye_y
        marker.pose.position.z = self.median_eye_z
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

    def visualize_pointing(self):
        self.marker_array_msg = MarkerArray()
        for t in range(self._t):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = t
            # Object type, 矢印やったら0
            marker.type = 2
            marker.action = Marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = (1 - t) * self.median_eye_x + t * self.median_wrist_x
            marker.pose.position.y = (1 - t) * self.median_eye_y + t * self.median_wrist_y
            marker.pose.position.z = (1 - t) * self.median_eye_z + t * self.median_wrist_z
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
            marker.ns = "Goal-%u"%t
            self.marker_array_msg.markers.append(marker)
        # self.marker_array_msg.lifetime = rospy.Duration(10)
        self.vector_pub.publish(self.marker_array_msg)
        # self.rate.sleep()

    def image_callback(self, message):
        # rospy.loginfo("Entered callback")
        self.rgb_img = self.cv_bridge.compressed_imgmsg_to_cv2(message)

    def point_cloud_callback(self, point_cloud):
        self._point_cloud = pointcloud2_to_xyz_array(point_cloud, False)
        # rospy.loginfo("PointCloud frame id : {}".format(point_cloud.header.frame_id))
        self._point_cloud_header = point_cloud.header

    def global_pose_callback(self, msg):
        self._global_pose_x = msg.pose.position.x
        self._global_pose_y = msg.pose.position.y
        self._global_pose_z = msg.pose.position.z

    def get_point(self, x, y):
        try:
            return self._point_cloud[y][x]
        except:
            # rospy.loginfo("GET POINT ERROR")
            pass



if __name__ == '__main__':
    rospy.init_node('exophora')
    # get_depth_from_rgb = GetDepthPointFromFrame()
    node = Exophora()
    node.main()
    # rospy.spin()
