#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
import math
# 正規分布
from scipy.stats import multivariate_normal

# ガウス分布可視化のためのもの
import rviz_gaussian_distribution_msgs.msg as rgd_msgs
from matplotlib import colors


class DemonstrativeRegionEstimator():
    def __init__(self):

        self.distribution_publisher = rospy.Publisher("/gaussian_distribution/input/add", rgd_msgs.GaussianDistribution,
                                                      queue_size=1)

        self.subscriber_for_global_pose = rospy.Subscriber(
            "/global_pose",
            PoseStamped,
            self.global_pose_callback,
            queue_size=1
        )


    def calculate_demonstrative_region(self, position, wrist_frame, eye_frame, kosoa):
        demonstrative_p = []
        # 手首の中央値
        # self.median_wrist_x = statistics.median(self._wrist_x)
        # self.median_wrist_y = statistics.median(self._wrist_y)
        # self.median_wrist_z = statistics.median(self._wrist_z)
        # rospy.loginfo("wrist_median \n cX:{} cY:{} cZ:{}".format(self.median_wrist_x, self.median_wrist_y, self.median_wrist_z))
        # 目の中央値
        # self.median_eye_x = statistics.median(self._eye_x)
        # self.median_eye_y = statistics.median(self._eye_y)
        # self.median_eye_z = statistics.median(self._eye_z)
        # rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_eye_x, self.median_eye_y, self.median_eye_z))

        # wrist_frame = np.array([self.median_wrist_x, self.median_wrist_y, self.median_wrist_z])
        # eye_frame = np.array([self.median_eye_x, self.median_eye_y, self.median_eye_z])
        robot_frame = np.array([self._global_pose_x, self._global_pose_y, self._global_pose_z])

        # 人とロボットの距離
        # dis_hr = (eye_frame[0] - robot_frame[0])**2 + (eye_frame[1] - robot_frame[1])**2 + (eye_frame[2] - robot_frame[2])**2
        dis_hr = (eye_frame[0] - robot_frame[0]) ** 2 + (eye_frame[1] - robot_frame[1]) ** 2 + (0 - robot_frame[2]) ** 2
        self._distance = math.sqrt(dis_hr)

        # 目と肘間の長さ
        dis_ew = (eye_frame[0] - wrist_frame[0]) ** 2 + (eye_frame[1] - wrist_frame[1]) ** 2 + (
                    eye_frame[2] - wrist_frame[2]) ** 2
        self._length = math.sqrt(dis_ew)

        # 分散のパラメータ
        variance_alpha = 0.1
        variance_beta = 0.1

        # コ系列の平均と分散と可視化
        ko_gaussian_mean = wrist_frame
        ko_gaussian_variance = self._distance * variance_alpha / variance_beta
        self.publish_distribution(0, [ko_gaussian_mean[0], ko_gaussian_mean[1]],
                                  [[ko_gaussian_variance, 0], [0, ko_gaussian_variance]], 0)
        print(ko_gaussian_variance)
        # ソ系列の平均と分散と可視化
        so_gaussian_mean = (1 - variance_alpha) * robot_frame + variance_alpha * eye_frame
        so_gaussian_variance = self._distance * variance_alpha / variance_beta
        self.publish_distribution(1, [so_gaussian_mean[0], so_gaussian_mean[1]],
                                  [[so_gaussian_variance, 0], [0, so_gaussian_variance]], 1)
        # ア系列の平均と分散と可視化
        a = 2
        a_gaussian_mean = (1 - a * self._distance / self._length) * eye_frame + (
                    a * self._distance / self._length) * wrist_frame
        a_gaussian_variance = self._distance * variance_alpha / variance_beta
        self.publish_distribution(2, [a_gaussian_mean[0], a_gaussian_mean[1]],
                                  [[a_gaussian_variance, 0], [0, a_gaussian_variance]], 2)

        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            if (kosoa == "ko"):
                # print("ko")
                # self._gaussian_mean = wrist_frame
                # self._gaussian_variance = self._distance * variance_alpha / variance_beta
                # 各物体に対してその物体の座標を入力して確率値を計算
                gaussian_dis = multivariate_normal([ko_gaussian_mean[0], ko_gaussian_mean[1], ko_gaussian_mean[2]],
                                                   [[ko_gaussian_variance, 0, 0], [0, ko_gaussian_variance, 0],
                                                    [0, 0, ko_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if (kosoa == "so"):
                # print("so")
                # self._gaussian_mean = (1 - variance_alpha) * robot_frame + variance_alpha * eye_frame
                # self._gaussian_variance = self._distance * variance_alpha / variance_beta
                # 各物体に対してその物体の座標を入力して確率値を計算
                gaussian_dis = multivariate_normal([so_gaussian_mean[0], so_gaussian_mean[1], so_gaussian_mean[2]],
                                                   [[so_gaussian_variance, 0, 0], [0, so_gaussian_variance, 0],
                                                    [0, 0, so_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if (kosoa == "a"):
                # print("a")
                # a = 2
                # self._gaussian_mean = (1 - a * self._distance / self._length) * eye_frame + (a * self._distance / self._length) * wrist_frame
                # self._gaussian_variance = self._distance * variance_alpha / variance_beta
                # 各物体に対してその物体の座標を入力して確率値を計算
                gaussian_dis = multivariate_normal([a_gaussian_mean[0], a_gaussian_mean[1], a_gaussian_mean[2]],
                                                   [[a_gaussian_variance, 0, 0], [0, a_gaussian_variance, 0],
                                                    [0, 0, a_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            demonstrative_p.append(probability)
            return demonstrative_prob
        # self.visualize_kosoa()

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

    def global_pose_callback(self, msg):
        self._global_pose_x = msg.pose.position.x
        self._global_pose_y = msg.pose.position.y
        self._global_pose_z = msg.pose.position.z


if __name__ == '__main__':
    rospy.init_node('demonstrative_region_estimator')
    demonstrative_region_estimator = DemonstrativeRegionEstimator()
    # rospy.spin()

