#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import csv
import yaml
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

import random

# 下4つはmediapipe用
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
# ガウス分布可視化のためのもの
import rviz_gaussian_distribution_msgs.msg as rgd_msgs


class Baseline_vgpn():

    def __init__(self):

        self._object_category = ["Bottle", "Stuffed Toy", "Book", "Cup"]

        with open('yolov5m_Object365.yaml', 'r') as yml:
            object_yaml = yaml.load(yml, Loader=yaml.SafeLoader)
            self.object_365 = object_yaml['names']

        # self._demonstrative_p = []
        # self._pointing_p = []

        # _point_ground = np.array([0, 0, 0])


        self.cv_bridge = CvBridge()


    def main(self):
        # self.setup_subscriber()
        # rospy.sleep(1)

        # 物体のインデックスとカテゴリと3次元座標のリストを作成
        object_list = self.load_objects_position_data()
        # リストの中身は [object number, object name, x, y, z] が20個
        object_frame = []
        for j in range(len(object_list)):
            object_frame.append([object_list[j][2], object_list[j][3], object_list[j][4]])
        object_frame = np.array(object_frame, dtype = float)

        # 人の座標取得、指示語、指した物体
        human_info = self.load_human_data()
        # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]

        correct = 0

        for i in range(len(human_info[0])):
            # print(len(human_info[0]))
            ppp = [0, 0, 0, 0]
            self._object_class_p = []
            # self._demonstrative_p = []
            # self._pointing_p = []

        # 人の座標を変数に入れる
            self.ex_data(human_info, i)
            # self._target_object_id = self.object_365.index(self._target_object_name)

            # rospy.loginfo("Skeleton could be detected!")

            # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]

            # 物体カテゴリ情報が取れたとき
            # if np.isnan(self._target_object_name) is False:
            if self._target_object_name != 'nan':
                # 物体カテゴリ確率を取得
                # [object_label, max_conf, [365個の確率]]
                self._target_object_id = self.object_365.index(self._target_object_name)
                self._object_class_p = self.load_conf()
                kouho = []
                for j in range(len(self._object_class_p)):
                    if self._object_class_p[j] > 0.3:
                        kouho.append(j)
                ppp[0] = 1

            # 指差しベクトルは取れたとき
            if ((np.isnan(self._wrist[0]) == False) and (np.isnan(self._eye[0]) == False)):
                # self.pointing_inner_product(object_list)
                ppp[1] = 1
                point = self._pointing_vector()
                point = np.array(point, dtype = float)
                point_object_d = object_frame - point
                point_object_d = point_object_d ** 2
                # print(len(point_object_d))
                # print(point_object_d)
                pointing_p = []
                for j in range(len(point_object_d)):
                    pointing_p.append(math.sqrt(point_object_d[j][0] + point_object_d[j][1] + point_object_d[j][2]))
                # print(pointing_p)

            # print(self._eye[0])
            # print(self._kosoa)

            # 指差しも指示語も取れたとき
            # if ((np.isnan(self._wrist[0]) == False) and (np.isnan(self._eye[0]) == False) and ((self._kosoa == 'nan') == False)):
            #     self.pointing_inner_product(object_list)
            #     self.kosoa(object_list)
            #     # print("iiiiiiiiiiiiiiiiiiiiiiiiiii")
            #     ppp[3] = 1
            # print("sssssssssssssssssssss")
            object_class_p = np.array(self._object_class_p)

            # 正規化
            # sum_object_class_p = np.sum(object_class_p)
            # object_class_p = object_class_p / sum_object_class_p


            # 3つの確率値を掛け合わせる
            # 物体カテゴリ情報が取れなかったとき
            kouho_object = []
            if ppp[0] == 0:
                aaa = np.argmin(pointing_p)
                target = aaa
                baai = "no object category"
            # 指差しベクトルがなかったとき
            elif ppp[1] == 0:
                aaa = random.randrange(5)
                target = kouho[aaa]
                baai = "no pointing"
            elif (ppp[0] == 1 and ppp[1] == 1):
                for j in range(len(kouho)):
                    kouho_object.append(pointing_p[kouho[j]])
                    aaa = np.argmin(kouho_object)
                    target = kouho[aaa]
                print(kouho)
                print(kouho_object)
                baai = "ALL"

            # sum_target_probability = np.sum(target_probability)
            # # 正規化
            # target_probability = target_probability / sum_target_probability
            # target = np.amax(target_probability)
            # target_index = np.argmax(target_probability)
            # # print用に小数点第3位を四捨五入
            # target_probability = np.round(target_probability, decimals=2)
            # # print(object_list)
            # target_probability_reshape = np.reshape(target_probability, (4, 5))
            print(i, "番目")
            # for j in range(4):
            #     print("対象確率:", target_probability_reshape[j])
            # print("対象確率", target_probability)
            print("kosoa", self._kosoa)
            print("予測した目標物体:", target)
            print("正解:",self._answer_object_id)
            print("---------------------------------------------")
            if int(target) == int(self._answer_object_id):
                correct += 1
        print(baai)
        sr = correct / len(human_info[0])
        print(correct, sr)
        # print(np.round(object_class_p, decimals=2))

    # def publish_distribution(self, id_, mu_k, sigma_k, i):
        
    # def publish_clear(self):

    def load_objects_position_data(self):
        files = os.listdir("/root/HSR/catkin_ws/src/mediapipe_ros/data/objects_position/")
        datas = []
        data = []
        for i in range(len(files)):
            with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/objects_position/object_position_{}.csv".format(str(i+1))) as f1:
                reader = csv.reader(f1)
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

    def load_conf(self):
        # confを取得
        # files = os.listdir("/root/HSR/catkin_ws/src/mediapipe_ros/data/objects_position/")
        data = []
        datas = []
        with open("/root/HSR/catkin_ws/src/mediapipe_ros/src/object_conf.csv") as f3:
            reader = csv.reader(f3)
            for row in reader:
                row[2] = row[2].replace('(','')
                row[2] = row[2].replace(')','')
                r_list = row[2].split(',')
                for i in range(len(r_list)):
                    r_list[i] = float(r_list[i])
                plist = r_list
                plist_np = np.array(plist)
                sum_plist = np.sum(plist_np)
                # 正規化
                plist_np = plist_np / sum_plist
                data.append([row[0], plist_np[self._target_object_id]])

            for k in range(len(self._object_category)):
                for i in range(len(data)):
                    if(data[i][0] == self._object_category[k]):
                        datas.append(data[i][1])

        return datas

    def load_human_data(self):
        # files = os.listdir("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/")
        datas = []
        data = []
        # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]
        # for i in range(len(files)):

        # data kaeru toki kokokaeru
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_no_demonstrative.csv") as f2:
        with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_no_object.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_no_pointing.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_miss_pointing.csv") as f2:
            reader = csv.reader(f2)
            for row in reader:
                # print(row[0])
                for j in range(5):
                    row[j] = row[j].replace('[','')
                    row[j] = row[j].replace(']','')
                    r_list = row[j].split(',')
                    for i in range(len(r_list)):
                        if(r_list[i] == 'nan'):
                            r_list[i] = np.nan
                        else:
                            r_list[i] = float(r_list[i])
                    row[j] = r_list
                    # plist_np = np.array(plist)
                    # sum_plist = np.sum(plist_np)
                data.append(row)
            datas.append(data)
            # data = []
            # print(datas[0][0])
        return datas

    def ex_data(self, human, i):
        # ここの関数で読み込んだ人のマップ座標のリストを変数に入れる
        # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]
        self._wrist = np.array(human[0][i][0])
        self._eye = np.array(human[0][i][1])
        self._l_shoulder = np.array(human[0][i][2])
        self._r_shoulder = np.array(human[0][i][3])
        self._global_pose = np.array(human[0][i][4])
        self._kosoa = human[0][i][5]
        self._target_object_name = human[0][i][6]
        self._answer_object_id = human[0][i][7]
        # もし、どちらかの座標が取れなかったら目だけ入れる

    def _pointing_vector(self):

        map_xlim = [-3.5, 3.3]
        map_ylim = [-7.6, 1]
        map_zlim = [0, 3]
        point_vector = np.array([0, 0, 0])
        # t = 1
        
        # 地面との交点
        u = self._eye[2] / (self._eye[2] - self._wrist[2])
        point_ground_x = (1 - u) * self._eye[0] + u * self._wrist[0]
        point_ground_y = (1 - u) * self._eye[1] + u * self._wrist[1]
        _point_ground = np.array([point_ground_x, point_ground_y, 0])
        self._t = int(u)
        # print(self._t)
        # self.visualize_point()
        t = 1
        # 天井に当たったときに判定だけやってない
        if ((point_ground_x < map_xlim[0]) or (map_xlim[1] < point_ground_x) or (point_ground_y < map_ylim[0]) or (map_ylim[1] < point_ground_y)):
            while True:
                t += 1
                # print(t)
                for i in range(3):
                    point_vector[i] = (1 - t) * self._eye[i] + t * self._wrist[i]
                if point_vector[0] < map_xlim[0] - 0.5:
                    x3 = map_xlim[0]
                    t = (self._eye[0] - x3) / (self._eye[0] - self._wrist[0])
                    y3 = (1 - t) * self._eye[1] + t * self._wrist[1]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    _point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[0] > map_xlim[1] + 0.5:
                    x3 = map_xlim[1]
                    t = (self._eye[0] - x3) / (self._eye[0] - self._wrist[0])
                    y3 = (1 - t) * self._eye[1] + t * self._wrist[1]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    _point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] < map_ylim[0] - 0.5:
                    y3 = map_ylim[0]
                    t = (self._eye[1] - y3) / (self._eye[1] - self._wrist[1])
                    x3 = (1 - t) * self._eye[0] + t * self._wrist[0]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    _point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] > map_ylim[1] + 0.5:
                    y3 = map_ylim[1]
                    t = (self._eye[1] - y3) / (self._eye[1] - self._wrist[1])
                    x3 = (1 - t) * self._eye[0] + t * self._wrist[0]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    _point_ground = np.array([x3, y3, z3])
                    break
            self._t = int(t)
            # point_ground = np.array([x3, y3, z3])
        return _point_ground

    # def kosoa(self, position):

    # def kosoa_lack(self, position):

    # def pointing_inner_product(self, position):

    # def visualize_kosoa(self):


    # def visualize_point(self):

    # def visualize_eye(self):

    # def visualize_pointing(self):

    # def image_callback(self, message):

    # def point_cloud_callback(self, point_cloud):

    # def global_pose_callback(self, msg):

    # def get_point(self, x, y):


if __name__ == '__main__':
    rospy.init_node('beseline_vgpn')
    # get_depth_from_rgb = GetDepthPointFromFrame()
    node = Baseline_vgpn()
    node.main()
    # rospy.spin()
