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

# 下4つはmediapipe用
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
# ガウス分布可視化のためのもの
import rviz_gaussian_distribution_msgs.msg as rgd_msgs


class Exophora():

    def __init__(self):

        self._object_category = ["Bottle", "Stuffed Toy", "Book", "Cup"]

        with open('yolov5m_Object365.yaml', 'r') as yml:
            object_yaml = yaml.load(yml, Loader=yaml.SafeLoader)
            self.object_365 = object_yaml['names']

        # self._bottle_id = self.object_365.index('Bottle')
        # self._stuffedtoy_id = self.object_365.index('Stuffed Toy')
        # self._book_id = self.object_365.index('Book')
        # self._cup_id = self.object_365.index('Cup')

        # 人の座標取得、指示語、指した物体
        # human_info = self.load_human_data()
        # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]

        # # 人の座標を変数に入れる
        # self.ex_data(human_info)

        # self._object_class_p = []
        self._demonstrative_p = []
        self._pointing_p = []

        # self._wrist_x = []
        # self._wrist_y = []
        # self._wrist_z = []
        # self._eye_x = []
        # self._eye_y = []
        # self._eye_z = []

        self._point_ground = np.array([0, 0, 0])

        # self._kosoa = "a"
        # self._target_object_name = self._object_category[0]
        # self._target_object_id = self.object_365.index(self._target_object_name)

        self.cv_bridge = CvBridge()

        # tfらへん
        # self._point_cloud_header = Header()
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # mediapipeのための追加部分
        #self.mp_pose = mp.solutions.hands
        # self.mp_pose = mp.solutions.pose
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mesh_drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
        # self.mark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=3, circle_radius=1, color=(0, 0, 255))
        #self.sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed", CompressedImage, self.main)

        # self.distribution_publisher = rospy.Publisher("/gaussian_distribution/input/add", rgd_msgs.GaussianDistribution, queue_size=1)

    def main(self):
        # self.setup_subscriber()
        # rospy.sleep(1)

        # 物体のインデックスとカテゴリと3次元座標のリストを作成
        object_list = self.load_objects_position_data()
        # リストの中身は [object number, object name, x, y, z] が20個

        # 人の座標取得、指示語、指した物体
        human_info = self.load_human_data()
        # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]

        correct = 0

        for i in range(len(human_info[0])):
            # print(i)
            ppp = [0, 0, 0, 0]
            self._object_class_p = []
            self._demonstrative_p = []
            self._pointing_p = []

        # 人の座標を変数に入れる
            self.ex_data(human_info, i)
            # self._target_object_id = self.object_365.index(self._target_object_name)

            # rospy.loginfo("Skeleton could be detected!")

            # [wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx]

            # 物体カテゴリ情報が取れたとき
            # if np.isnan(self._target_object_name) is False:
            if self._target_object_name != 'nan':
                # print("aaaaaaaaaaaaaaaaaa")
                # 物体カテゴリ確率を取得
                # [object_label, max_conf, [365個の確率]]
                self._target_object_id = self.object_365.index(self._target_object_name)
                self._object_class_p = self.load_conf()
                ppp[0] = 1

            # 指差しベクトルは取れたが、指示語がなかったとき
            if ((np.isnan(self._wrist[0]) == False) and (np.isnan(self._eye[0]) == False) and ((self._kosoa == 'nan') == True)):
                self.pointing_inner_product(object_list)
                ppp[1] = 1
                # self._pointing_vector()
            #     # self.visualize_kosoa()

            # 指差しが取れなくて指示語は取れたとき
            if (((np.isnan(self._wrist[0]) == True) or (np.isnan(self._eye[0]) == True)) and ((self._kosoa == 'nan') == False)):
                self.kosoa_lack(object_list)
                ppp[2] = 1

            # print(self._eye[0])
            # print(self._kosoa)

            # 指差しも指示語も取れたとき
            if ((np.isnan(self._wrist[0]) == False) and (np.isnan(self._eye[0]) == False) and ((self._kosoa == 'nan') == False)):
                self.pointing_inner_product(object_list)
                self.kosoa(object_list)
                ppp[3] = 1

            object_class_p = np.array(self._object_class_p)
            demonstrative_p = np.array(self._demonstrative_p)
            pointing_p = np.array(self._pointing_p)

            # 正規化
            sum_object_class_p = np.sum(object_class_p)
            object_class_p = object_class_p / sum_object_class_p
            sum_demonstrative_p = np.sum(demonstrative_p)
            demonstrative_p = demonstrative_p / sum_demonstrative_p
            sum_pointing_p = np.sum(pointing_p)
            pointing_p = pointing_p / sum_pointing_p

            # graph
            # self._three_graph(object_class_p, demonstrative_p, pointing_p)

            # print(len(object_class_p))
            # print(len(demonstrative_p))
            # print(len(pointing_p))

            # 3つの確率値を掛け合わせる
            # 物体カテゴリ情報が取れなかったとき
            if ppp[0] == 0:
                target_probability = demonstrative_p * pointing_p
                baai = "no object category"
            # 指差しベクトルは取れたが、指示語がなかったとき
            elif ppp[1] == 1:
                target_probability = object_class_p * pointing_p
                baai = "no demonstrative"
            # 指差しが取れなくて指示語は取れたとき
            elif ppp[2] == 1:
                target_probability = object_class_p * demonstrative_p
                baai = "no pointing"
            # 指差しも指示語も取れたとき
            elif ppp[3] == 1:
                target_probability = object_class_p * demonstrative_p * pointing_p
                # ablation study用
                # target_probability = object_class_p
                # target_probability = demonstrative_p
                # target_probability = pointing_p
                # target_probability = demonstrative_p * pointing_p
                # target_probability = object_class_p * pointing_p
                # target_probability = object_class_p * demonstrative_p
                baai = "ALL"

            sum_target_probability = np.sum(target_probability)
            # 正規化
            target_probability = target_probability / sum_target_probability
            target = np.amax(target_probability)
            target_index = np.argmax(target_probability)
            # グラフ出力
            # self._graph(target_probability)
            # print用に小数点第3位を四捨五入
            target_probability = np.round(target_probability, decimals=2)
            # print(object_list)
            target_probability_reshape = np.reshape(target_probability, (4, 5))
            print(i, "番目")
            for j in range(4):
                print("対象確率:", target_probability_reshape[j])
            # print("対象確率", target_probability)
            print("予測した目標物体:", object_list[target_index])
            print("正解:",self._answer_object_id)
            if int(object_list[target_index][0]) == int(self._answer_object_id):
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
        with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_1data.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_no_demonstrative.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_no_object.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_no_pointing.csv") as f2:
        # with open("/root/HSR/catkin_ws/src/mediapipe_ros/data/human_position/wrist_eye_miss_pointing.csv") as f2:
            reader = csv.reader(f2)
            for row in reader:
                # print(row[0])
                for j in range(5):
                    # print(row[j])
                    row[j] = row[j].replace('[','')
                    row[j] = row[j].replace(']','')
                    r_list = row[j].split(',')
                    # print(r_list)
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
        self._point_ground = np.array([point_ground_x, point_ground_y, 0])
        self._t = int(u)
        # self.visualize_point()
        t = 1
        # 天井に当たったときに判定だけやってない
        if ((point_ground_x < map_xlim[0]) or (map_xlim[1] < point_ground_x) or (point_ground_y < map_ylim[0]) or (map_ylim[1] < point_ground_y)):
            while True:
                t += 1
                for i in range(3):
                    point_vector[i] = (1 - t) * self._eye[i] + t * self._eye[i]
                if point_vector[0] < map_xlim[0] - 0.5:
                    x3 = map_xlim[0]
                    t = (self._eye[0] - x3) / (self._eye[0] - self._eye[0])
                    y3 = (1 - t) * self._eye[1] + t * self._wrist[1]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[0] > map_xlim[1] + 0.5:
                    x3 = map_xlim[1]
                    t = (self._eye[0] - x3) / (self._eye[0] - self._wrist[0])
                    y3 = (1 - t) * self._eye[1] + t * self._wrist[1]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] < map_ylim[0] - 0.5:
                    y3 = map_ylim[0]
                    t = (self._eye[1] - y3) / (self._eye[1] - self._wrist[1])
                    x3 = (1 - t) * self._eye[0] + t * self._wrist[0]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] > map_ylim[1] + 0.5:
                    y3 = map_ylim[1]
                    t = (self._eye[1] - y3) / (self._eye[1] - self._wrist[1])
                    x3 = (1 - t) * self._eye[0] + t * self._wrist[0]
                    z3 = (1 - t) * self._eye[2] + t * self._wrist[2]
                    self._point_ground = np.array([x3, y3, z3])
                    break
            self._t = int(t)

    def kosoa(self, position):
        # print("kosoakosoakosoakosoa")
        robot_frame = np.array([self._global_pose[0], self._global_pose[1], self._global_pose[2]])
        # print(type(self._eye[0]))

        # 人とロボットの距離
        # dis_hr = (self._eye[0] - robot_frame[0])**2 + (self._eye[1] - robot_frame[1])**2 + (self._eye[2] - robot_frame[2])**2
        dis_hr = (self._eye[0] - robot_frame[0])**2 + (self._eye[1] - robot_frame[1])**2
        self._distance = math.sqrt(dis_hr)

        # 目と肘間の長さ
        dis_ew = (self._eye[0] - self._wrist[0])**2 + (self._eye[1] - self._wrist[1])**2 + (self._eye[2] - self._wrist[2])**2
        self._length = math.sqrt(dis_ew)

        # 分散のパラメータ
        variance_alpha = 0.1
        variance_beta = 0.1

        # コ系列の平均と分散と可視化
        ko_gaussian_mean = self._wrist
        ko_gaussian_variance = self._distance * variance_alpha / variance_beta
        # self.publish_distribution(0, [ko_gaussian_mean[0], ko._gaussian_mean[1]], [[ko._gaussian_variance, 0], [0, ko._gaussian_variance]], 0)
        # ソ系列の平均と分散と可視化
        so_gaussian_mean = (1 - variance_alpha) * robot_frame + variance_alpha * self._eye
        so_gaussian_variance = self._distance * variance_alpha / variance_beta
        # self.publish_distribution(1, [so_gaussian_mean[0], so_gaussian_mean[1]], [[so_gaussian_variance, 0], [0, so_gaussian_variance]], 1)
        # ア系列の平均と分散と可視化
        a = 3
        a_gaussian_mean = (1 - a * self._distance / self._length) * self._eye + (a * self._distance / self._length) * self._wrist
        a_gaussian_variance = self._distance * variance_alpha / variance_beta
        # self.publish_distribution(2, [a_gaussian_mean[0], a_gaussian_mean[1]], [[a_gaussian_variance, 0], [0, a_gaussian_variance]], 2)

        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            if(self._kosoa == "ko"):
                gaussian_dis = multivariate_normal([ko_gaussian_mean[0], ko_gaussian_mean[1], ko_gaussian_mean[2]], [[ko_gaussian_variance, 0, 0], [0, ko_gaussian_variance, 0], [0, 0, ko_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if(self._kosoa == "so"):
                gaussian_dis = multivariate_normal([so_gaussian_mean[0], so_gaussian_mean[1], so_gaussian_mean[2]], [[so_gaussian_variance, 0, 0], [0, so_gaussian_variance, 0], [0, 0, so_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if(self._kosoa == "a"):
                gaussian_dis = multivariate_normal([a_gaussian_mean[0], a_gaussian_mean[1], a_gaussian_mean[2]], [[a_gaussian_variance, 0, 0], [0, a_gaussian_variance, 0], [0, 0, a_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            # print(probability)
            self._demonstrative_p.append(probability)
        # self.visualize_kosoa()

    def kosoa_lack(self, position):
        robot_frame = np.array([self._global_pose[0], self._global_pose[1], self._global_pose[2]])
        human_frame = np.array([(self._l_shoulder[0] + self._r_shoulder[0]) / 2, (self._l_shoulder[1] + self._r_shoulder[1]) / 2, (self._l_shoulder[2] + self._r_shoulder[2]) / 2])
        # print(type(self._eye[0]))

        # 人とロボットの距離
        # dis_hr = (self._eye[0] - robot_frame[0])**2 + (self._eye[1] - robot_frame[1])**2 + (self._eye[2] - robot_frame[2])**2
        dis_hr = (human_frame[0] - robot_frame[0])**2 + (human_frame[1] - robot_frame[1])**2
        self._distance = math.sqrt(dis_hr)

        # # 目と肘間の長さ
        # dis_ew = (self._eye[0] - self._wrist[0])**2 + (self._eye[1] - self._wrist[1])**2 + (self._eye[2] - self._wrist[2])**2
        # self._length = math.sqrt(dis_ew)

        # 分散のパラメータ
        variance_alpha = 0.1
        variance_beta = 0.1

        # コ系列の平均と分散と可視化
        ko_gaussian_mean = human_frame
        ko_gaussian_variance = self._distance * variance_alpha / variance_beta
        # self.publish_distribution(0, [ko_gaussian_mean[0], ko._gaussian_mean[1]], [[ko._gaussian_variance, 0], [0, ko._gaussian_variance]], 0)
        # ソ系列の平均と分散と可視化
        so_gaussian_mean = (1 - variance_alpha) * robot_frame + variance_alpha * human_frame
        so_gaussian_variance = self._distance * variance_alpha / variance_beta
        # self.publish_distribution(1, [so_gaussian_mean[0], so_gaussian_mean[1]], [[so_gaussian_variance, 0], [0, so_gaussian_variance]], 1)
        # ア系列の平均と分散と可視化
        
        # self.publish_distribution(2, [a_gaussian_mean[0], a_gaussian_mean[1]], [[a_gaussian_variance, 0], [0, a_gaussian_variance]], 2)

        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            if(self._kosoa == "ko"):
                gaussian_dis = multivariate_normal([ko_gaussian_mean[0], ko_gaussian_mean[1], ko_gaussian_mean[2]], [[ko_gaussian_variance, 0, 0], [0, ko_gaussian_variance, 0], [0, 0, ko_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if(self._kosoa == "so"):
                gaussian_dis = multivariate_normal([so_gaussian_mean[0], so_gaussian_mean[1], so_gaussian_mean[2]], [[so_gaussian_variance, 0, 0], [0, so_gaussian_variance, 0], [0, 0, so_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])

            if(self._kosoa == "a"):
                gaussian_dis = multivariate_normal([ko_gaussian_mean[0], ko_gaussian_mean[1], ko_gaussian_mean[2]], [[ko_gaussian_variance, 0, 0], [0, ko_gaussian_variance, 0], [0, 0, ko_gaussian_variance]])
                probability = gaussian_dis.pdf([object_x, object_y, object_z])
                # 1-コ系列
                probability = 1- probability

            # print(probability)
            self._demonstrative_p.append(probability)

    def pointing_inner_product(self, position):
        # print("pointingpointing")
        object_theta = []
        pointing_vector = np.array([self._wrist[0] - self._eye[0], self._wrist[1] - self._eye[1], self._wrist[2] - self._eye[2]])
        length_arm = pointing_vector[0]**2 + pointing_vector[1]**2 + pointing_vector[2]**2
        length_pointing_vector = math.sqrt(length_arm)

        # ガウスでいう分散を決めるパラメータ
        vonmises_kappa = 1

        # 物体の3次元マップ座標読み取る
        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            object_vector = np.array([object_x - self._eye[0], object_y - self._eye[1], object_z - self._eye[2]])
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
        
    def _graph(self, p):
        # left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        height = p
        left = np.arange(len(height))
        labels = ['Bottle_A', 'Bottle_B', 'Bottle_C', 'Bottle_D', 'Bottle_E', 'Stuffed Toy_A', 'Stuffed Toy_B', 'Stuffed Toy_C', 'Stuffed Toy_D', 'Stuffed Toy_E', 'Book_A', 'Book_B', 'Book_C', 'Book_D', 'Book_E', 'Cup_A', 'Cup_B', 'Cup_C', 'Cup_D', 'Cup_E', ]
        plt.bar(left, height, width=0.5, color='#377eb8', edgecolor='#377eb8', linewidth=2, tick_label=labels)
        #0096c8
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.ylim(0, 1)
        plt.show()

    def _three_graph(self, p1, p2, p3):
        width = 0.25
        height1 = p1
        height2 = p2
        height3 = p3
        left = np.arange(len(height1))
        labels = ['Bottle_A', 'Bottle_B', 'Bottle_C', 'Bottle_D', 'Bottle_E', 'Stuffed Toy_A', 'Stuffed Toy_B', 'Stuffed Toy_C', 'Stuffed Toy_D', 'Stuffed Toy_E', 'Book_A', 'Book_B', 'Book_C', 'Book_D', 'Book_E', 'Cup_A', 'Cup_B', 'Cup_C', 'Cup_D', 'Cup_E', ]
        plt.bar(left, height1, color='r', width=width, align='center')
        plt.bar(left+width, height2, color='b', width=width, align='center')
        plt.bar(left+2*width, height3, color='g', width=width, align='center')
        plt.xticks(left + width, labels)
        plt.xticks(rotation=90)
        plt.tight_layout()
        # plt.ylim(0, 1)
        plt.show()


    # def visualize_kosoa(self):


    # def visualize_point(self):

    # def visualize_eye(self):

    # def visualize_pointing(self):

    # def image_callback(self, message):

    # def point_cloud_callback(self, point_cloud):

    # def global_pose_callback(self, msg):

    # def get_point(self, x, y):


if __name__ == '__main__':
    rospy.init_node('exophora')
    # get_depth_from_rgb = GetDepthPointFromFrame()
    node = Exophora()
    node.main()
    # rospy.spin()
