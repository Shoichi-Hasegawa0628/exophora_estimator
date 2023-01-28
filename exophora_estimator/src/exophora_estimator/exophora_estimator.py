#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROS
import rospy
from sensor_msgs.msg import CompressedImage, Image

# Standard
import numpy as np
import yaml
# from PIL import Image
# import matplotlib.pyplot as plt

# Self
from modules import (
    calculate_pointing_vector,
    dataset,
    demonstrative_region_estimator,
    get_pose_landmark,
    pointing_estimator,
    visualize_pointing_trajectory
)

calculate_pointing_func = calculate_pointing_vector.CalculatePointingVector()
dataset_func = dataset.Dataset()
demonstrative_estimator_func = demonstrative_region_estimator.DemonstrativeRegionEstimator()
get_pose_landmark_func = get_pose_landmark.GetPoseLandmark()
pointing_estimator_func = pointing_estimator.PointingEstimator()
visualize_pointing_func = visualize_pointing_trajectory.VisualizePointingTrajectory()


class ExophoraEstimator():
    def __init__(self):
        self.object_category = ["Bottle", "Stuffed Toy", "Book", "Cup"]
        self.target_object_name = "Bottle"

        with open('yolov5m_Object365.yaml', 'r') as yml:
            object_yaml = yaml.load(yml, Loader=yaml.SafeLoader)
            self.object_365 = object_yaml['names']

        self.kosoa = "a"
        # self.sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed", CompressedImage, self.main)
        self.img_pub = rospy.Publisher("/annotated_msg/compressed", CompressedImage, queue_size=1)


    ### non_realtimeとrealtimeで存在する関数と存在しないものがあるため、両方を参照しながらプログラムを作成
    # 基本的には、
    def main(self):
        # 物体のインデックスとカテゴリと3次元座標のリスト作成
        # リストの中身は [object number, object name, x, y, z] が20個
        object_list = dataset_func.load_object_position_data(self.object_category)

        # リストに6個の座標が格納されたらループを抜ける
        num = 0
        while num < 7:
            eye_x, eye_y, eye_z, wrist_x, wrist_y, wrist_z = get_pose_landmark_func.landmark_frame()
            num = len(eye_x)

        rospy.loginfo("Skeleton could be detected!")

        # 指差しベクトルの計算と可視化
        point_ground, param, wrist_frame, eye_frame = calculate_pointing_func.calculate_pointing_vector(wrist_x, wrist_y, wrist_z, eye_x, eye_y, eye_z)
        visualize_pointing_func.visualize_point(point_ground)
        visualize_pointing_func.visualize_eye(eys_frame)
        visualize_pointing_func.visualize_pointing(param, wrist_frame, eye_frame)

        # フォン・ミーゼス分布を用いた指差し方向に基づく確率推定器により、対象確率の出力
        # for i in range(len(object_position)):
        pointing_prob = pointing_estimator_func.pointing_inner_product(object_list, wrist_frame, eye_frame)

        # 指示語領域に基づく確率推定器により、対象確率の出力
        demonstrative_prob = demonstrative_estimator_func.calculate_demonstrative_region(object_list, wrist_frame, eye_frame, self.kosoa)
        # self.visualize_kosoa()

        # 物体カテゴリの信頼度スコアを取得 (物体カテゴリに基づく確率推定器)
        self.target_object_id = self.object_365.index(self.target_object_name)
        object_class_prob = dataset_func.load_conf(self.object_category, self.target_object_id)

        demonstrative_prob = np.array(demonstrative_prob)
        pointing_prob = np.array(pointing_prob)


        # 3つの確率値を掛け合わせる
        target_probability = object_class_prob * demonstrative_prob * pointing_prob
        # target_probability = demonstrative_prob * pointing_prob
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


if __name__ == '__main__':
    rospy.init_node('exophora_estimator')
    node = ExophoraEstimator()
    node.main()
    # rospy.spin()

