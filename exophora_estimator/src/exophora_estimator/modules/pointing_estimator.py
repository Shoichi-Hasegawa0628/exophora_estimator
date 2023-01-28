#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
# フォンミーゼス分布
from scipy.stats import vonmises


class PointingEstimator():
    def __init__(self):
        pass

    def pointing_inner_product(self, position, wrist_frame, eye_frame):
        pointing_p = []
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

        pointing_vector = np.array([wrist_frame[0] - eye_frame[0], wrist_frame[1] - eye_frame[1], wrist_frame[2] - eye_frame[2]])
        length_arm = pointing_vector[0] ** 2 + pointing_vector[1] ** 2 + pointing_vector[2] ** 2
        length_pointing_vector = math.sqrt(length_arm)

        # ガウスでいう分散を決めるパラメータ
        vonmises_kappa = 1

        # 物体の3次元マップ座標読み取る
        for i in range(20):
            object_x = float(position[i][2])
            object_y = float(position[i][3])
            object_z = float(position[i][4])

            object_vector = np.array(
                [object_x - eye_frame[0], object_y - eye_frame[1], object_z - eye_frame[2]])
            dis_obj = object_vector[0] ** 2 + object_vector[1] ** 2 + object_vector[2] ** 2
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
            pointing_p.append(probability)
            # self._object_probability.append(probability)

            return pointing_p

if __name__ == '__main__':
    pointing_estimator = PointingEstimator()

