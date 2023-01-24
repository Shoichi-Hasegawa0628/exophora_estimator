#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import statistics


class CalculatePointingVector():
    def __init__(self):
        pass

    def pointing_vector(self, wrist_x, wrist_y, wrist_z, eye_x, eye_y, eye_z):
        # 手首の中央値、肘の中央値
        wrist_frame = np.array([statistics.median(wrist_x), statistics.median(wrist_y), statistics.median(wrist_z)])
        eye_frame = np.array([statistics.median(eye_x), statistics.median(eye_y), statistics.median(eye_z)])

        map_xlim = [-3.5, 3.3]
        map_ylim = [-7.6, 1]
        map_zlim = [0, 3]
        point_vector = np.array([0, 0, 0])

        # 地面との交点
        u = eye_frame[2] / (eye_frame[2] - wrist_frame[2])
        point_ground_x = (1 - u) * eye_frame[0] + u * wrist_frame[0]
        point_ground_y = (1 - u) * eye_frame[1] + u * wrist_frame[1]
        point_ground = np.array([point_ground_x, point_ground_y, 0])
        param = int(u)
        t = 1

        # 天井に当たったときに判定だけやってない
        if ((point_ground_x < map_xlim[0]) or (map_xlim[1] < point_ground_x) or (point_ground_y < map_ylim[0]) or (
                map_ylim[1] < point_ground_y)):
            while True:
                t += 1
                for i in range(3):
                    point_vector[i] = (1 - t) * eye_frame[i] + t * wrist_frame[i]
                if point_vector[0] < map_xlim[0] - 0.5:
                    x3 = map_xlim[0]
                    t = (eye_frame[0] - x3) / (eye_frame[0] - wrist_frame[0])
                    y3 = (1 - t) * eye_frame[1] + t * wrist_frame[1]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[0] > map_xlim[1] + 0.5:
                    x3 = map_xlim[1]
                    t = (eye_frame[0] - x3) / (eye_frame[0] - wrist_frame[0])
                    y3 = (1 - t) * eye_frame[1] + t * wrist_frame[1]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] < map_ylim[0] - 0.5:
                    y3 = map_ylim[0]
                    t = (eye_frame[1] - y3) / (eye_frame[1] - wrist_frame[1])
                    x3 = (1 - t) * eye_frame[0] + t * wrist_frame[0]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    point_ground = np.array([x3, y3, z3])
                    break
                if point_vector[1] > map_ylim[1] + 0.5:
                    y3 = map_ylim[1]
                    t = (eye_frame[1] - y3) / (eye_frame[1] - wrist_frame[1])
                    x3 = (1 - t) * eye_frame[0] + t * wrist_frame[0]
                    z3 = (1 - t) * eye_frame[2] + t * wrist_frame[2]
                    point_ground = np.array([x3, y3, z3])
                    break
            param = int(t)

        return point_ground, param


if __name__ == '__main__':
    calculate_pointing_vector = CalculatePointingVector()
