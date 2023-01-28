#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv

class Dataset():
    def __init__(self):
        pass

    def load_object_position_data(self, object_category):
        # 物体の3次元座標のロード
        files = os.listdir("/root/HSR/catkin_ws/src/exophora_estimator/exophora_estimator/data/objects_position/")
        datas = []
        data = []
        for i in range(len(files)):
            with open("/root/HSR/catkin_ws/src/exophora_estimator/exophora_estimator/data/objects_position/object_position_{}.csv".format(str(i+1))) as f:
                reader = csv.reader(f)
                for row in reader:
                    data.append(row)
            datas.append(data)
            data = []
        number = 0
        for k in range(len(object_category)):
            for i in range(len(datas)):
                for j in range(len(datas[i])):
                    if(datas[i][j][1] == object_category[k]):
                        datas[i][j][0] = number
                        data.append(datas[i][j])
                        number += 1
            # print(datas)
        return data

    def load_object_confidence(self, object_category, target_object_id):
        # confを取得
        # files = os.listdir("/root/HSR/catkin_ws/src/mediapipe_ros/data/objects_position/")
        data = []
        datas = []
        with open("/root/HSR/catkin_ws/src/exophora_estimator/exophora_estimator/src/exophora_estimator/object_conf.csv") as f3:
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
                data.append([row[0], plist_np[target_object_id]])

            for k in range(len(object_category)):
                for i in range(len(data)):
                    if(data[i][0] == object_category[k]):
                        datas.append(data[i][1])

        return datas


if __name__ == '__main__':
    dataset = Dataset()


