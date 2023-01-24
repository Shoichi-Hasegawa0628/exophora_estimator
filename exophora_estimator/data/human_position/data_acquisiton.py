#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
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

from PIL import Image
import matplotlib.pyplot as plt

# csv
import csv

# 下4つはmediapipe用
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array

# rosbag record /hsrb/head_rgbd_sensor/rgb/image_raw/compressed /hsrb/head_rgbd_sensor/depth_registered/rectified_points /global_pose /tf

class DataAcquisition():

    def __init__(self, rgb_image="/hsrb/head_rgbd_sensor/rgb/image_raw/compressed", point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points", global_pose_topic = "/global_pose"):

        self._object_class_p = []
        self._demonstrative_p = []
        self._visual_indicate_p = []

        self._wrist_x = []
        self._wrist_y = []
        self._wrist_z = []
        self._eye_x = []
        self._eye_y = []
        self._eye_z = []
        self._l_shoulder_x = []
        self._l_shoulder_y = []
        self._l_shoulder_z = []
        self._r_shoulder_x = []
        self._r_shoulder_y = []
        self._r_shoulder_z = []

        self._point_ground = np.array([0, 0, 0])

        # ["Bottle", "Stuffed Toy", "Book", "Cup"]
        # 0,1,2,3,4   5,6,7,8,9    10,11,12,13,14    15,16,17,18,19

        # 記録する
        self._kosoa = "a"
        # ko, so, a
        self._object_category = "Bottle"
        #self._object_category = "Stuffed Toy"
        #self._object_category = "Book"
        #self._object_category = "Cup"
        self._object_idx = 2
        # self._position = 1　(いらない)

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
        time.sleep(1)

        # リストに6個の座標が格納されたらループを抜ける
        # while len(self._eye_x) < 7:
        #     self.landmark_frame()

        # 20秒経過かリストの中身が6コになったら終了
        time_start = time.perf_counter()
        t = 0
        while (len(self._eye_x) < 7):
            print(t)
            self.landmark_frame()
            time_end = time.perf_counter()
            t = time_end - time_start
            if t > 10:
                break

        # rospy.loginfo("Skeleton could be detected!!!!!!")

        self.data_input()

        # 指差しベクトル
        if ((len(self._wrist_z) >= 5) and (len(self._eye_z) >= 5)):
            self.pointing_vector()
            self.visualize_point()
            self.visualize_eye()
            self.visualize_pointing()

        # # 指差しから各物体の確率
        # self.pointing_inner_product()

        # # こそあ領域、確率
        # self.kosoa()
        # self.visualize_kosoa()


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


    def landmark_frame(self):
        try:
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            height, width, channels = self.rgb_img.shape[:3]
            # print("aaaaaaaaaaa")
            # ここからmediapipe追加部分
            # 検出結果を別の画像名としてコピーして作成
            self.annotated_image = self.rgb_img.copy()
            with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                results = pose.process(self.rgb_img)

                l_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * height
                r_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * height

            if l_wrist_y < r_wrist_y:
                wrist = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * height)
                eye = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * height)
            else:
                wrist = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * height)
                eye = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * height)

            # 両肩の画像座標
            l_shoulder = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * height)
            r_shoulder = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * width, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * height)

            wrist_x = int(wrist[0])
            wrist_y = int(wrist[1])
            eye_x = int(eye[0])
            eye_y = int(eye[1])
            l_shoulder_x = int(l_shoulder[0])
            l_shoulder_y = int(l_shoulder[1])
            r_shoulder_x = int(r_shoulder[0])
            r_shoulder_y = int(r_shoulder[1])

            _position_wrist = self.get_point(wrist_x, wrist_y)
            _position_eye = self.get_point(eye_x, eye_y)
            _position_l_shoulder = self.get_point(l_shoulder_x, l_shoulder_y)
            _position_r_shoulder = self.get_point(r_shoulder_x, r_shoulder_y)

        except Exception as e1:
            print(f"Unexpected {e1=}, {type(e1)=}")
            return

        self.trans = self.tf_buffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time(0), rospy.Duration(1.0))
        _pose_stamped = PoseStamped()
        _pose_stamped.header.frame_id = 'map'

        if not ((_position_wrist is None) or (math.isnan(_position_wrist[0]) is True)):
            try:
                _pose_stamped.pose.position.x = _position_wrist[0]
                _pose_stamped.pose.position.y = _position_wrist[1]
                _pose_stamped.pose.position.z = _position_wrist[2]
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                # rospy.loginfo("wrist : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z))
                self._wrist_x.append(transformed.pose.position.x)
                self._wrist_y.append(transformed.pose.position.y)
                self._wrist_z.append(transformed.pose.position.z)
            except Exception as e:
                print(f"Unexpected {e=}, {type(e)=}")
                return

        if not ((_position_eye is None) or (math.isnan(_position_eye[0]) is True)):
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            try:
                _pose_stamped.pose.position.x = _position_eye[0]
                _pose_stamped.pose.position.y = _position_eye[1]
                _pose_stamped.pose.position.z = _position_eye[2]
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                # rospy.loginfo("eye : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z))
                self._eye_x.append(transformed.pose.position.x)
                self._eye_y.append(transformed.pose.position.y)
                self._eye_z.append(transformed.pose.position.z)
            except Exception as e:
                print(f"Unexpected {e=}, {type(e)=}")
                return

        if not ((_position_l_shoulder is None) or (math.isnan(_position_l_shoulder[0]) is True)):
            try:
                _pose_stamped.pose.position.x = _position_l_shoulder[0]
                _pose_stamped.pose.position.y = _position_l_shoulder[1]
                _pose_stamped.pose.position.z = _position_l_shoulder[2]
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                # rospy.loginfo("eye : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z))
                self._l_shoulder_x.append(transformed.pose.position.x)
                self._l_shoulder_y.append(transformed.pose.position.y)
                self._l_shoulder_z.append(transformed.pose.position.z)
            except Exception as e:
                print(f"Unexpected {e=}, {type(e)=}")
                return

        if  not ((_position_r_shoulder is None) or (math.isnan(_position_r_shoulder[0]) is True)):
            try:
                _pose_stamped.pose.position.x = _position_r_shoulder[0]
                _pose_stamped.pose.position.y = _position_r_shoulder[1]
                _pose_stamped.pose.position.z = _position_r_shoulder[2]
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                # rospy.loginfo("eye : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z))
                self._r_shoulder_x.append(transformed.pose.position.x)
                self._r_shoulder_y.append(transformed.pose.position.y)
                self._r_shoulder_z.append(transformed.pose.position.z)
            except Exception as e:
                print(f"Unexpected {e=}, {type(e)=}")
                return

        # こっからtf
        # if not (((_position_wrist is None) and (_position_eye is None)) or (math.isnan(_position_wrist[0]) is True) or (math.isnan(_position_eye[0]) is True)):

    def data_input(self):

        # テキストファイルの書き込み（新規作成・上書き）
        # 書き込み用でファイルオープン: 引数mode='w'
        # 文字列を書き込み: write()
        # リストを書き込み: writelines()

        if len(self._wrist_z) < 5:
            self.median_wrist_x = np.nan
            self.median_wrist_y = np.nan
            self.median_wrist_z = np.nan
        else:
            # 手首の中央値
            self.median_wrist_x = statistics.median(self._wrist_x)
            self.median_wrist_y = statistics.median(self._wrist_y)
            self.median_wrist_z = statistics.median(self._wrist_z)
            rospy.loginfo("wrist_median \n cX:{} cY:{} cZ:{}".format(self.median_wrist_x, self.median_wrist_y, self.median_wrist_z))
        wrist_map = [self.median_wrist_x, self.median_wrist_y, self.median_wrist_z]

        if len(self._eye_z) < 5:
            self.median_eye_x = np.nan
            self.median_eye_y = np.nan
            self.median_eye_z = np.nan
        else:
            # 目の中央値
            self.median_eye_x = statistics.median(self._eye_x)
            self.median_eye_y = statistics.median(self._eye_y)
            self.median_eye_z = statistics.median(self._eye_z)
            rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_eye_x, self.median_eye_y, self.median_eye_z))
        eye_map = [self.median_eye_x, self.median_eye_y, self.median_eye_z]

        if len(self._l_shoulder_z) < 5:
            self.median_l_shoulder_x = np.nan
            self.median_l_shoulder_y = np.nan
            self.median_l_shoulder_z = np.nan
        else:
            # 左肩の中央値
            self.median_l_shoulder_x = statistics.median(self._l_shoulder_x)
            self.median_l_shoulder_y = statistics.median(self._l_shoulder_y)
            self.median_l_shoulder_z = statistics.median(self._l_shoulder_z)
            rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_l_shoulder_x, self.median_l_shoulder_y, self.median_l_shoulder_z))
        l_shoulder_map = [self.median_l_shoulder_x, self.median_l_shoulder_y, self.median_l_shoulder_z]

        if len(self._r_shoulder_z) < 5:
            self.median_r_shoulder_x = np.nan
            self.median_r_shoulder_y = np.nan
            self.median_r_shoulder_z = np.nan
        else:
            # 右肩の中央値
            self.median_r_shoulder_x = statistics.median(self._r_shoulder_x)
            self.median_r_shoulder_y = statistics.median(self._r_shoulder_y)
            self.median_r_shoulder_z = statistics.median(self._r_shoulder_z)
            rospy.loginfo("eye_median \n cX:{} cY:{} cZ:{}".format(self.median_r_shoulder_x, self.median_r_shoulder_y, self.median_r_shoulder_z))
        r_shoulder_map = [self.median_r_shoulder_x, self.median_r_shoulder_y, self.median_r_shoulder_z]

        # global pose
        global_pose = [self._global_pose_x, self._global_pose_y, self._global_pose_z]

        with open('wrist_eye_miss_pointing.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([wrist_map, eye_map, l_shoulder_map, r_shoulder_map, global_pose, self._kosoa, self._object_category, self._object_idx])

    def pointing_vector(self):
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

    # def kosoa(self):

    # 指示語領域の可視化(ヒートマップ)
    # def visualize_kosoa(self):


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
        marker.lifetime = rospy.Duration(60)
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
        marker.lifetime = rospy.Duration(60)
        # marker.ns = "Goal-%u"%i
        # self.marker_msg.markers.append(marker)
        # self.marker_msg.lifetime = rospy.Duration(10)
        # self.point_pub.publish(self.marker_msg)
        self.eye_pub.publish(marker)
        # self.rate.sleep()

    def visualize_pointing(self):
        self.marker_array_msg = MarkerArray()
        i = 0
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
            marker.lifetime = rospy.Duration(60)
            marker.ns = "Goal-%u"%t
            self.marker_array_msg.markers.append(marker)
            # i += 1
        # self.marker_array_msg.lifetime = rospy.Duration(10)
        self.vector_pub.publish(self.marker_array_msg)
        # self.rate.sleep()


    def image_callback(self, message):
        # rospy.loginfo("Entered callback")
        self.rgb_img = self.cv_bridge.compressed_imgmsg_to_cv2(message)

    def point_cloud_callback(self, point_cloud):
        # print("aaaaaaaaaaaaaaaa")
        self._point_cloud = pointcloud2_to_xyz_array(point_cloud, False)
        # rospy.loginfo("PointCloud frame id : {}".format(point_cloud.header.frame_id))
        self._point_cloud_header = point_cloud.header

    def global_pose_callback(self, msg):
        self._global_pose_x = msg.pose.position.x
        self._global_pose_y = msg.pose.position.y
        self._global_pose_z = msg.pose.position.z

    def get_point(self, x, y):
        try:
            # print("aaaaaaaaaaaaaaaaaaaaaa")
            return self._point_cloud[y][x]
        except:
            # rospy.loginfo("GET POINT ERROR")
            pass



if __name__ == '__main__':
    rospy.init_node('data_acquisition')
    # get_depth_from_rgb = GetDepthPointFromFrame()
    node = DataAcquisition()
    node.main()
    # rospy.spin()
