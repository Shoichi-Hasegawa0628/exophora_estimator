#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from tf import TransformListener
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import math

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array

class GetPoseLandmark():
    def __init__(self, rgb_image="/hsrb/head_rgbd_sensor/rgb/image_raw/compressed",
                 point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points"):

        self._wrist_x = []
        self._wrist_y = []
        self._wrist_z = []
        self._eye_x = []
        self._eye_y = []
        self._eye_z = []

        self.cv_bridge = CvBridge()

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mesh_drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
        self.mark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=3, circle_radius=1, color=(0, 0, 255))

        self._point_cloud_header = Header()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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

    def landmark_frame(self):
        try:
            height, width, channels = self.rgb_img.shape[:3]
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
                wrist = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * width,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * height)
                eye = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * width,
                       results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * height)
            else:
                wrist = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * width,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * height)
                eye = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * width,
                       results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * height)

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

        # tfによる3次元座標の取得
        if not (((_position_wrist is None) and (_position_eye is None)) or (math.isnan(_position_wrist[0]) is True) or (
                math.isnan(_position_eye[0]) is True)):
            # if not ((_position_wrist is None) and (_position_eye is None)):
            pass
            # tf_buffer = tf2_ros.Buffer()
            # tf_listener = tf2_ros.TransformListener(tf_buffer)
            try:
                # self.trans = self.tf_buffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time.now(), rospy.Duration(1.0))
                self.trans = self.tf_buffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time(0),
                                                             rospy.Duration(1.0))
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
                rospy.loginfo("wrist : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x,
                                                                             transformed.pose.position.y,
                                                                             transformed.pose.position.z))
                self._wrist_x.append(transformed.pose.position.x)
                self._wrist_y.append(transformed.pose.position.y)
                self._wrist_z.append(transformed.pose.position.z)
                # 肘
                _pose_stamped.pose.position.x = _position_eye[0]
                _pose_stamped.pose.position.y = _position_eye[1]
                _pose_stamped.pose.position.z = _position_eye[2]
                # tf
                transformed = tf2_geometry_msgs.do_transform_pose(_pose_stamped, self.trans)
                rospy.loginfo("eye : mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x,
                                                                           transformed.pose.position.y,
                                                                           transformed.pose.position.z))
                self._eye_x.append(transformed.pose.position.x)
                self._eye_y.append(transformed.pose.position.y)
                self._eye_z.append(transformed.pose.position.z)
                return self._eye_x, self._eye_y, self._eye_z, self._wrist_x, self._wrist_y, self._wrist_z
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            except Exception as e2:
                print(f"Unexpected {e2=}, {type(e2)=}")
                return self._eye_x, self._eye_y, self._eye_z, self._wrist_x, self._wrist_y, self._wrist_z
        else:
            print("Skeleton could not be detected.")
            return self._eye_x, self._eye_y, self._eye_z, self._wrist_x, self._wrist_y, self._wrist_z

    def image_callback(self, message):
        self.rgb_img = self.cv_bridge.compressed_imgmsg_to_cv2(message)

    def point_cloud_callback(self, point_cloud):
        self._point_cloud = pointcloud2_to_xyz_array(point_cloud, False)
        self._point_cloud_header = point_cloud.header

    def get_point(self, x, y):
        try:
            return self._point_cloud[y][x]
        except:
            pass

if __name__ == '__main__':
    rospy.init_node('get_pose_landmark')
    get_pose_landmark = GetPoseLandmark()
    # rospy.spin()

