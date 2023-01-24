#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from tf import TransformListener
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import time
import numpy as np
import statistics
import math


class ExophoraEstimator():
    def __init__(self):
        pass


if __name__ == '__main__':
    rospy.init_node('exophora_estimator')
    node = ExophoraEstimator()
    node.main()
    # rospy.spin()

