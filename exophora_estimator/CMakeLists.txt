cmake_minimum_required(VERSION 2.8.3)
project(exophora_estimator)

find_package(catkin REQUIRED COMPONENTS
	rospy
)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   GaussianDistribution.msg
 )

