import rospy
from typing import List
import os
import math
import numpy as np
import time
import rospkg
from geometry_msgs.msg import PoseStamped
from panda_ros import Panda
from panda_ros.pose_transform_functions import transform_pose, pos_quat_2_pose_st, pose_st_2_transformation
from copy import deepcopy 
from triangle_detector import TriangleDetector

class SliderSolver():
    def __init__(self):
        self.control_rate = 30
        self.rate=rospy.Rate(self.control_rate)
        self.move_increment = 0.0001

        self.robot = Panda()
        self.triangle_detector = TriangleDetector()

        self.safe_distance_lin=0.005
        self.safe_distance_ori=0.020

        self.success_threshold = 5

        self.acceptable_camera_delay_steps = 2

        package_path = rospkg.RosPack().get_path('platonics_vision')
        self.image_dir_path = package_path + '/data/triangle_templates'

        self.goal_pose_sub = rospy.Subscriber('/equilibrium_pose', PoseStamped, self.ee_pos_goal_callback)
        self.goal_pose = None

        rospy.sleep(1)

    def ee_pos_goal_callback(self, goal_conf):
        self.goal_pose = goal_conf

    def execute_start(self, task_stage):
        rospy.loginfo("Executing trajectory.")
        self.rate=rospy.Rate(self.control_rate)

        self.robot.set_stiffness(3000, 3000, 3000, 30, 30, 30, 0)
        self.robot.set_K.update_configuration({"max_delta_lin": 0.05})
        self.robot.set_K.update_configuration({"max_delta_ori": 0.50}) 
        self.robot.set_K.update_configuration({"joint_default_damping": 0.00})

        self.robot.change_in_safety_check = False
        if task_stage == 1:
            object_ids = ['red', 'white_center']
        if task_stage == 2:
            object_ids = ['red', 'green']
        self.triangle_detector.load_template_images(self.image_dir_path, object_ids, debug=True)
        self.end = False

    def execute_step(self, task_stage) -> bool:
        ### Safety check
        if self.robot.change_in_safety_check:
            if self.robot.safety_check:
                    self.robot.set_stiffness(self.robot.K_pos, self.robot.K_pos, self.robot.K_pos, self.robot.K_ori, self.robot.K_ori, self.robot.K_ori, 0)
            else:
                # print("Safety violation detected. Making the robot compliant")
                self.robot.set_stiffness(self.robot.K_pos_safe, self.robot.K_pos_safe, self.robot.K_pos_safe, self.robot.K_ori_safe, self.robot.K_ori_safe, self.robot.K_ori_safe, 0)
                return True
            
        self.triangles_distance = self.triangle_detector.detect_triangles(debug=True)
        if self.triangles_distance is not None:
            direction = np.sign(self.triangles_distance)
            if self.goal_pose is not None:
                curr_goal = self.robot.goal_pose
            else:
                curr_goal = self.robot.curr_pose
            new_goal_in_ee_frame = pos_quat_2_pose_st(np.array([0., direction*self.move_increment, 0.]), np.quaternion(1.0,0.,0.,0.))
            transform_world_ee = pose_st_2_transformation(curr_goal)
            new_goal_in_world_frame = transform_pose(new_goal_in_ee_frame, transform_world_ee)

            ### Publish the goal pose
            new_goal_in_world_frame.header.seq = 1
            new_goal_in_world_frame.header.stamp = rospy.Time.now()
            new_goal_in_world_frame.header.frame_id = "panda_link0" 
            self.robot.goal_pub.publish(new_goal_in_world_frame) 

            if np.abs(self.triangles_distance) < self.success_threshold:
                return True

        self.rate.sleep()
        return False