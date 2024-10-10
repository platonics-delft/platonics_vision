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
import copy

class SliderSolver():
    def __init__(self):
        self.control_rate = 30
        self.rate=rospy.Rate(self.control_rate)
        self.move_increment = 0.001
        self.triangles_distance = None

        self.robot = Panda()
        self._triangle_detector = TriangleDetector()

        self.safe_distance_lin=0.005
        self.safe_distance_ori=0.020

        self.success_threshold = 5
        self.green_2_yellow_threshold = 5
        self.maximum_distance_2_previous_yellow = 10

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
        object_ids = ['screen','red', 'green', 'yellow']
        self._triangle_detector.load_template_images(self.image_dir_path, object_ids, debug=True)
        self.end = False

    def execute_step(self, task_stage) -> int:
        ### Safety check
        if self.robot.change_in_safety_check:
            if self.robot.safety_check:
                    self.robot.set_stiffness(self.robot.K_pos, self.robot.K_pos, self.robot.K_pos, self.robot.K_ori, self.robot.K_ori, self.robot.K_ori, 0)
            else:
                # print("Safety violation detected. Making the robot compliant")
                self.robot.set_stiffness(self.robot.K_pos_safe, self.robot.K_pos_safe, self.robot.K_pos_safe, self.robot.K_ori_safe, self.robot.K_ori_safe, self.robot.K_ori_safe, 0)
                return True
            
        self.triangles_positions, errorflag = self._triangle_detector.detect_triangles(debug=True)
        # rospy.loginfo(f"Positions: {self.triangles_positions}")
        if errorflag == -1:
            self.rate.sleep()
            return -1
        else:
            if task_stage == 1:
                if len(self.triangles_positions['yellow']) == 0 or len(self.triangles_positions['red']) == 0:
                    return -1 
                best_centroid_yellow = self.triangles_positions['yellow'][0]
                best_centroid_red = self.triangles_positions['red'][0]
                self.triangles_distance = best_centroid_yellow[0] - best_centroid_red[0]
                if abs(self.triangles_distance) < self.success_threshold:
                    return 0
            elif task_stage == 2:
                best_centroid_yellow = None
                best_centroid_green = None
                if len(self.triangles_positions['red']) == 0 or len(self.triangles_positions['green']) == 0:
                    return -1
                for match_centroid_yellow in self.triangles_positions['yellow']:
                    distance_previous = np.linalg.norm(np.array(match_centroid_yellow) - np.array(self.previous_best_centroid_yellow))
                    if distance_previous >= self.maximum_distance_2_previous_yellow: 
                        continue
                    else:
                        best_centroid_yellow = match_centroid_yellow
                        break
                if best_centroid_yellow is None:
                    rospy.logerr(f"Current yellow distance of {distance_previous} to previous position is too large")
                    return -1
                
                for match_centroid_green in self.triangles_positions['green']:
                    distance_green_yellow = np.linalg.norm(np.array(match_centroid_green) - np.array(best_centroid_yellow))
                    if distance_green_yellow <= self.green_2_yellow_threshold:
                        continue
                    else:
                        best_centroid_green = match_centroid_green
                        break
                if best_centroid_green is None:
                    rospy.logerr("No green detection far away from yellow")
                    return -1
                
                best_centroid_red = self.triangles_positions['red'][0]
                self.triangles_distance = best_centroid_green[0] - best_centroid_red[0]
                # if abs(self.triangles_distance['green_yellow']) < 5:
                #     return 
                if abs(self.triangles_distance) < self.success_threshold:
                    return 0
            else:
                rospy.logerr("Invalid task stage")
                return -1
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
        if best_centroid_yellow is not None:
            self.previous_best_centroid_yellow = best_centroid_yellow

        self.rate.sleep()
        return 1
