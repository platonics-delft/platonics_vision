#!/bin/python3
import cv2
import rospy
import time
import os
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import rospkg
from panda_ros import Panda
from panda_ros.pose_transform_functions import transformation_2_pose

class SiftTemplate():
    def __init__(self):
        self.panda=Panda()
        self.params = dict()
        self.points = []
        cv2.startWindowThread()

    def crop_image(self, event, x, y, flags, param):
        global mouseX, mouseY, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouseX2, mouseY2 = x, y
            cropping = False
            cv2.imshow("image", self.image)
            y_low = min(mouseY, mouseY2)
            y_high = max(mouseY, mouseY2)
            x_low = min(mouseX, mouseX2)
            x_high = max(mouseX, mouseX2)
            self.params['crop'] = [x_low, x_high, y_low, y_high]
            print(f'{x_low}, {x_high}, {y_low}, {y_high}')

            cropped_image = self.image[y_low: y_high, x_low:x_high]
            cv2.imshow("cropped", cropped_image)
            cv2.imwrite(f"{self.save_dir}/template/template.png", cropped_image)

    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point when left mouse button is clicked
            self.points.append([x, y])
            # Draw the selected points
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self._window_name, self.image)
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 2:
            # If right button is clicked after at least 3 points, close the polygon
            cv2.polylines(self.image, [np.array(self.points)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imshow(self._window_name, self.image)
            #cv2.imwrite(f"{self.save_dir}/template/polygon.png", self.image)
    
    def record(self, img: Image, depth_img: Image, name='template_test', panda_link_tf=np.eye(4)):
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        self.depth_image = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")

        self.save_dir = name
        self.params['template_path'] = self.save_dir + "/full_image.png"
        # force overwrite
        if os.path.exists(self.save_dir):
            os.system(f"rm -r {self.save_dir}")
        os.mkdir(self.save_dir)
        os.mkdir(self.save_dir + "/template")
        cv2.imwrite(f"{self.save_dir}/template/full_image.png", self.image)
        np.save(f"{self.save_dir}/template/depth.npy", self.depth_image)

        print("Click and drag to select template")
        print("Press 'q' to quit")
        # Create window and set mouse callback function
        #cv2.destroyAllWindows()
        # random name to avoid conflict
        self._window_name = "image" + str(time.time())
        cv2.namedWindow(self._window_name )
        #cv2.setMouseCallback("image", self.crop_image)
        cv2.setMouseCallback(self._window_name , self.select_points)
        cv2.startWindowThread()
        cv2.imshow(self._window_name , self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.crop_image)
        cv2.imshow("image", self.image)

        # Loop until user presses 'q'
        """
        cropping = False
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(self._window_name)
                #cv2.destroyAllWindows()
                break
        """
            
        # Create mask from polygon points
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points)], 255)
        self.params['polygon'] = self.points

        # Apply mask to depth image and get median
        depth_row = self.depth_image[mask == 255].reshape(-1)
        self.params['depth'] = float(np.median(depth_row))

        #depth_row=self.depth_image[self.params['crop'][2]:self.params['crop'][3], self.params['crop'][0]:self.params['crop'][1]].reshape(-1)
        #self.params['depth'] = float(np.median(depth_row))

        self.params['position']={'x': float(self.panda.curr_pos[0]), 'y': float(self.panda.curr_pos[1]), 'z': float(self.panda.curr_pos[2])}
        self.params['orientation']={'w': float(self.panda.curr_ori[0]) ,'x': float(self.panda.curr_ori[1]) , 'y': float(self.panda.curr_ori[2]), 'z': float(self.panda.curr_ori[3])}
        panda_link_tf_pose = transformation_2_pose(panda_link_tf)
        self.params['panda_link_tf'] = {
            'position': {
                'x': float(panda_link_tf_pose.pose.position.x),
                'y': float(panda_link_tf_pose.pose.position.y),
                'z': float(panda_link_tf_pose.pose.position.z)
            },
            'orientation': {
                'w': float(panda_link_tf_pose.pose.orientation.w),
                'x': float(panda_link_tf_pose.pose.orientation.x),
                'y': float(panda_link_tf_pose.pose.orientation.y),
                'z': float(panda_link_tf_pose.pose.orientation.z)
            }
        }
        with open(f"{self.save_dir}/template/params.yaml", 'w') as file:
            yaml.dump(self.params, file)


