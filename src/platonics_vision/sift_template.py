#!/bin/python3
import cv2
import rospy
import os
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import rospkg
from panda_ros import Panda

class SiftTemplate():
    def __init__(self):
        self.panda=Panda()
        self.params = dict()
        self.bridge = CvBridge()

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
            cv2.imwrite(f"{self.save_dir}/template.png", cropped_image)
    
    def record(self, img: Image, depth_img: Image, name='template_test'):
        self.image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        self.save_dir = name
        self.params['template_path'] = self.save_dir + "/full_image.png"
        os.mkdir(self.save_dir)
        depth=None
        cv2.imwrite(f"{self.save_dir}/full_image.png", self.image)
        cv2.imwrite(f"{self.save_dir}/depth.png", self.depth_image)

        print("Click and drag to select template")
        print("Press 'q' to quit")
        # Create window and set mouse callback function
        cv2.destroyAllWindows()
        cv2.setMouseCallback("image", self.crop_image)
        cv2.imshow("image", self.image)

        # Loop until user presses 'q'
        cropping = False
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
        depth_row=self.depth_image[self.params['crop'][2]:self.params['crop'][3], self.params['crop'][0]:self.params['crop'][1]].reshape(-1)
        self.params['depth'] = float(np.median(depth_row))

        self.params['position']={'x': float(self.panda.curr_pos[0]), 'y': float(self.panda.curr_pos[1]), 'z': float(self.panda.curr_pos[2])}
        self.params['orientation']={'w': float(self.panda.curr_ori[0]) ,'x': float(self.panda.curr_ori[1]) , 'y': float(self.panda.curr_ori[2]), 'z': float(self.panda.curr_ori[3])}
        with open(f"{self.save_dir}/params.yaml", 'w') as file:
            yaml.dump(self.params, file)


