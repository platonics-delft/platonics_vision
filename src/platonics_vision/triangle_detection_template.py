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

class TriangleTemplate():
    def __init__(self):
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
            cv2.imwrite(f"{self.save_location}.png", cropped_image)
    
    def record(self, img: Image, name='template_test'):
        self.image = img
        self.save_location = name
        # force overwrite
        if os.path.exists(self.save_location):
            os.system(f"rm -r {self.save_location}")
        os.mkdir(self.save_location)
        depth=None
        cv2.imwrite(f"{self.save_location}.png", self.image)

        print("Click and drag to select template")
        print("Press 'q' to quit")
        # Create window and set mouse callback function
        cv2.destroyAllWindows()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.crop_image)
        cv2.imshow("image", self.image)

        # Loop until user presses 'q'
        cropping = False
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


