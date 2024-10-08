import rospy
from typing import List
import os
import math
import numpy as np
import time
import rospkg
from copy import deepcopy
import pickle
import cv2
from cv_bridge import CvBridgeError, CvBridge
import datetime
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

class TriangleDetector():
    def __init__(self):
        self.detection_score_threshold = 0.8
        self.cv2_text_label_font_ = cv2.FONT_HERSHEY_SIMPLEX
        self.cv2_text_label_font_scale_ = 0.35
        self.template_matching_method = cv2.TM_CCOEFF_NORMED
        self.text_label_colors_ = dict(zip(['red', 'yellow', 'green', 'lcd'], 
                                    [(0, 0, 255), (255, 255, 255), 
                                    (0, 255, 0), (255, 0, 0)]))

        self.acceptable_camera_delay_steps = 2

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.slider_distance_publisher = rospy.Publisher('/triangle_detector_distance', Float32, queue_size=0)
        self.slider_solver_image_publisher = rospy.Publisher('/triangle_detection_image', Image, queue_size=0)
        self.slider_solver_hsv_image_publisher = rospy.Publisher('/triangle_detector_hsv_image', Image, queue_size=0)

        self.bridge = CvBridge()
        self.template_images_dict = None
        package_path = rospkg.RosPack().get_path('platonics_vision')
        self.image_dir_path = package_path + '/data/triangle_templates/'

        #self.load_template_images(self.image_dir_path, object_ids=['red', 'white_center'])

        rospy.sleep(1)

    def image_callback(self, msg):
        # Convert the ROS message to a OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            self.curr_image = cv_image
        except CvBridgeError as e:
            print(e)

    def load_template_images(self, image_dir_path, 
                         object_ids=['red', 'yellow', 'lcd'],
                         debug=False):
        """
        Loads all images that can be uses as templates for the given object_ids
        from the image_dir_path directory.

        Note: expects all images within image_dir_path whose filename contains the
        substring "template" to be candidates for extracting the templates.

        Parameters
        ----------
        image_dir_path: str
            Path to directory containing candidate template images.
        object_ids: list
            Template object ID strings (included in filenames).
        debug: bool
            Whether to print some debugging messages

        Returns
        -------
        template_images_dict: dict
            Mapping between object IDs and lists of ndarrays containing loaded images.
        """

        # Load template images:
        self.template_images_dict = {}

        image_filename_list = [filename for filename in os.listdir(image_dir_path) \
                                if 'template' in filename]


        for object_id in object_ids:
            if object_id not in self.template_images_dict.keys():
                self.template_images_dict[object_id] = []
            # Grab first valid image for each template:
            for image_filename in image_filename_list:
                # if image_filename is folder skip
                if os.path.isdir(os.path.join(image_dir_path, image_filename)):
                    continue
                if object_id in image_filename:
                    if debug:
                        print(f'[DEBUG] Using image for {object_id}: ' + \
                            f' {image_filename}')
                    image_path = os.path.join(image_dir_path, image_filename)
                    image_array = cv2.imread(image_path)  # Note: in BGR format
                    self.template_images_dict[object_id].append(image_array)
            if not image_filename_list:
                raise AssertionError(f'Unable to extract a single template' + \
                                    f' for object {object_id}')
            self.initial_point_id, self.goal_point_id = object_ids[0], object_ids[1]
    
    def detect_triangles(self, debug=False, publish_visual_output=True):
        # Code for template matching heavily based on slider_solver_node from 
        # https://github.com/eurobin-wp1/tum-tb-perception
        
        # Detect templates in input image:
        errorflag = 0
        vis_image_array = self.curr_image.copy()
        template_positions_dict = {}
        detection_scores_dict = {}

        for template_id, temp_image_array_list in self.template_images_dict.items():
            for temp_image_array in temp_image_array_list:
                # Apply template Matching
                # Source: https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
                res = cv2.matchTemplate(self.curr_image, temp_image_array, 
                                        self.template_matching_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if debug:
                    print(f'\n[DEBUG] Checking for template: {template_id}')
                    print(f'[DEBUG] Current score: {max_val}')

                if max_val < self.detection_score_threshold:
                    if debug:
                        print(f'[DEBUG] Could not find a detection with a' + \
                                f' score that is higher than threshold ' + \
                                f' {self.detection_score_threshold}.')
                        print(f'[DEBUG] Trying another template source...')
                else:
                    if debug:
                        print(f'[DEBUG] Suitable detection found!')
                    detection_scores_dict[template_id] = max_val

                    w, h = (temp_image_array.shape[1], temp_image_array.shape[0])
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    centroid = (int(top_left[0] + (w / 2.)), 
                                int(top_left[1] + (h / 2.)))
                    template_positions_dict[template_id] = centroid

                    if publish_visual_output:
                        # Annotate image with BB and centroid point:
                        cv2.rectangle(vis_image_array, top_left, bottom_right,
                                        color=self.text_label_colors_[template_id], 
                                        thickness=2)
                        cv2.circle(vis_image_array, centroid, 1, 
                                    color=(0., 0., 0.), thickness=1)

                        # Annotate image with faded text labels:
                        overlay = np.copy(vis_image_array)
                        overlay = cv2.rectangle(overlay,
                                                (top_left[0], top_left[1] - 15),
                                                (top_left[0] + w, top_left[1]),
                                                self.text_label_colors_[template_id], -1)
                        overlay = cv2.putText(overlay, template_id,
                                                (top_left[0], top_left[1] - 5),
                                                self.cv2_text_label_font_, 
                                                self.cv2_text_label_font_scale_,
                                                (0., 0., 0.), 1)
                        alpha = 0.5
                        cv2.addWeighted(overlay, alpha, vis_image_array,
                                        1 - alpha, 0, vis_image_array)
                    break
            else:
                rospy.logwarn(f'[slider_task_solver] "{template_id}" ' + \
                                f'could not be detected!')

        if debug:
            print(f'[DEBUG] Template positions: {template_positions_dict}')
            print(f'[DEBUG] Detection scores: {detection_scores_dict}')

        # Estimate distance between positions in image space:
        estimated_pixel_distance = {'red_green': 0, 'red_yellow': 0, 'green_yellow': 0}
        positions = {}
        for color in ['red', 'yellow', 'green']:
            if color not in template_positions_dict.keys():
                rospy.logerr(f'[slider_task_solver] Could not find ' + \
                                f'position for color {color}!')
                positions[color] = None
            else:
                positions[color] = template_positions_dict[color]
        for distance in list(estimated_pixel_distance.keys()):
            color1, color2 = distance.split('_')
            if positions[color1] is not None and positions[color2] is not None:
                estimated_pixel_distance[distance] = positions[color2][0] - positions[color1][0]
            else:
                rospy.logerr(f'[slider_task_solver] Could not estimate ' + \
                                f'distance between {color1} and {color2}!')
                errorflag = -1
                estimated_pixel_distance[distance] = None



        if estimated_pixel_distance is None:
            rospy.logerr(f'[slider_task_solver] Could not estimate ' + \
                            f'slider motion distance!')

        if publish_visual_output:
            debug_image_msg = self.bridge.cv2_to_imgmsg(vis_image_array, 
                                                    encoding="bgr8")

            debug_image_msg.header.stamp = rospy.Time.now()
            self.slider_solver_image_publisher.publish(debug_image_msg)

        return estimated_pixel_distance, errorflag
