import numpy as np
import cv2
import logging
from geometry_msgs.msg import Pose

from panda_ros.pose_transform_functions import pose_2_transformation
MIN_MATCH_COUNT = 2


class Localizer(object):
    _logger: logging.Logger

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.WARN)

    def set_cropping(self, cropping) -> None:
        self._cropping = cropping

    def set_box_depth(self, depth) -> None:
        self._box_depth = depth

    def set_log_level(self, level: int):
        self._logger.setLevel(level)

    def set_depth_template(self, depth_template_filename) -> None:
        self._depth_template = np.load(depth_template_filename)
        cv2.imwrite('/home/mspahn/temp/depth_template.png', self._depth_template)

    def set_template(self, template) -> None:
        assert isinstance(template, str)
        self._full_template = cv2.imread(template, 0)
        cropped_h = self._cropping[2:]
        cropped_w = self._cropping[:2]

        self._template = self._full_template[
            cropped_h[0] : cropped_h[1], cropped_w[0] : cropped_w[1]
        ]
        self.delta_translation = np.float32([
            cropped_w[0],
            cropped_h[0],
        ])

    def set_template_from_polygon_points(self, template, polygon_points) -> None:
        self._full_template = cv2.imread(template, 0)
        # Create mask from polygon points
        mask = np.zeros(self._full_template.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon_points)], 255)
        self._template = cv2.bitwise_and(self._full_template, self._full_template, mask=mask)

    def set_intrinsics(self, fx, fy, cx, cy) -> None:
        self._fx = fx
        self._fy = fy
        self.cx_cy_array = np.array([cx, cy])


    def set_depth_image(self, depth_image) -> None:
        self._depth_img = depth_image

    def set_image(self, img) -> None:
        self._img = img

    def detect_points(self):
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        if self._logger.level == logging.DEBUG:
            cv2.imshow('template', gray)
            cv2.waitKey(0)
            cv2.imshow('template', self._template)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self._template, None)
        kp2, des2 = sift.detectAndCompute(gray, None)


        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # find matches by knn which calculates point distance in 128 dim
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # translate keypoints back to full source template
        """
        for k in kp1:
            k.pt = (k.pt[0] + self.delta_translation[0], k.pt[1] + self.delta_translation[1])
        """


        if len(good) > MIN_MATCH_COUNT:
            self._src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            self._dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        try:
            M, mask = cv2.findHomography(self._src_pts, self._dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matchesMask,
                flags=2,
            )
            self._annoted_image = cv2.drawMatches(self._full_template, kp1, self._img, kp2, good, None, **draw_params)
            cv2.imwrite('/home/mspahn/temp/annoted_image.png', self._annoted_image)


            if self._logger.level == logging.DEBUG:
                displayed_image = cv2.resize(self._annoted_image, (self._annoted_image.shape[1] // 2, self._annoted_image.shape[0] // 2))
                cv2.imshow('annoted_image', displayed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(e)

    def annoted_image(self):
        return self._annoted_image

    def set_template_tf(self, tf: dict) -> None:
        pose = Pose()
        pose.position.x = float(tf['position']['x'])
        pose.position.y = float(tf['position']['y'])
        pose.position.z = float(tf['position']['z'])
        pose.orientation.x = float(tf['orientation']['x'])
        pose.orientation.y = float(tf['orientation']['y'])
        pose.orientation.z = float(tf['orientation']['z'])
        pose.orientation.w = float(tf['orientation']['w'])
        self._template_tf = pose_2_transformation(pose)

    def get_depth_value_of_feature(self, feature_position_pixel, template=False) -> float:
        # spatial median filter of are of nx x ny pixels
        nx = 10
        ny = 10
        x = round(feature_position_pixel[0])
        y = round(feature_position_pixel[1])
        x_start = x - nx // 2
        x_end = x + nx // 2
        y_start = y - ny // 2
        y_end = y + ny // 2
        if nx == 1:
            x_start = x
            x_end = x+1
        if ny == 1:
            y_start = y
            y_end = y+1

        if template:
            value = np.median(self._depth_template[y_start:y_end, x_start:x_end]) * 0.001
        else:
            value = np.median(self._depth_img[y_start:y_end, x_start:x_end]) * 0.001
        return value

    def save_all_heights(self, panda_link_tf) -> None:
        pixel_values_src = np.array(self._src_pts)[:, 0, :]
        pixel_values_dst = np.array(self._dst_pts)[:, 0, :]
        p_src = np.transpose(pixel_values_src) - self.cx_cy_array
        p_dst = np.transpose(pixel_values_dst) - self.cx_cy_array 
        n_features = pixel_values_src.shape[0]
        nx = 480
        ny = 848
        xyz_src = np.zeros((nx, ny, 3))

        for i in range(nx):
            for j in range(ny):
                xyz_src[i, j, 2] = self.get_depth_value_of_feature([i, j])
                xyz_src[i, j, 0] = (i - self.cx_cy_array[0])/self._fx * xyz_src[i, j, 2]
                xyz_src[i, j, 1] = (j - self.cx_cy_array[1])/self._fy * xyz_src[i, j, 2]

        np.save('/home/mspahn/temp/xyz_src.npy', xyz_src)

    def get_depths(self):

        pixel_values_src = np.array(self._src_pts)[:, 0, :]
        pixel_values_dst = np.array(self._dst_pts)[:, 0, :]
        p_src = np.transpose(pixel_values_src) - self.cx_cy_array
        p_dst = np.transpose(pixel_values_dst) - self.cx_cy_array 
        n_features = pixel_values_src.shape[0]
        z_src = np.zeros(n_features)
        z_dst = np.zeros(n_features)
        for i in range(n_features):
            z_src[i] = self.get_depth_value_of_feature(pixel_values_src[i], template=True)
            z_dst[i] = self.get_depth_value_of_feature(pixel_values_dst[i], template=False)
        return np.median(z_src), np.median(z_dst)


    def compute_tf(self, panda_link_tf) -> np.ndarray:
        version_2d = True
        pixel_values_src = np.array(self._src_pts)[:, 0, :]
        pixel_values_dst = np.array(self._dst_pts)[:, 0, :]
        p_src = np.transpose(pixel_values_src) - self.cx_cy_array
        p_dst = np.transpose(pixel_values_dst) - self.cx_cy_array 
        n_features = pixel_values_src.shape[0]
        z_src = np.zeros(n_features)
        z_dst = np.zeros(n_features)
        for i in range(n_features):
            z_src[i] = self.get_depth_value_of_feature(pixel_values_src[i], template=True)
            z_dst[i] = self.get_depth_value_of_feature(pixel_values_dst[i], template=False)

        if version_2d:
            z_src = np.ones_like(z_src) * np.median(z_src)
            z_dst = np.ones_like(z_dst) * np.median(z_dst)
        xy_src = np.zeros((2, n_features))
        xy_dst = np.zeros((2, n_features))
        xy_src[0, :] = p_src[0, :] / self._fx * z_src
        xy_src[1, :] = p_src[1, :] / self._fy * z_src
        xy_dst[0, :] = p_dst[0, :] / self._fx * z_dst
        xy_dst[1, :] = p_dst[1, :] / self._fy * z_dst
        xyz1_src = np.vstack((xy_src, z_src, np.ones(n_features)))
        xyz1_dst = np.vstack((xy_dst, z_dst, np.ones(n_features)))
        xyz1_dst_panda_link = np.dot(panda_link_tf, xyz1_dst)[:3, :]
        xyz1_src_panda_link = np.dot(self._template_tf, xyz1_src)[:3, :]
        if version_2d:
            xy_src = xyz1_src_panda_link[:2, :]
            xy_dst = xyz1_dst_panda_link[:2, :]
            T0_2d = compute_transform(xy_src, xy_dst)
            T0 = np.eye(4)
            T0[0:2, 0:2] = T0_2d[0:2, 0:2]
            T0[0:2, 3] = T0_2d[0:2, 2]
            print(f'using depth : {T0}')


        else:
            np.save('/home/mspahn/temp/xyz1_src_panda_link.npy', xyz1_src_panda_link)
            np.save('/home/mspahn/temp/xyz1_dst_panda_link.npy', xyz1_dst_panda_link)
            T0, mask = compute_transform_3d(xyz1_src_panda_link, xyz1_dst_panda_link)
            np.save('/home/mspahn/temp/mask.npy', mask)

        return T0

    def compute_full_tf_in_m_new(self, panda_link_tf: np.ndarray) -> np.ndarray:
        print('new')
        return self.compute_tf(panda_link_tf)

    def compute_tf_old(self, panda_link_tf) -> np.ndarray:
        p0 = np.transpose(np.array(self._src_pts))[:, 0, :] - self.cx_cy_array
        p1 = np.transpose(np.array(self._dst_pts))[:, 0, :] - self.cx_cy_array 
        T0 = compute_transform(p0, p1) #this matrix is a 3x3
        return T0

    def compute_full_tf_in_m(self, panda_link_tf: np.ndarray) -> np.ndarray:
        print('old')
        depth_src, depth_dst = self.get_depths()
        depth_difference = depth_src - depth_dst
        T0 = self.compute_tf_old(panda_link_tf)
        self._pixel_m_factor_u =  self._fx / self._box_depth
        self._pixel_m_factor_v =  self._fy / self._box_depth
        T0[0, 2] /= self._pixel_m_factor_u
        T0[1, 2] /= self._pixel_m_factor_v
        T = np.identity(4)
        T[0:2, 0:2] = T0[0:2, 0:2]
        T[0:2, 3] = T0[0:2, 2]
        T[2, 3] = depth_difference
        return T
        T_world = np.dot(np.dot(panda_link_tf, T), np.linalg.inv(panda_link_tf))
        T_world[2, 3] = depth_difference
        return T_world

def compute_transform_3d(points: np.ndarray, transformed_points: np.ndarray):
    assert isinstance(points, np.ndarray)
    assert isinstance(transformed_points, np.ndarray)
    assert points.shape == transformed_points.shape
    affine3d_output = cv2.estimateAffine3D(points.T, transformed_points.T, ransacThreshold=5e-3)
    T = np.identity(4)
    T[0:3, 0:4] = affine3d_output[1]
    mask = affine3d_output[2]
    return T, mask
    
def compute_transform(points: np.ndarray, transformed_points: np.ndarray):
    assert isinstance(points, np.ndarray)
    assert isinstance(transformed_points, np.ndarray)
    assert points.shape == transformed_points.shape
    cv2_matrix, _ = cv2.estimateAffinePartial2D(points.T, transformed_points.T)
    return cv2_matrix



