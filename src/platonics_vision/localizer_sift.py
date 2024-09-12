import numpy as np
import cv2
import logging
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

    def set_template(self, template) -> None:
        assert isinstance(template, str)
        print(template)
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

    def set_intrinsics(self, fx, fy, cx, cy) -> None:
        self._fx = fx
        self._fy = fy
        self.cx_cy_array = np.array([cx, cy])


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
        for k in kp1:
            k.pt = (k.pt[0] + self.delta_translation[0], k.pt[1] + self.delta_translation[1])


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


            if self._logger.level == logging.DEBUG:
                displayed_image = cv2.resize(self._annoted_image, (self._annoted_image.shape[1] // 2, self._annoted_image.shape[0] // 2))
                cv2.imshow('annoted_image', displayed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(e)

    def annoted_image(self):
        return self._annoted_image

    def compute_tf(self) -> np.ndarray:

        p0 = np.transpose(np.array(self._src_pts))[:, 0, :] - self.cx_cy_array
        p1 = np.transpose(np.array(self._dst_pts))[:, 0, :] - self.cx_cy_array 
        T0 = compute_transform(p0, p1) #this matrix is a 3x3

        return T0

    def compute_full_tf_in_m(self) -> np.ndarray:
        T0 = self.compute_tf()
        self._pixel_m_factor_u =  self._fx / self._box_depth
        self._pixel_m_factor_v =  self._fy / self._box_depth
        T0[0, 2] /= self._pixel_m_factor_u
        T0[1, 2] /= self._pixel_m_factor_v
        T = np.identity(4)
        T[0:2, 0:2] = T0[0:2, 0:2]
        T[0:2, 3] = T0[0:2, 2]
        return T
    
def compute_transform(points: np.ndarray, transformed_points: np.ndarray):
    assert isinstance(points, np.ndarray)
    assert isinstance(transformed_points, np.ndarray)
    assert points.shape == transformed_points.shape
    cv2_matrix, _ = cv2.estimateAffinePartial2D(points.T, transformed_points.T)
    return cv2_matrix


