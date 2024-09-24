import rospy
import rospkg
from triangle_detector import TriangleDetector
import dynamic_reconfigure.client

class SliderDetector():
    def __init__(self):
        rospy.init_node('triangle_detector_node')
        self.control_rate = 30
        self.rate=rospy.Rate(self.control_rate)
        self._triangle_detector = TriangleDetector()

        package_path = rospkg.RosPack().get_path('platonics_vision')
        self.image_dir_path = package_path + '/data/triangle_templates'

        rospy.sleep(1)
        object_ids = ['red', 'green', 'yellow']
        self._triangle_detector.load_template_images(self.image_dir_path, object_ids, debug=True)
        self._exposure_client = dynamic_reconfigure.client.Client("/camera/stereo_module", timeout=10)
        self._detection_params = {'exposure': 7000, 'enable_auto_exposure': False}
        self._running_params = {'enable_auto_exposure': True}

    def run(self):
        self._exposure_client.update_configuration(self._detection_params)
        while not rospy.is_shutdown():
            self.triangles_distance, errorflag = self._triangle_detector.detect_triangles(debug=False)
            if errorflag == 0:
                print("Distance: ", self.triangles_distance)
            self.rate.sleep()

if __name__ == '__main__':
    node = SliderDetector()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
