# Platonics Vision

## Installation

```bash
pip install -r requirements.txt
cd <workspace-root>/src
git clone --branch ros1-legacy https://github.com/IntelRealSense/realsense-ros.git
cd <workspace-root>
catkin build
```

## Usage

```bash
roslaunch platonics_vision depth_camera.launch rviz:=False
rosrun platonics_vision sift_service
rosrun platonics_vision iterative_sift_service
rosrun platonics_vision slider_solver_action_server
```


