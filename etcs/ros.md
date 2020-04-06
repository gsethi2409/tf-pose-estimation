# tf-pose-estimation for ROS

Human pose estimation is expected to use on mobile robots which need human interactions. 

## Installation

Cloning this repository under src folder in your ros workstation. And the same should be carried out as [README.md](README.md).

```
$ cd $(ros-workspace)
$ cd src
$ git clone https://github.com/ildoonet/tf-pose-estimation
$ pip install -r tf-pose-estimation/requirements.txt
```

There are dependencies to launch demo, 

- video_stream_opencv
- image_view
- ros_video_recorder : https://github.com/ildoonet/ros-video-recorder

## Video/camera demo

| CMU<br/>640x360 | Mobilenet_Thin<br/>432x368 |  
|:----------------|:---------------------------|
| ![cmu-model](/etcs/openpose_p40_cmu.gif) | ![cmu-model](/etcs/openpose_p40_mobilenet.gif) |

Above tests were run on a P40 gpu. Latency between current video frames and processed frames is much lower on mobilenet version.

Source : https://www.youtube.com/watch?v=rSZnyBuc6tc

```
$ roslaunch tfpose_ros demo_video.launch
```

You can specify 'video' arguments to launch realtime video demo using your camera. See [./launch/demo_video.launch](ros launch file). 