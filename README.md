# AirVO
## An Illumination-Robust Point-Line Visual Odometry

 <img src="images/pipeline.jpg" width = "800" alt="pipeline" />

AirVO is an **illumination-robust** and accurate **stereo visual odometry system** based on **point and line features**. To be robust to illumination variation, we introduce the **learning-based feature extraction ([SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)) and matching ([SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)) method** and design a novel VO pipeline, including feature tracking, triangulation, key-frame selection, and graph optimization etc. We also employ long line features in the environment to improve the accuracy of the system. Different from the traditional line processing pipelines in visual odometry systems, we propose **an illumination-robust line tracking method**, where point feature tracking and distribution of point and line features are utilized to match lines. By accelerating the feature extraction and matching network using Nvidia TensorRT Toolkit, AirVO can run in **real time** on GPU.

**Authors:** [Kuan Xu](https://github.com/xukuanHIT), [Yuefan Hao](https://github.com/yuefanhao), [Chen Wang](https://chenwang.site/), and [Lihua Xie](https://personal.ntu.edu.sg/elhxie/)


## Demos

### UMA-VI dataset
[UMA-VI dataset](http://mapir.isa.uma.es/mapirwebsite/?p=2108) contains many sequences where images may suddenly darken as a result of turning off the lights. Here are demos on two sequences.

<img src="images/demo_uma.gif" height = "200" alt="uma" border="10" /><img src="images/uma_traj.png" height = "220" alt="uma_traj" align="top" border="10" />

### OIVIO dataset
[OIVIO dataset](https://arpg.github.io/oivio/) collects data in mines and tunnels with onboard illumination.

<img src="images/demo_oivio.gif" height = "200" alt="oivio" /> <img src="images/oivio_traj.png" height = "100" alt="oivio_traj" />


### Live demo with realsense camera
We also test AirVO on sequences collected by Realsense D435I in the environment with continuous changing illumination. 

<img src="images/demo_realsense.gif" width = "539" height = "211" alt="realsense" />

### More
[Video demo](https://www.youtube.com/watch?v=ZBggy5syysY)


## Test Environment
### Dependencies
* OpenCV 4.2
* Eigen 3
* G2O
* TensorRT 8.4
* CUDA 11.6
* python
* onnx
* ROS noetic
* Boost
* Glog


### Docker (Recommend)
```bash
docker pull xukuanhit/air_slam:v1
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name air_slam xukuanhit/air_slam:v1 /bin/bash
```

## Data
The data should be organized using the Automous Systems Lab (ASL) dataset format just like the following:

```
dataroot
├── cam0
│   └── data
│       ├── 00001.jpg
│       ├── 00002.jpg
│       ├── 00003.jpg
│       └── ......
└── cam1
    └── data
        ├── 00001.jpg
        ├── 00002.jpg
        ├── 00003.jpg
        └── ......
```

## Build
```
    cd ~/catkin_ws/src
    git clone https://github.com/xukuanHIT/AirVO.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## Run 

### [OIVIO Dataset](https://arpg.github.io/oivio/)
```
roslaunch air_vo oivio.launch 
```

### [UMA-VI Dataset](http://mapir.isa.uma.es/mapirwebsite/?p=2108)
```
roslaunch air_vo uma_bumblebee_indoor.launch 
```

### [Euroc Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
```
roslaunch air_vo euroc.launch 
```

## Acknowledgements
We would like to thank [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) for making their project public.
