# Demo
[Video demo](https://www.youtube.com/watch?v=ZBggy5syysY)

# Dependencies:
* OpenCV
* Eigen
* G2O
* TensorRT 8.4
* CUDA 11.6
* python
* onnx
* ROS noetic
* Boost
* Glog
* Gflag


# Docker     
```bash
docker push xukuanhit/air_slam:v1
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --privileged --network host --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name air_slam xukuanhit/air_slam:v1 /bin/bash
```

# Data
Euroc style stereo data.


# Build
Build the code as a ros package.


# Run 

## OIVIO Dataset
```
roslaunch air_vo oivio.launch 
```

## UMA-VI Dataset
```
roslaunch air_vo uma_bumblebee_indoor.launch 
```

## Euroc Dataset
```
roslaunch air_vo euroc.launch 
```