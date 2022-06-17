#### 测试
运行前修改main.cpp文件中参数.

#### 主要依赖:
* OpenCV
* Eigen
* TensorRT 8.4
* CUDA 11.6

#### 环境配置
https://github.com/NVIDIA/TensorRT

https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html

#### Docker
docker pull yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --privileged --network host --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name air_slam yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest /bin/bash
