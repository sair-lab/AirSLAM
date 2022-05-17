#### 测试
运行前修改main.cpp文件中参数.

#### 主要依赖:
* OpenCV
* Eigen
* TensorRT 8
* CUDA 11.3

#### 环境配置
https://github.com/NVIDIA/TensorRT

https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html

#### Docker 
```bash
docker run -it --gpus all -p LOCAL_PORT:CONTAINER_PORT tensorrt8/ubuntu18.04-cuda11.3:v0.3 /bin/bash
```
