#!/bin/bash
python -m onnxsim ../output/superpoint_v1.onnx ../output/superpoint_v1_sim.onnx --dynamic-input-shape --input-shape input:1,1,480,752
