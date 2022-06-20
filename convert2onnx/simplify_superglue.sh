#!/bin/bash
python -m onnxsim ../output/superglue_indoor.onnx ../output/superglue_indoor_sim.onnx --dynamic-input-shape --input-shape keypoints_0:1,512,2 scores_0:1,512 descriptors_0:1,256,512 keypoints_1:1,512,2 scores_1:1,512 descriptors_1:1,256,512
