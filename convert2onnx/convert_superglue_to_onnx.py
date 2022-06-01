#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import onnx
import onnxruntime
import torch

import superglue


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def reduce_l2(desc):
    dn = np.linalg.norm(desc, ord=2, axis=1)  # Compute the norm.
    desc = desc / np.expand_dims(dn, 1)  # Divide by norm to normalize.
    return desc


def main():
    parser = argparse.ArgumentParser(
        description='script to convert superpoint model from pytorch to onnx')
    parser.add_argument('--weight_file', default="weights/superglue_indoor.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--height', type=int, default=480, help="height in pixels of input image")
    parser.add_argument('--width', type=int, default=848, help="width in pixels of input image")
    parser.add_argument('--output_dir', default="output", help="output directory")
    parser.add_argument('--batch_size', default=1, type=int, help="batch size of input")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file
    batch_size = args.batch_size
    h = args.height
    w = args.width

    # Load model.
    superpoint_model = superglue.SuperGlue()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print('total number of params: ', pytorch_total_params)

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    superpoint_model.load_state_dict(torch.load(weight_file, map_location=map_location))
    superpoint_model.eval()

    # Create input to the model for onnx trace.
    feature_number0 = random.randint(1, 500)
    kpts0 = torch.randn(batch_size, feature_number0, 2, requires_grad=True)
    scores0 = torch.randn(batch_size, feature_number0, requires_grad=True)
    desc0 = torch.randn(batch_size, 256, feature_number0, requires_grad=True)
    feature_number1 = random.randint(1, 500)
    kpts1 = torch.randn(batch_size, feature_number1, 2, requires_grad=True)
    scores1 = torch.randn(batch_size, feature_number1, requires_grad=True)
    desc1 = torch.randn(batch_size, 256, feature_number1, requires_grad=True)
    shape = torch.tensor([float(h), float(w)], requires_grad=True)
    torch_out = superpoint_model(kpts0, scores0, desc0, kpts1, scores1, desc1, shape)
    onnx_filename = os.path.join(output_dir,
                                 weight_file.split("/")[-1].split(".")[0] + "_{}x{}.onnx".format(h,
                                                                                                 w))

    # Export the model
    torch.onnx.export(superpoint_model,  # model being run
                      (kpts0, scores0, desc0, kpts1, scores1, desc1, shape),
                      # model input (or a tuple for multiple inputs)
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,
                      # whether to execute constant folding for optimization
                      input_names=["keypoints_0",  # batch x number1 x 2
                                   "scores_0",  # batch x number1
                                   "descriptors_0",  # batch x 256 x number1
                                   "keypoints_1",  # batch x number1 x 2
                                   "scores_1",  # batch x number1
                                   "descriptors_1",  # batch x 256 x number1
                                   "shape"],  # the model's input names
                      output_names=["scores"],
                      # the model's output names
                      dynamic_axes={'keypoints_0': {1: 'feature_number_0'},
                                    'scores_0': {1: 'feature_number_0'},
                                    'descriptors_0': {2: 'feature_number_0'},
                                    'keypoints_1': {1: 'feature_number_1'},
                                    'scores_1': {1: 'feature_number_1'},
                                    'descriptors_1': {2: 'feature_number_1'},
                                    }
                      )

    # Check onnx converion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    onnxruntime_inputs = {onnxruntime_session.get_inputs()[0].name: to_numpy(kpts0),
                          onnxruntime_session.get_inputs()[1].name: to_numpy(scores0),
                          onnxruntime_session.get_inputs()[2].name: to_numpy(desc0),
                          onnxruntime_session.get_inputs()[3].name: to_numpy(kpts1),
                          onnxruntime_session.get_inputs()[4].name: to_numpy(scores1),
                          onnxruntime_session.get_inputs()[5].name: to_numpy(desc1),
                          onnxruntime_session.get_inputs()[6].name: to_numpy(shape)}
    onnxruntime_outs = onnxruntime_session.run(None, onnxruntime_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), onnxruntime_outs[0][0], rtol=1e-03,
                               atol=1e-05)

    print("exported model has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
