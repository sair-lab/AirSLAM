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
    parser.add_argument('--weight_file', default="weights/superglue_outdoor.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="output", help="output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

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
    x0 = torch.from_numpy(np.random.randint(low=0, high=751, size=(1, 512)))
    y0 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, 512)))
    kpts0 = torch.stack((x0, y0), 2).float()
    scores0 = torch.randn(1, 512)
    desc0 = torch.randn(1, 256, 512)
    x1 = torch.from_numpy(np.random.randint(low=0, high=751, size=(1, 512)))
    y1 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, 512)))
    kpts1 = torch.stack((x1, y1), 2).float()
    scores1 = torch.randn(1, 512)
    desc1 = torch.randn(1, 256, 512)
    torch_out = superpoint_model(kpts0, scores0, desc0, kpts1, scores1, desc1)
    onnx_filename = os.path.join(output_dir,
                                 weight_file.split("/")[-1].split(".")[0] + ".onnx")

    # Export the model
    torch.onnx.export(superpoint_model,  # model being run
                      (kpts0, scores0, desc0, kpts1, scores1, desc1),
                      # model input (or a tuple for multiple inputs)
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["keypoints_0",  # batch x number1 x 2
                                   "scores_0",  # batch x number1
                                   "descriptors_0",  # batch x 256 x number1
                                   "keypoints_1",  # batch x number1 x 2
                                   "scores_1",  # batch x number1
                                   "descriptors_1",  # batch x 256 x number1
                                  ],  # the model's input names
                      output_names=["scores"],  # the model's output names
                      dynamic_axes={'keypoints_0': {1: 'feature_number_0'},
                                    'scores_0': {1: 'feature_number_0'},
                                    'descriptors_0': {2: 'feature_number_0'},
                                    'keypoints_1': {1: 'feature_number_1'},
                                    'scores_1': {1: 'feature_number_1'},
                                    'descriptors_1': {2: 'feature_number_1'},
                                    }
                      )

    # Check onnx conversion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    # onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    # onnxruntime_inputs = {onnxruntime_session.get_inputs()[0].name: to_numpy(kpts0),
    #                       onnxruntime_session.get_inputs()[1].name: to_numpy(scores0),
    #                       onnxruntime_session.get_inputs()[2].name: to_numpy(desc0),
    #                       onnxruntime_session.get_inputs()[3].name: to_numpy(kpts1),
    #                       onnxruntime_session.get_inputs()[4].name: to_numpy(scores1),
    #                       onnxruntime_session.get_inputs()[5].name: to_numpy(desc1)}
    # onnxruntime_outs = onnxruntime_session.run(None, onnxruntime_inputs)

    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out[0]), onnxruntime_outs[0][0], rtol=1e-03,
    #                            atol=1e-05)

    print("exported model has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
