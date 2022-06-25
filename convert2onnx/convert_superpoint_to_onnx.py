#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch

import superpoint


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='script to convert superpoint model from pytorch to onnx')
    parser.add_argument('--weight_file', default="weights/superpoint_v1.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="output", help="onnx model file output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    # Load model.
    superpoint_model = superpoint.SuperPoint()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print('total number ff params: ', pytorch_total_params)

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    superpoint_model.load_state_dict(torch.load(weight_file, map_location=map_location))
    superpoint_model.eval()

    # Create input to the model for onnx trace.
    input = torch.randn(1, 1, 480, 752)

    torch_out = superpoint_model(input)
    onnx_filename = os.path.join(output_dir, weight_file.split("/")[-1].split(".")[0] + ".onnx")

    # Export the model
    torch.onnx.export(superpoint_model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['scores', 'descriptors'],  # the model's output names
                      dynamic_axes={'input': {2: 'image_height', 3: "image_width"}}
                      )

    # Check onnx conversion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    onnxruntime_inputs = {onnxruntime_session.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outs = onnxruntime_session.run(None, onnxruntime_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), onnxruntime_outs[0], rtol=1e-03,
                               atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), onnxruntime_outs[1], rtol=1e-03, atol=1e-05)

    print("exported model has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
