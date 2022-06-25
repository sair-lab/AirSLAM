import onnx
from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto
from onnx import numpy_helper as nph
import numpy as np
from collections import OrderedDict


def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def convert_params_to_int32(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        if data.data_type == TensorProto.INT64:
            data_cvt = nph.to_array(data).astype(np.int32)
            data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params


def convert_constant_nodes_to_int32(nodes):
    new_nodes = []
    for node in nodes:
        if (
                node.op_type == "Constant"
                and node.attribute[0].t.data_type == TensorProto.INT64
        ):
            data = nph.to_array(node.attribute[0].t).astype(np.int32)
            new_t = nph.from_array(data)
            new_node = h.make_node(
                "Constant",
                inputs=[],
                outputs=node.output,
                name=node.name,
                value=new_t,
            )
            new_nodes += [new_node]
        else:
            new_nodes += [node]

    return new_nodes


def convert_model_to_int32(model_path: str, out_path: str):
    print("ONNX INT64 --> INT32 Converter")
    print("Loading Model: " + model_path)
    # load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph
    # The initializer holds all non-constant weights.
    init = graph.initializer
    # collect model params in a dictionary.
    params_dict = make_param_dictionary(init)
    print("Converting INT64 model params to INT32...")
    # convert all INT64 params to INT32.
    converted_params = convert_params_to_int32(params_dict)
    print("Converting constant INT64 nodes to INT32...")
    new_nodes = convert_constant_nodes_to_int32(graph.node)

    graph_name = f"{graph.name}-int32"
    print("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_int32 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    print("Creating new int32 model...")
    model_int32 = h.make_model(graph_int32, producer_name="onnx-typecast")
    model_int32.opset_import[0].version = opset_version
    ch.check_model(model_int32)
    print(f"Saving converted model as: {out_path}")
    onnx.save_model(model_int32, out_path)
    print(f"Done.")
    return


if __name__ == "__main__":
    convert_model_to_int32(out_path="output/superglue_outdoor_int32.onnx",
                           model_path="output/superglue_outdoor.onnx")
