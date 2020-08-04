#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: onnx2trt.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.03.31 18:55    shengyang      v0.1        creation

import onnx
import tensorrt as trt
from onnx import TensorProto


def modify_onnx(onnxfile):
    h = 480
    w = 800
    onnx_model = onnx.load(onnxfile)
    graph = onnx_model.graph

    input0 = graph.input[0]
    new_input0 = onnx.helper.make_tensor_value_info("input.1", TensorProto.FLOAT, (1, 3, h, w))
    graph.input.remove(input0)
    graph.input.insert(0, new_input0)

    output0 = graph.output[0]
    new_output0 = onnx.helper.make_tensor_value_info("537", TensorProto.FLOAT, (1, 1, h//4, w//4))
    graph.output.remove(output0)
    graph.output.insert(0, new_output0)

    output1 = graph.output[1]
    new_output1 = onnx.helper.make_tensor_value_info("538", TensorProto.FLOAT, (1, 2, h//4, w//4))
    graph.output.remove(output1)
    graph.output.insert(1, new_output1)

    output2 = graph.output[2]
    new_output2 = onnx.helper.make_tensor_value_info("539", TensorProto.FLOAT, (1, 2, h//4, w//4))
    graph.output.remove(output2)
    graph.output.insert(2, new_output2)

    output3 = graph.output[3]
    new_output3 = onnx.helper.make_tensor_value_info("540", TensorProto.FLOAT, (1, 10, h//4, w//4))
    graph.output.remove(output3)
    graph.output.insert(3, new_output3)

    # onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "../models/modify.onnx")


def convert_trt(onnx_file_path="./model/centerface_800_480.onnx", output_trt="./model/centerface_800_480.onnx", batchsize=32):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    # get network object via builder
    network = builder.create_network()

    # create ONNX parser object
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as onnx_fin:
        parser.parse(onnx_fin.read())

    # print possible errors
    num_error = parser.num_errors
    if num_error != 0:
        for i in range(num_error):
            temp_error = parser.get_error(i)
            print(temp_error.desc())

    # create engine via builder
    builder.max_batch_size = batchsize
    builder.average_find_iterations = 2
    builder.max_workspace_size = 1 << 30   # 1G
    builder.fp16_mode = False

    engine = builder.build_cuda_engine(network)

    with open(output_trt, 'wb') as fout:
        fout.write(engine.serialize())


if __name__ == "__main__":
    # modify_onnx("./model/faceclassification.onnx")
    convert_trt("/home/zhex/pre_models/AILuoGang/faceclassification.onnx", "/home/zhex/pre_models/AILuoGang/faceclassification.trt")
