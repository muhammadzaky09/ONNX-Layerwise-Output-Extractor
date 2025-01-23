import onnx
import os
import json
from onnx import helper
from utils import get_tensor_shape, run_onnx_model
import numpy as np

input_path = 'lenet5_test8.onnx'
model = onnx.load(input_path)
directory_list = os.listdir('input/lenet5-8')

for layer in directory_list:
    input_inject_data = json.load(open('input/lenet5-8' + "/" + layer))
    shape = get_tensor_shape(model, input_inject_data['output_tensor'])
    intermediate_layer_value_info = helper.make_tensor_value_info(
        name=input_inject_data['output_tensor'],
        elem_type=onnx.TensorProto.FLOAT,
        shape=shape
    )
    model.graph.output.append(intermediate_layer_value_info)
    
onnx.save(model, "lenet5-8_modified.onnx")

sample_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
output= run_onnx_model("lenet5-8_modified.onnx", sample_input)
print(output)

   