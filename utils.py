import onnx
from typing import List
import onnxruntime as ort

def get_tensor_shape(model: onnx.ModelProto, tensor_name: str) -> List[int]:
    # Check all possible tensor locations
    for tensor in (list(model.graph.input) + 
                  list(model.graph.output) + 
                  list(model.graph.value_info)):
        if tensor.name == tensor_name:
            try:
                shape = [dim.dim_value for dim in 
                        tensor.type.tensor_type.shape.dim]
                if all(isinstance(d, int) for d in shape):
                    return shape
            except AttributeError:
                continue
                
    raise ValueError(f"Could not find valid shape for tensor: {tensor_name}")

def run_onnx_model(model_path, input_data):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_dict = {input_name: input_data}
    output = session.run(None, input_dict)
    output_names = [output.name for output in session.get_outputs()]
    return output, output_names