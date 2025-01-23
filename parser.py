import onnx
import json
import os
from typing import Dict

def extract_model_info(model_path: str, output_dir: str):
    model = onnx.load(model_path)
    graph = model.graph
    
    os.makedirs(output_dir, exist_ok=True)
    
    conv_count = 0
    for node in graph.node:
        if node.op_type == 'Conv' or node.op_type == 'Gemm':
            conv_count += 1
            info = {
                "target_layer": node.name,
                "input_tensor": node.input[0],
                "weight_tensor": node.input[1],
                "bias_tensor": node.input[2] if len(node.input) > 2 else None,
                "output_tensor": node.output[0],
                "target_tensor": None,  # We'll update this later
                "model_name": model_path
            }
            
            # Find the target tensor (next QuantizeLinear or Clip node)
            for next_node in graph.node:
                if next_node.input[0] == info["output_tensor"]:
                    if next_node.op_type in ['QuantizeLinear', 'Clip']:
                        info["target_tensor"] = next_node.output[0]
                        break
            
            # If no QuantizeLinear or Clip found, look for any next node
            if info["target_tensor"] is None:
                for next_node in graph.node:
                    if info["output_tensor"] in next_node.input:
                        info["target_tensor"] = next_node.output[0]
                        break
            
            save_to_json(info, os.path.join(output_dir, f"input{conv_count}.json"))
    
    print(f"Created {conv_count} JSON files in {output_dir}")

def save_to_json(data: Dict, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    model_path = "lenet5_test8.onnx"
    output_dir = "./input/lenet5-8"
    
    extract_model_info(model_path, output_dir)