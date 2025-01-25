import onnx

model_path = 'lenet5_test8.onnx'
onnx_model = onnx.load(model_path)

# Print inputs/outputs
print("\nModel Inputs:")
for input in onnx_model.graph.input:
    print(f"  Name: {input.name}, Shape: {input.type.tensor_type.shape}")

print("\nModel Outputs:")
for output in onnx_model.graph.output:
    print(f"  Name: {output.name}, Shape: {output.type.tensor_type.shape}")

# Print nodes (operations)
print("\nNodes (Layers/Operations):")
for node in onnx_model.graph.node:
    print(f"  Name: {node.name}, Op: {node.op_type}")
    print(f"  Inputs: {node.input}")
    print(f"  Outputs: {node.output}\n")