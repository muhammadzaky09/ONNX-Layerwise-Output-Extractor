import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_onnx_model(model_path, test_loader):
    """
    Evaluate ONNX model accuracy on MNIST test set
    
    Args:
        model_path (str): Path to the ONNX model
        test_loader: DataLoader for test dataset
    
    Returns:
        float: Accuracy as percentage
    """
    # Initialize ONNX runtime session
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    correct = 0
    total = 0
    
    # Evaluation loop
    for images, labels in test_loader:
        # Convert to numpy and ensure correct format
        batch_images = images.numpy()
        
        # Run inference
        outputs = session.run(None, {input_name: batch_images})
        
        # Get predictions
        predictions = np.argmax(outputs[0], axis=1)
        
        # Calculate accuracy
        correct += np.sum(predictions == labels.numpy())
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    model_path = 'lenet5_test8.onnx'
    batch_size = 1  # Larger batch size for faster evaluation
    
    # Load test data
    test_loader = load_data(batch_size)
    
    # Evaluate model
    try:
        accuracy = evaluate_onnx_model(model_path, test_loader)
        print(f"Test Accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()