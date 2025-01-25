import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import IntBias
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int32Bias
from brevitas.quant import Int8WeightPerTensorFloat, Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int32Bias
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import numpy as np
import time
import warnings
import os
from brevitas.export import export_onnx_qcdq
import qonnx.util.cleanup

# 8-bit weight quantization configurations
class Common8bitWeightPerTensorQuant(Int8WeightPerTensorFloat):
    scaling_min_val = 2e-16

class Common8bitWeightPerChannelQuant(Int8WeightPerChannelFloat):
    scaling_per_output_channel = True
    scaling_min_val = 2e-16

# 8-bit activation quantization configurations
class Common8bitActQuant(Int8ActPerTensorFloat):
    scaling_min_val = 2e-16
    restrict_scaling_type = RestrictValueType.LOG_FP

class Common8bitUintActQuant(Uint8ActPerTensorFloat):
    scaling_min_val = 2e-16
    restrict_scaling_type = RestrictValueType.LOG_FP
model_name = 'lenet5_quantized8.pth'
class QuantizedLeNet5_8bit(nn.Module):
    def __init__(self):
        super(QuantizedLeNet5_8bit, self).__init__()
        
        # First convolutional layer (8-bit)
        self.conv1 = QuantConv2d(
            1, 6, kernel_size=5, stride=1, padding=2,
            weight_bit_width=8,
            bias=True,
            weight_quant=Common8bitWeightPerChannelQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu1 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # First average pooling layer
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer (8-bit)
        self.conv2 = QuantConv2d(
            6, 16, kernel_size=5, stride=1, padding=0,
            weight_bit_width=8,
            bias=True,
            weight_quant=Common8bitWeightPerChannelQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu2 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # Second average pooling layer
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # First fully connected layer (8-bit)
        self.fc1 = QuantLinear(
            16 * 5 * 5, 120,
            bias=True,
            weight_bit_width=8,
            weight_quant=Common8bitWeightPerTensorQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu3 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # Second fully connected layer (8-bit)
        self.fc2 = QuantLinear(
            120, 84,
            bias=True,
            weight_bit_width=8,
            weight_quant=Common8bitWeightPerTensorQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)
        self.relu4 = QuantReLU(
            bit_width=8,
            act_quant=Common8bitUintActQuant,
            return_quant_tensor=True)
        
        # Output layer (8-bit)
        self.fc3 = QuantLinear(
            84, 10,
            bias=True,
            weight_bit_width=8,
            weight_quant=Common8bitWeightPerTensorQuant,
            input_quant=Common8bitActQuant,
            output_quant=Common8bitActQuant,
            return_quant_tensor=True)

    def forward(self, x):
     
        x = self.relu1(self.conv1(x))
        x = self.avg_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.avg_pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create model instance and move to device
    model = QuantizedLeNet5_8bit()
    model = model.to(device)
    
    
    train_loader, test_loader = load_data()
    # Set up optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(10):
        train(model, device, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, device, test_loader, criterion)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_name)
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # Load the best model for ONNX export
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()
    
    model_filename = "lenet5_test8.onnx"
    export_onnx_qcdq(model, input_shape=([1,1,28,28]), export_path=(model_filename), opset_version=13)
    model = qonnx.util.cleanup.cleanup(in_file=model_filename, out_file=model_filename)
    
    return model, best_accuracy

if __name__ == '__main__':
    main()