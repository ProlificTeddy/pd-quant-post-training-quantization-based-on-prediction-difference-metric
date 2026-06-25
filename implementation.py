import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PDQuant:
    def __init__(self, model, bit_weights=8, bit_activations=8, calibration_data=None):
        self.model = model
        self.bit_weights = bit_weights
        self.bit_activations = bit_activations
        self.calibration_data = calibration_data
        self.device = next(model.parameters()).device

    def quantize_tensor(self, tensor, bit):
        qmin = 0
        qmax = 2 ** bit - 1
        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        zero_point = qmin - tensor.min() / scale
        zero_point = torch.clamp(zero_point, qmin, qmax).round()
        quantized = torch.clamp((tensor / scale + zero_point).round(), qmin, qmax)
        dequantized = (quantized - zero_point) * scale
        return dequantized, scale, zero_point

    def quantize_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                quantized_weight, scale, zero_point = self.quantize_tensor(weight, self.bit_weights)
                module.weight.data = quantized_weight
                module.register_buffer("scale", torch.tensor(scale))
                module.register_buffer("zero_point", torch.tensor(zero_point))

    def calibrate_activations(self):
        self.model.eval()
        with torch.no_grad():
            for data, _ in self.calibration_data:
                data = data.to(self.device)
                self.model(data)

    def compute_prediction_difference(self, original_output, quantized_output):
        return F.mse_loss(original_output, quantized_output)

    def optimize_quantization(self):
        self.model.eval()
        with torch.no_grad():
            for data, _ in self.calibration_data:
                data = data.to(self.device)
                original_output = self.model(data)

                # Quantize weights
                self.quantize_model()

                # Calibrate activations
                self.calibrate_activations()

                # Get quantized model output
                quantized_output = self.model(data)

                # Compute prediction difference
                pd_metric = self.compute_prediction_difference(original_output, quantized_output)
                print(f"Prediction Difference Metric: {pd_metric.item()}")

    def apply(self):
        self.optimize_quantization()

if __name__ == '__main__':
    # Define a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    # Dummy data for testing
    torch.manual_seed(0)
    dummy_data = torch.rand(10, 3, 32, 32)  # Batch of 10 images, 3 channels, 32x32
    dummy_labels = torch.randint(0, 10, (10,))  # Random labels for testing
    dummy_dataset = [(dummy_data, dummy_labels)]

    # Instantiate and test PD-Quant
    model = SimpleModel().to('cpu')
    pd_quant = PDQuant(model, bit_weights=2, bit_activations=2, calibration_data=dummy_dataset)
    pd_quant.apply()