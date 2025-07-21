import torch
import torch.nn as nn
import numpy as np

# Load hex file and convert to tensor
def load_hex_tensor(file, shape):
    with open(file, 'r') as f:
        data = [int(line.strip(), 16) for line in f.readlines()]
    tensor = torch.tensor(data, dtype=torch.float32).reshape(shape)
    return tensor

# Parameters matching Verilog testbench
BATCH_SIZE   = 1
IN_CHANNELS  = 1
IN_HEIGHT    = 4
IN_WIDTH     = 4
OUT_CHANNELS = 1
KERNEL_SIZE  = 3
STRIDE       = 1
PADDING      = 0

# File paths
input_file   = 'input_hex.txt'
weight_file  = 'weights_hex.txt'
bias_file    = 'bias_hex.txt'

# Load tensors
input_tensor  = load_hex_tensor(input_file, (BATCH_SIZE, IN_CHANNELS, IN_HEIGHT, IN_WIDTH))
weights_tensor = load_hex_tensor(weight_file, (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE))
bias_tensor   = load_hex_tensor(bias_file, (OUT_CHANNELS,))

# Define convolution layer manually
conv = nn.Conv2d(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    bias=True
)

# Manually assign weights and bias
with torch.no_grad():
    conv.weight = nn.Parameter(weights_tensor)
    conv.bias = nn.Parameter(bias_tensor)

# Run forward pass
output_tensor = conv(input_tensor)

# Display output
print("✅ PyTorch Convolution Output:")
print(output_tensor)

# Convert to hex and print
output_flat = output_tensor.flatten().tolist()
output_hex = [format(int(v), '08x') for v in output_flat]

print("\n🧾 Output in Hex (for comparison with Verilog):")
for i, val in enumerate(output_hex):
    print(f"Output[{i}] = {val}")
