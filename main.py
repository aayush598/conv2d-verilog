import numpy as np
import torch
import torch.nn as nn
import os

class Conv2dVerilogReady:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Conv2d implementation designed for easy Verilog translation
        
        Args:
            in_channels: Number of input channels (IN_CH parameter in Verilog)
            out_channels: Number of output channels (OUT_CH parameter in Verilog)
            kernel_size: Size of convolution kernel (KERNEL_SIZE parameter)
            stride: Stride of convolution (STRIDE parameter)
            padding: Padding added to input (PADDING parameter)
        """
        # Store parameters (these become Verilog parameters)
        self.IN_CHANNELS = in_channels
        self.OUT_CHANNELS = out_channels
        self.KERNEL_SIZE = kernel_size
        self.STRIDE = stride
        self.PADDING = padding
        
        # Initialize weights and bias (will be loaded from file)
        self.weights = None
        self.bias = None
        
        # Debug counters (useful for Verilog debugging)
        self.mac_operations = 0
        self.memory_accesses = 0
    
    def load_weights_from_file(self, weights_file, bias_file):
        """
        Load weights and bias from text files
        
        Args:
            weights_file: Text file containing weights (one per line)
            bias_file: Text file containing bias values (one per line)
        """
        
        # Load weights
        with open(weights_file, 'r') as f:
            weight_values = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    weight_values.append(float(line))
        
        # Reshape weights to [OUT_CH, IN_CH, KERNEL_H, KERNEL_W]
        expected_weight_count = self.OUT_CHANNELS * self.IN_CHANNELS * self.KERNEL_SIZE * self.KERNEL_SIZE
        if len(weight_values) != expected_weight_count:
            raise ValueError(f"Expected {expected_weight_count} weights, got {len(weight_values)}")
        
        self.weights = np.array(weight_values).reshape(
            self.OUT_CHANNELS, self.IN_CHANNELS, self.KERNEL_SIZE, self.KERNEL_SIZE
        )
        
        # Load bias
        with open(bias_file, 'r') as f:
            bias_values = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    bias_values.append(float(line))
        
        if len(bias_values) != self.OUT_CHANNELS:
            raise ValueError(f"Expected {self.OUT_CHANNELS} bias values, got {len(bias_values)}")
        
        self.bias = np.array(bias_values)
        
    def calculate_output_dimensions(self, input_height, input_width):
        """
        Calculate output dimensions - this logic translates directly to Verilog parameters
        
        In Verilog, these would be calculated as:
        localparam OUT_HEIGHT = (IN_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
        localparam OUT_WIDTH = (IN_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
        """
        padded_height = input_height + 2 * self.PADDING
        padded_width = input_width + 2 * self.PADDING
        
        output_height = (padded_height - self.KERNEL_SIZE) // self.STRIDE + 1
        output_width = (padded_width - self.KERNEL_SIZE) // self.STRIDE + 1
        
        return output_height, output_width
    
    def check_padding_bounds(self, coord, max_coord):
        """
        Padding boundary check - direct translation to Verilog conditional
        
        In Verilog:
        wire valid_coord = (coord >= 0) && (coord < MAX_COORD);
        wire input_data = valid_coord ? memory_data : 8'b0; // Zero padding
        """
        return 0 <= coord < max_coord
    
    def get_input_coordinate(self, output_coord, kernel_coord):
        """
        Calculate input coordinate from output coordinate and kernel position
        
        In Verilog:
        wire [ADDR_WIDTH-1:0] input_coord = output_coord * STRIDE + kernel_coord - PADDING;
        """
        return output_coord * self.STRIDE + kernel_coord - self.PADDING
    
    def forward_verilog_style(self, input_tensor):
        """
        Forward pass with explicit nested loops matching Verilog implementation
        Each loop level corresponds to a Verilog always block or generate loop
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be loaded before forward pass")
        
        # Input dimensions
        BATCH_SIZE, IN_CH, IN_HEIGHT, IN_WIDTH = input_tensor.shape
        
        # Calculate output dimensions
        OUT_HEIGHT, OUT_WIDTH = self.calculate_output_dimensions(IN_HEIGHT, IN_WIDTH)
        
        # Initialize output tensor (equivalent to output BRAM initialization)
        output_tensor = np.zeros((BATCH_SIZE, self.OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH))
        
        # Reset counters
        self.mac_operations = 0
        self.memory_accesses = 0
        
        # VERILOG TRANSLATION: These nested loops become nested generate statements
        # or sequential always blocks depending on implementation choice
        
        # Batch loop (in Verilog: often handled by external controller)
        for batch_idx in range(BATCH_SIZE):
            
            # Output channel loop (VERILOG: generate loop or counter)
            for out_ch in range(self.OUT_CHANNELS):
                
                # Output spatial loops (VERILOG: nested counters)
                for out_h in range(OUT_HEIGHT):
                    for out_w in range(OUT_WIDTH):
                        
                        # Initialize accumulator with bias (VERILOG: MAC unit initialization)
                        accumulator = self.bias[out_ch]
                        self.memory_accesses += 1  # Bias memory read
                        
                        # Input channel loop (VERILOG: inner processing loop)
                        for in_ch in range(self.IN_CHANNELS):
                            
                            # Kernel loops (VERILOG: convolution window processing)
                            for kernel_h in range(self.KERNEL_SIZE):
                                for kernel_w in range(self.KERNEL_SIZE):
                                    
                                    # Calculate input coordinates (VERILOG: address calculation)
                                    input_h = self.get_input_coordinate(out_h, kernel_h)
                                    input_w = self.get_input_coordinate(out_w, kernel_w)
                                    
                                    # Padding check (VERILOG: conditional data selection)
                                    if (self.check_padding_bounds(input_h, IN_HEIGHT) and 
                                        self.check_padding_bounds(input_w, IN_WIDTH)):
                                        # Valid input coordinate - read from memory
                                        input_value = input_tensor[batch_idx, in_ch, input_h, input_w]
                                        self.memory_accesses += 1
                                    else:
                                        # Padding region - use zero
                                        input_value = 0.0
                                    
                                    # Weight memory access (VERILOG: weight BRAM read)
                                    weight_value = self.weights[out_ch, in_ch, kernel_h, kernel_w]
                                    self.memory_accesses += 1
                                    
                                    # MAC operation (VERILOG: DSP48 or custom MAC unit)
                                    accumulator += input_value * weight_value
                                    self.mac_operations += 1
                        
                        # Store final result (VERILOG: output BRAM write)
                        output_tensor[batch_idx, out_ch, out_h, out_w] = accumulator
                        self.memory_accesses += 1
        
        return output_tensor
    
    def get_verilog_parameters(self):
        """Return parameters in Verilog format"""
        return {
            'IN_CHANNELS': self.IN_CHANNELS,
            'OUT_CHANNELS': self.OUT_CHANNELS,
            'KERNEL_SIZE': self.KERNEL_SIZE,
            'STRIDE': self.STRIDE,
            'PADDING': self.PADDING
        }
    
    def get_resource_utilization(self):
        """Estimate FPGA resource utilization"""
        weight_memory = self.OUT_CHANNELS * self.IN_CHANNELS * self.KERNEL_SIZE * self.KERNEL_SIZE
        bias_memory = self.OUT_CHANNELS
        
        return {
            'mac_operations': self.mac_operations,
            'memory_accesses': self.memory_accesses,
            'weight_memory_words': weight_memory,
            'bias_memory_words': bias_memory,
            'total_parameters': weight_memory + bias_memory
        }

def load_input_from_file(input_file, batch_size, in_channels, height, width):
    """
    Load input tensor from text file
    
    Args:
        input_file: Text file containing input values (one per line)
        batch_size: Batch size
        in_channels: Number of input channels
        height: Input height
        width: Input width
    """
    print(f"Loading input data from: {input_file}")
    
    with open(input_file, 'r') as f:
        input_values = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                input_values.append(float(line))
    
    expected_count = batch_size * in_channels * height * width
    if len(input_values) != expected_count:
        raise ValueError(f"Expected {expected_count} input values, got {len(input_values)}")
    
    input_tensor = np.array(input_values).reshape(batch_size, in_channels, height, width)
    return input_tensor

def create_sample_data_files():
    """Create sample input data files for testing"""
    
    # Test parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 4
    KERNEL_SIZE = 3
    BATCH_SIZE = 1
    IN_HEIGHT, IN_WIDTH = 8, 8
    
    print("Creating sample data files...")
    
    # Create weights file
    np.random.seed(42)
    weight_count = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
    weights = np.random.randn(weight_count) * 0.1
    
    with open('weights.txt', 'w') as f:
        f.write("# Conv2D Weights - one per line\n")
        f.write(f"# Shape: [{OUT_CHANNELS}, {IN_CHANNELS}, {KERNEL_SIZE}, {KERNEL_SIZE}]\n")
        for weight in weights:
            f.write(f"{weight:.6f}\n")
    
    # Create bias file
    bias = np.random.randn(OUT_CHANNELS) * 0.01
    
    with open('bias.txt', 'w') as f:
        f.write("# Conv2D Bias - one per line\n")
        f.write(f"# Shape: [{OUT_CHANNELS}]\n")
        for b in bias:
            f.write(f"{b:.6f}\n")
    
    # Create input file
    input_count = BATCH_SIZE * IN_CHANNELS * IN_HEIGHT * IN_WIDTH
    input_data = np.random.randn(input_count) * 0.5
    
    with open('input.txt', 'w') as f:
        f.write("# Input data - one per line\n")
        f.write(f"# Shape: [{BATCH_SIZE}, {IN_CHANNELS}, {IN_HEIGHT}, {IN_WIDTH}]\n")
        for val in input_data:
            f.write(f"{val:.6f}\n")
    
    return IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, BATCH_SIZE, IN_HEIGHT, IN_WIDTH

def print_tensor_summary(tensor, name, max_elements=10):
    """Print a summary of tensor values"""
    print(f"\n{name} Summary:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Min: {np.min(tensor):.6f}")
    print(f"  Max: {np.max(tensor):.6f}")
    print(f"  Mean: {np.mean(tensor):.6f}")
    print(f"  Std: {np.std(tensor):.6f}")
    
    # Print first few elements
    flat_tensor = tensor.flatten()
    print(f"  First {min(max_elements, len(flat_tensor))} elements:")
    for i in range(min(max_elements, len(flat_tensor))):
        print(f"    [{i}]: {flat_tensor[i]:.6f}")

def test_file_based_implementation():
    """Test the Verilog-ready implementation with file-based inputs"""
    
    # Check if data files exist, create them if they don't
    if not (os.path.exists('weights.txt') and os.path.exists('bias.txt') and os.path.exists('input.txt')):
        print("Data files not found. Creating sample files...")
        IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, BATCH_SIZE, IN_HEIGHT, IN_WIDTH = create_sample_data_files()
    else:
        # If using existing files, you'll need to specify these parameters
        IN_CHANNELS = 3
        OUT_CHANNELS = 4
        KERNEL_SIZE = 3
        BATCH_SIZE = 1
        IN_HEIGHT, IN_WIDTH = 8, 8
    
    STRIDE = 1
    PADDING = 1

    
    # Load input tensor from file
    input_tensor = load_input_from_file('input.txt', BATCH_SIZE, IN_CHANNELS, IN_HEIGHT, IN_WIDTH)
    
    # Create and configure Verilog implementation
    verilog_conv = Conv2dVerilogReady(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING)
    verilog_conv.load_weights_from_file('weights.txt', 'bias.txt')
    
    # Create PyTorch implementation with same weights
    pytorch_conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, padding=PADDING)
    pytorch_conv.weight.data = torch.tensor(verilog_conv.weights, dtype=torch.float32)
    pytorch_conv.bias.data = torch.tensor(verilog_conv.bias, dtype=torch.float32)
    
    print(f"\nConfiguration: Conv2d({IN_CHANNELS}, {OUT_CHANNELS}, {KERNEL_SIZE}, {STRIDE}, padding={PADDING})")
    print(f"Input shape: {input_tensor.shape}")
    
    # Run forward passes
    print("\nRunning forward passes...")
    
    # Verilog-style implementation
    verilog_output = verilog_conv.forward_verilog_style(input_tensor)
    
    # PyTorch implementation
    with torch.no_grad():
        pytorch_output = pytorch_conv(torch.tensor(input_tensor, dtype=torch.float32))
        pytorch_output_np = pytorch_output.numpy()
    
    print(f"Output shape: {verilog_output.shape}")
    
    # Compare results
    max_diff = np.max(np.abs(verilog_output - pytorch_output_np))
    matches = np.allclose(verilog_output, pytorch_output_np, rtol=1e-5, atol=1e-6)
    
    
    # Print side-by-side comparison for first few outputs
    print("First 10 output values comparison:")
    scratch_flat = verilog_output.flatten()
    pytorch_flat = pytorch_output_np.flatten()
    
    print(f"{'Index':<6} {'Scratch':<12} {'PyTorch':<12} {'Difference':<12}")
    print("-" * 50)
    for i in range(min(10, len(scratch_flat))):
        diff = scratch_flat[i] - pytorch_flat[i]
        print(f"{i:<6} {scratch_flat[i]:<12.6f} {pytorch_flat[i]:<12.6f} {diff:<12.2e}")
    
    return verilog_conv

if __name__ == "__main__":
    # Run the test
    conv_layer = test_file_based_implementation()