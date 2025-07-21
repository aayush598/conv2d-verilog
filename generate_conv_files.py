# generate_conv_files.py

def to_hex(value, width=32):
    """Convert integer to fixed width hex string (e.g., 32-bit => 8 hex digits)."""
    return format(value & (2**width - 1), '08x')

def write_file(filename, count, value):
    """Write 'count' lines of 'value' (converted to hex) to the given file."""
    with open(filename, 'w') as f:
        hex_val = to_hex(value)
        for _ in range(count):
            f.write(hex_val + '\n')

# Parameters based on your convolution layer
BATCH_SIZE = 1
IN_CHANNELS = 8
OUT_CHANNELS = 32
IN_HEIGHT = 7
IN_WIDTH = 7
KERNEL_SIZE = 7

# Input: 1x8x7x7 = 392 values (all 1s)
input_count = BATCH_SIZE * IN_CHANNELS * IN_HEIGHT * IN_WIDTH
write_file("input_hex.txt", input_count, 1)

# Weights: 32x8x7x7 = 12544 values (all 1s)
weights_count = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
write_file("weights_hex.txt", weights_count, 1)

# Bias: 32 values (all 0s)
bias_count = OUT_CHANNELS
write_file("bias_hex.txt", bias_count, 0)

print("✅ input_hex.txt, weights_hex.txt, and bias_hex.txt created successfully.")
