import struct

def float_to_hex(f):
    return format(struct.unpack('<I', struct.pack('<f', float(f)))[0], '08X')

# Convert weights
with open("weights.txt") as f:
    weights = [float_to_hex(line.strip()) for line in f if line.strip() and not line.startswith("#")]
with open("weights_hex.txt", "w") as f:
    f.write("\n".join(weights))

# Convert biases
with open("bias.txt") as f:
    biases = [float_to_hex(line.strip()) for line in f if line.strip() and not line.startswith("#")]
with open("bias_hex.txt", "w") as f:
    f.write("\n".join(biases))

# Convert input
with open("input.txt") as f:
    biases = [float_to_hex(line.strip()) for line in f if line.strip() and not line.startswith("#")]
with open("input_hex.txt", "w") as f:
    f.write("\n".join(biases))