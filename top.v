module weight_loader #(
    parameter OUT_CHANNELS = 4,
    parameter IN_CHANNELS  = 3,
    parameter KERNEL_SIZE  = 3,
    parameter DATA_WIDTH   = 32
)(
    input  logic clk,
    input  logic sel_bias, // 1: output bias, 0: output weights
    output logic [(TOTAL_OUTPUT_WIDTH-1):0] data_out
);

    localparam TOTAL_WEIGHTS = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    localparam TOTAL_BIASES  = OUT_CHANNELS;
    localparam MAX_COUNT     = (TOTAL_WEIGHTS > TOTAL_BIASES) ? TOTAL_WEIGHTS : TOTAL_BIASES;
    localparam TOTAL_OUTPUT_WIDTH = MAX_COUNT * DATA_WIDTH;

    // Declare memories
    logic [DATA_WIDTH-1:0] weights [0:TOTAL_WEIGHTS-1];
    logic [DATA_WIDTH-1:0] bias    [0:TOTAL_BIASES-1];

    // Flattened outputs
    logic [TOTAL_OUTPUT_WIDTH-1:0] flat_weights;
    logic [TOTAL_OUTPUT_WIDTH-1:0] flat_bias;

    // Load weights and biases from files
    initial begin
        $readmemh("weights.txt", weights);
        $readmemh("bias.txt", bias);
    end

    // Flatten arrays
    integer i;
    always_comb begin
        flat_weights = '0;
        for (i = 0; i < TOTAL_WEIGHTS; i = i + 1) begin
            flat_weights[i*DATA_WIDTH +: DATA_WIDTH] = weights[i];
        end

        flat_bias = '0;
        for (i = 0; i < TOTAL_BIASES; i = i + 1) begin
            flat_bias[i*DATA_WIDTH +: DATA_WIDTH] = bias[i];
        end
    end

    // Output the selected array
    always_ff @(posedge clk) begin
        data_out <= (sel_bias) ? flat_bias : flat_weights;
    end

endmodule

module conv_dimensions(
    input  [15:0] IN_HEIGHT,
    input  [15:0] IN_WIDTH,
    input  [15:0] KERNEL_SIZE,
    input  [15:0] STRIDE,
    input  [15:0] PADDING,
    output reg [15:0] OUT_HEIGHT,
    output reg [15:0] OUT_WIDTH
);

    always @(*) begin
        if (STRIDE != 0) begin
            OUT_HEIGHT = (IN_HEIGHT + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;
            OUT_WIDTH  = (IN_WIDTH  + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;
        end else begin
            OUT_HEIGHT = 16'd0;  // Prevent divide-by-zero
            OUT_WIDTH  = 16'd0;
        end
    end

endmodule
module padding_check #(
    parameter COORD_WIDTH = 16,
    parameter DATA_WIDTH  = 8
)(
    input  signed [COORD_WIDTH-1:0] coord,       // Coordinate (can be negative)
    input         [COORD_WIDTH-1:0] max_coord,   // Maximum allowed coordinate
    input         [DATA_WIDTH-1:0]  memory_data, // Input data from memory

    output                          valid_coord, // 1 if within bounds
    output        [DATA_WIDTH-1:0]  input_data   // Padded data output
);

    // Boundary check
    assign valid_coord = (coord >= 0) && (coord < max_coord);

    // Apply zero padding if out of bounds
    assign input_data = valid_coord ? memory_data : {DATA_WIDTH{1'b0}};

endmodule

module input_coordinate_calc #(
    parameter ADDR_WIDTH = 16,
    parameter STRIDE     = 1,
    parameter PADDING    = 1
)(
    input  wire [ADDR_WIDTH-1:0] output_coord,
    input  wire [ADDR_WIDTH-1:0] kernel_coord,
    output wire signed [ADDR_WIDTH:0] input_coord  // one bit wider for signed result
);

    // Intermediate wires for better synthesis
    wire [ADDR_WIDTH-1:0] stride_mult;
    assign stride_mult = output_coord * STRIDE;

    // Final input coordinate calculation
    assign input_coord = $signed(stride_mult + kernel_coord) - PADDING;

endmodule

module conv_forward_pass (
    input clk,
    input rst,

    // Flattened input tensor: [BATCH][IN_CH][IN_HEIGHT][IN_WIDTH]
    input [31:0] input_tensor [0:IN_CHANNELS-1][0:IN_HEIGHT-1][0:IN_WIDTH-1],

    // Weights: [OUT_CH][IN_CH][K_H][K_W]
    input [31:0] weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],

    // Biases: [OUT_CH]
    input [31:0] bias [0:OUT_CHANNELS-1],

    // Output tensor: [OUT_CH][OUT_HEIGHT][OUT_WIDTH]
    output reg [31:0] output_tensor [0:OUT_CHANNELS-1][0:OUT_HEIGHT-1][0:OUT_WIDTH-1]
);

    integer out_ch, in_ch, out_h, out_w, k_h, k_w;
    integer input_h, input_w;
    reg signed [31:0] input_val, weight_val, acc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // Clear output tensor
            for (out_ch = 0; out_ch < OUT_CHANNELS; out_ch = out_ch + 1)
                for (out_h = 0; out_h < OUT_HEIGHT; out_h = out_h + 1)
                    for (out_w = 0; out_w < OUT_WIDTH; out_w = out_w + 1)
                        output_tensor[out_ch][out_h][out_w] <= 32'd0;
        end else begin
            // Main forward convolution pass
            for (out_ch = 0; out_ch < OUT_CHANNELS; out_ch = out_ch + 1) begin
                for (out_h = 0; out_h < OUT_HEIGHT; out_h = out_h + 1) begin
                    for (out_w = 0; out_w < OUT_WIDTH; out_w = out_w + 1) begin
                        acc = bias[out_ch];  // Start with bias
                        for (in_ch = 0; in_ch < IN_CHANNELS; in_ch = in_ch + 1) begin
                            for (k_h = 0; k_h < KERNEL_SIZE; k_h = k_h + 1) begin
                                for (k_w = 0; k_w < KERNEL_SIZE; k_w = k_w + 1) begin
                                    
                                    input_h = out_h * STRIDE + k_h - PADDING;
                                    input_w = out_w * STRIDE + k_w - PADDING;

                                    // Check bounds for zero-padding
                                    if ((input_h >= 0) && (input_h < IN_HEIGHT) &&
                                        (input_w >= 0) && (input_w < IN_WIDTH)) begin
                                        input_val = input_tensor[in_ch][input_h][input_w];
                                    end else begin
                                        input_val = 32'd0;
                                    end

                                    weight_val = weights[out_ch][in_ch][k_h][k_w];
                                    acc = acc + input_val * weight_val;
                                end
                            end
                        end
                        output_tensor[out_ch][out_h][out_w] <= acc;
                    end
                end
            end
        end
    end
endmodule
