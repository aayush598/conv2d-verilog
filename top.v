module top #(
    parameter BATCH_SIZE   = 1,
    parameter IN_CHANNELS  = 2,
    parameter OUT_CHANNELS = 1,
    parameter IN_HEIGHT    = 4,
    parameter IN_WIDTH     = 4,
    parameter KERNEL_SIZE  = 2,
    parameter STRIDE       = 2,
    parameter PADDING      = 0,
    parameter DATA_WIDTH   = 32
)(
    input clk,
    input rst,

    input  [BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH*DATA_WIDTH-1:0] input_tensor_flat,
    input  [OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_flat,
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] bias_flat,
    output reg [BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH*DATA_WIDTH-1:0] output_tensor_flat
);

    parameter OUT_HEIGHT = (IN_HEIGHT + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;
    parameter OUT_WIDTH  = (IN_WIDTH  + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;
    parameter TOTAL_WEIGHTS = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    parameter TOTAL_BIASES  = OUT_CHANNELS;
    parameter TOTAL_OUTPUT_WIDTH = (TOTAL_WEIGHTS > TOTAL_BIASES ? TOTAL_WEIGHTS : TOTAL_BIASES) * DATA_WIDTH;


    integer b, out_ch, in_ch, out_h, out_w, k_h, k_w;
    integer input_h, input_w;
    integer in_index, w_index, out_index;

    reg signed [DATA_WIDTH-1:0] input_val, weight_val, acc;

    // Internal unpacked memories
    reg [DATA_WIDTH-1:0] input_tensor  [0:BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH-1];
    reg [DATA_WIDTH-1:0] weights       [0:OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE-1];
    reg [DATA_WIDTH-1:0] bias          [0:OUT_CHANNELS-1];
    reg [DATA_WIDTH-1:0] output_tensor [0:BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH-1];

    initial begin
        $readmemh("weights.txt", weights);
        $readmemh("bias.txt", bias);
    end
    
    // Unpack input, weights, bias
    always @(*) begin
        for (integer i = 0; i < BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH; i = i + 1)
            input_tensor[i] = input_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH];
        for (integer i = 0; i < OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE; i = i + 1)
            weights[i] = weights_flat[i*DATA_WIDTH +: DATA_WIDTH];
        for (integer i = 0; i < OUT_CHANNELS; i = i + 1)
            bias[i] = bias_flat[i*DATA_WIDTH +: DATA_WIDTH];
    end

    // Convolution Logic (with BATCH)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (integer i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1)
                output_tensor[i] <= 0;
        end else begin
            for (b = 0; b < BATCH_SIZE; b = b + 1) begin
                for (out_ch = 0; out_ch < OUT_CHANNELS; out_ch = out_ch + 1) begin
                    for (out_h = 0; out_h < OUT_HEIGHT; out_h = out_h + 1) begin
                        for (out_w = 0; out_w < OUT_WIDTH; out_w = out_w + 1) begin
                            acc = bias[out_ch];
                            for (in_ch = 0; in_ch < IN_CHANNELS; in_ch = in_ch + 1) begin
                                for (k_h = 0; k_h < KERNEL_SIZE; k_h = k_h + 1) begin
                                    for (k_w = 0; k_w < KERNEL_SIZE; k_w = k_w + 1) begin
                                        input_h = out_h * STRIDE + k_h - PADDING;
                                        input_w = out_w * STRIDE + k_w - PADDING;

                                        if (input_h >= 0 && input_h < IN_HEIGHT &&
                                            input_w >= 0 && input_w < IN_WIDTH) begin
                                            in_index = b*IN_CHANNELS*IN_HEIGHT*IN_WIDTH +
                                                       in_ch*IN_HEIGHT*IN_WIDTH +
                                                       input_h*IN_WIDTH + input_w;
                                            input_val = input_tensor[in_index];
                                        end else begin
                                            input_val = 0;
                                        end

                                        w_index = out_ch*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE +
                                                  in_ch*KERNEL_SIZE*KERNEL_SIZE +
                                                  k_h*KERNEL_SIZE + k_w;
                                        weight_val = weights[w_index];

                                        acc = acc + input_val * weight_val;
                                    end
                                end
                            end
                            out_index = b*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH +
                                        out_ch*OUT_HEIGHT*OUT_WIDTH +
                                        out_h*OUT_WIDTH + out_w;
                            output_tensor[out_index] <= acc;
                        end
                    end
                end
            end
        end
    end

    // Pack output tensor to flat
    always @(*) begin
        for (integer i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1)
            output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH] = output_tensor[i];
    end

endmodule