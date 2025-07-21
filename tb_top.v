`timescale 1ns/1ps

module tb_top;

  parameter DATA_WIDTH = 32;

  // Parameters from top module
  parameter BATCH_SIZE   = 1;
  parameter IN_CHANNELS  = 2;
  parameter IN_HEIGHT    = 4;
  parameter IN_WIDTH     = 4;
  parameter OUT_CHANNELS = 1;
  parameter KERNEL_SIZE  = 2;
  parameter STRIDE       = 2;
  parameter PADDING      = 0;
  parameter OUT_HEIGHT   = (IN_HEIGHT + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;
  parameter OUT_WIDTH    = (IN_WIDTH  + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;

  // Signals
  reg clk, rst;
  reg [BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH*DATA_WIDTH-1:0] input_tensor_flat;
  reg [OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_flat;
  reg [OUT_CHANNELS*DATA_WIDTH-1:0] bias_flat;
  wire [BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH*DATA_WIDTH-1:0] output_tensor_flat;

  // Instantiate top module
  top uut (
    .clk(clk),
    .rst(rst),
    .input_tensor_flat(input_tensor_flat),
    .weights_flat(weights_flat),
    .bias_flat(bias_flat),
    .output_tensor_flat(output_tensor_flat)
  );

  // Clock
  always #5 clk = ~clk;

  // Memory arrays for file loading
  reg [DATA_WIDTH-1:0] input_tensor   [0:BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH-1];
  reg [DATA_WIDTH-1:0] weights_tensor [0:OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE-1];
  reg [DATA_WIDTH-1:0] bias_tensor    [0:OUT_CHANNELS-1];

  integer i;

  initial begin
    // Initialize
    clk = 0;
    rst = 1;
    input_tensor_flat = 0;
    weights_flat = 0;
    bias_flat = 0;

    // Read hex values
    $readmemh("input_data.txt", input_tensor);    // already in hex
    $readmemh("weights_hex.txt", weights_tensor); // you must generate this from weights.txt
    $readmemh("bias_hex.txt", bias_tensor);       // generate this too

    // Flatten inputs
    for (i = 0; i < BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH; i = i + 1)
      input_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH] = input_tensor[i];

    for (i = 0; i < OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE; i = i + 1)
      weights_flat[i*DATA_WIDTH +: DATA_WIDTH] = weights_tensor[i];

    for (i = 0; i < OUT_CHANNELS; i = i + 1)
      bias_flat[i*DATA_WIDTH +: DATA_WIDTH] = bias_tensor[i];

    // Release reset
    #20 rst = 0;

    // Wait for convolution to complete
    #200;

    // Print output
    for (i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1)
      $display("Output[%0d] = %h", i, output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH]);

    $finish;
  end

endmodule
