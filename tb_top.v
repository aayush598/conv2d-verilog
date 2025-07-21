`timescale 1ns/1ps

module tb_top;

  parameter DATA_WIDTH   = 32;
  parameter BATCH_SIZE   = 1;
  parameter IN_CHANNELS  = 3;
  parameter IN_HEIGHT    = 8;
  parameter IN_WIDTH     = 8;
  parameter OUT_CHANNELS = 4;
  parameter KERNEL_SIZE  = 3;
  parameter STRIDE       = 1;
  parameter PADDING      = 1;
  parameter OUT_HEIGHT   = (IN_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
  parameter OUT_WIDTH    = (IN_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;

  reg clk, rst;
  reg [BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH*DATA_WIDTH-1:0] input_tensor_flat;
  reg [OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_flat;
  reg [OUT_CHANNELS*DATA_WIDTH-1:0] bias_flat;
  wire [BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH*DATA_WIDTH-1:0] output_tensor_flat;

  // Instantiate the DUT
  top #(
    .BATCH_SIZE(BATCH_SIZE),
    .IN_CHANNELS(IN_CHANNELS),
    .OUT_CHANNELS(OUT_CHANNELS),
    .IN_HEIGHT(IN_HEIGHT),
    .IN_WIDTH(IN_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .DATA_WIDTH(DATA_WIDTH)
  ) uut (
    .clk(clk),
    .rst(rst),
    .input_tensor_flat(input_tensor_flat),
    .weights_flat(weights_flat),
    .bias_flat(bias_flat),
    .output_tensor_flat(output_tensor_flat)
  );

  // Clock generation
  always #5 clk = ~clk;

  // Memory for loading hex files
  reg [DATA_WIDTH-1:0] input_tensor   [0:BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH-1];
  reg [DATA_WIDTH-1:0] weights_tensor [0:OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE-1];
  reg [DATA_WIDTH-1:0] bias_tensor    [0:OUT_CHANNELS-1];

  integer i;
  integer output_file;

  initial begin
    clk = 0;
    rst = 1;
    input_tensor_flat = 0;
    weights_flat = 0;
    bias_flat = 0;

    // Load values from hex files (created by Python script)
    $readmemh("input_hex.txt", input_tensor);
    $readmemh("weights_hex.txt", weights_tensor);
    $readmemh("bias_hex.txt", bias_tensor);

    // Flatten input tensor
    for (i = 0; i < BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH; i = i + 1)
      input_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH] = input_tensor[i];

    // Flatten weights
    for (i = 0; i < OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE; i = i + 1)
      weights_flat[i*DATA_WIDTH +: DATA_WIDTH] = weights_tensor[i];

    // Flatten bias
    for (i = 0; i < OUT_CHANNELS; i = i + 1)
      bias_flat[i*DATA_WIDTH +: DATA_WIDTH] = bias_tensor[i];

    // Deassert reset after 20ns
    #20 rst = 0;

    // Wait for output to settle
    #500;

    // Open output file for writing
    output_file = $fopen("output_hex.txt", "w");
    if (output_file == 0) begin
      $display("❌ Error: Could not open output_hex.txt for writing");
      $finish;
    end

    // Display & Write output
    for (i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1) begin
      $display("Output[%0d] = %h", i, output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH]);
      $fdisplay(output_file, "%h", output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH]);
    end

    $fclose(output_file);
    $display("✅ Output written to output_hex.txt");
    $finish;
  end

endmodule
