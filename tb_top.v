`timescale 1ns / 1ps

module tb_conv1;

  parameter DATA_WIDTH   = 32;
  parameter BATCH_SIZE   = 1;
  parameter IN_CHANNELS  = 8;
  parameter IN_HEIGHT    = 7;
  parameter IN_WIDTH     = 7;
  parameter OUT_CHANNELS = 32;
  parameter KERNEL_SIZE  = 7;
  parameter STRIDE       = 1;
  parameter PADDING      = 3;
  parameter OUT_HEIGHT   = (IN_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
  parameter OUT_WIDTH    = (IN_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;

  reg clk, rst;
  reg [BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH*DATA_WIDTH-1:0] input_tensor_flat;
  reg [OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_flat;
  reg [OUT_CHANNELS*DATA_WIDTH-1:0] bias_flat;
  wire [BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH*DATA_WIDTH-1:0] output_tensor_flat;

  // Instantiate DUT
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

  // Memory arrays
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

    $display("🚀 Simulation started");

    // Load values
    $display("📥 Loading input...");
    $readmemh("input_hex.txt", input_tensor);
    $display("📥 Loading weights...");
    $readmemh("weights_hex.txt", weights_tensor);
    $display("📥 Loading bias...");
    $readmemh("bias_hex.txt", bias_tensor);

    $display("🔃 Flattening input tensor...");
    for (i = 0; i < BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH; i = i + 1)
      input_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH] = input_tensor[i];

    $display("🔃 Flattening weights tensor...");
    for (i = 0; i < OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE; i = i + 1)
      weights_flat[i*DATA_WIDTH +: DATA_WIDTH] = weights_tensor[i];

    $display("🔃 Flattening bias tensor...");
    for (i = 0; i < OUT_CHANNELS; i = i + 1)
      bias_flat[i*DATA_WIDTH +: DATA_WIDTH] = bias_tensor[i];

    $display("🛑 Deasserting reset in 20ns...");
    #20 rst = 0;
    $display("▶️  Running simulation...");

    #2000;

    $display("💾 Writing output to output_hex.txt...");
    output_file = $fopen("output_hex.txt", "w");

    for (i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1) begin
      $display("Output[%0d] = %h", i, output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH]);
      $fdisplay(output_file, "%h", output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH]);
    end

    $fclose(output_file);
    $display("✅ Simulation completed, output saved.");
    $finish;
  end

endmodule
