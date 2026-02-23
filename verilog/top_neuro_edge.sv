/**
 * top_neuro_edge.sv
 *
 * Top-level synthesisable wrapper for the Neuro-Edge ReRAM Accelerator.
 * Targets: Xilinx Versal ACAP.
 *
 * This module integrates the full inference pipeline:
 * 1. Spike Encoding (Poisson)
 * 2. Crossbar Control (VMM)
 * 3. Spike accumulation
 */

module top_neuro_edge #(
    parameter int N_INPUTS  = 784,
    parameter int N_OUTPUTS = 10,
    parameter int DATA_WIDTH = 8
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   start,
    
    // Weight Configuration Port
    input  logic [9:0]             weight_addr_row,
    input  logic [3:0]             weight_addr_col,
    input  logic [DATA_WIDTH-1:0]  weight_data,
    input  logic                   weight_wen,

    // Input Data (784 pixels, simplified burst interface)
    input  logic [DATA_WIDTH-1:0]  pixel_data,
    input  logic                   pixel_valid,
    
    // Output Interface
    output logic [31:0]            spike_counts[N_OUTPUTS],
    output logic                   done
);

    // Internal signals
    logic [N_INPUTS-1:0]  input_spikes;
    logic [N_OUTPUTS-1:0] output_spikes;
    
    // 1. Spike Encoder Array
    // (Simplified: Single encoder per cycle or parallel array)
    spike_encoder #(
        .WIDTH(DATA_WIDTH)
    ) encoder_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(pixel_data),
        .valid_in(pixel_valid),
        .spike_out(input_spikes[0]) // Simplified routing
    );

    // 2. Crossbar Controller
    crossbar_controller #(
        .ROWS(N_INPUTS),
        .COLS(N_OUTPUTS),
        .DATA_WIDTH(DATA_WIDTH)
    ) controller_inst (
        .clk(clk),
        .rst_n(rst_n),
        .row_spikes(input_spikes),
        .col_spikes(output_spikes),
        .weight_wen(weight_wen),
        .weight_data(weight_data)
    );

    // 3. Accumulators
    accumulator #(
        .N(N_OUTPUTS)
    ) acc_inst (
        .clk(clk),
        .rst_n(rst_n),
        .spikes_in(output_spikes),
        .counts(spike_counts)
    );

    assign done = ~start; // Placeholder logic

endmodule
