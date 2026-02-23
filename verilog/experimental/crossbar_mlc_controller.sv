/**
 * crossbar_mlc_controller.sv
 *
 * EXPERIMENTAL Development Module
 *
 * Implementation of a Multi-Level Cell (MLC) ReRAM controller.
 * Unlike SLC (Single-Level Cell) which stores 1 bit, this module
 * simulates 4-bit per cell (16 conductance levels) in hardware logic.
 */

module crossbar_mlc_controller #(
    parameter int ROWS = 32,
    parameter int COLS = 10,
    parameter int WEIGHT_PRECISION = 4 // 4-bit MLC
)(
    input  logic              clk,
    input  logic              rst_n,
    
    // Virtual Crossbar Interface
    input  logic [ROWS-1:0]   row_spikes,
    output logic [31:0]       weighted_sum[COLS],
    
    // Weight Programming (MLC)
    input  logic              prog_en,
    input  logic [4:0]        addr_row,
    input  logic [3:0]        addr_col,
    input  logic [WEIGHT_PRECISION-1:0] weight_val
);

    // MLC Weight Matrix (Internal emulation)
    logic [WEIGHT_PRECISION-1:0] G_matrix[ROWS][COLS];

    // Inference Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<ROWS; i++) begin
                for (int j=0; j<COLS; j++) begin
                    G_matrix[i][j] <= 0;
                end
            end
        end else if (prog_en) begin
            G_matrix[addr_row][addr_col] <= weight_val;
        end
    end

    // Combinatorial VMM Integration
    always_comb begin
        for (int j=0; j<COLS; j++) begin
            weighted_sum[j] = 0;
            for (int i=0; i<ROWS; i++) begin
                if (row_spikes[i]) begin
                    weighted_sum[j] += G_matrix[i][j];
                end
            end
        end
    end

endmodule
