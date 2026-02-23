// Bitline Accumulator: Sums currents over integration period
// Parameterized for COLS and bit-width to prevent overflow
module accumulator #(
    parameter DATA_WIDTH = 8,   // Width of individual current samples (e.g. 1-bit spikes)
    parameter ACC_WIDTH  = 16,  // Width of internal accumulator
    parameter COLS       = 32
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,     // Synchronous reset for integration
    input  wire [COLS-1:0]       spikes_in, // Input spikes from bitlines
    output reg  [ACC_WIDTH-1:0]  sum [0:COLS-1]
);
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < COLS; i = i + 1)
                sum[i] <= {ACC_WIDTH{1'b0}};
        end else if (clear) begin
            for (i = 0; i < COLS; i = i + 1)
                sum[i] <= {ACC_WIDTH{1'b0}};
        end else begin
            for (i = 0; i < COLS; i = i + 1) begin
                // Simple increment for binary spikes; could be weighted
                if (spikes_in[i])
                    sum[i] <= sum[i] + 1'b1;
            end
        end
    end
endmodule

