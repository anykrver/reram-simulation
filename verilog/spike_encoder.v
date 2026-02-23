// Poisson Spike Encoder: LFSR-driven comparator
// Generates a spike if random < rate
module spike_encoder #(
    parameter RATE_WIDTH = 8,
    parameter LFSR_WIDTH = 16
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  en,        // Enable encoding
    input  wire [RATE_WIDTH-1:0] rate,      // Target firing rate
    output reg                   spike
);
    // 16-bit Galois LFSR (polynomial: x^16 + x^14 + x^13 + x^11 + 1)
    reg [LFSR_WIDTH-1:0] lfsr;
    wire feedback = lfsr[0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr  <= 16'hACE1; // Seed
            spike <= 1'b0;
        end else if (en) begin
            // Shift and XOR for Galois LFSR
            lfsr <= (lfsr >> 1) ^ (feedback ? 16'hB400 : 16'h0000);
            
            // Compare rate with bits from LFSR
            // We use the 8 MSB of the LFSR for comparison
            spike <= (lfsr[LFSR_WIDTH-1 -: RATE_WIDTH] < rate);
        end else begin
            spike <= 1'b0;
        end
    end
endmodule

