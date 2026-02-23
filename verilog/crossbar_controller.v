// Crossbar Controller: Manages integration over multiple timesteps
module crossbar_controller #(
    parameter TIMESTEPS = 50,
    parameter TS_WIDTH  = 6     // log2(TIMESTEPS)
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          clear_acc,  // Reset accumulators
    output reg          apply_v,    // Drive rows
    output reg          sample_i,   // Sample columns
    output reg          done
);
    typedef enum reg [2:0] {
        S_IDLE     = 3'd0,
        S_CLEAR    = 3'd1,
        S_INTEGRATE = 3'd2,
        S_SAMPLE   = 3'd3,
        S_DONE     = 3'd4
    } state_t;

    state_t state;
    reg [TS_WIDTH-1:0] timer;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            timer     <= 0;
            clear_acc <= 1'b0;
            apply_v   <= 1'b0;
            sample_i  <= 1'b0;
            done      <= 1'b0;
        end else case (state)
            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    state     <= S_CLEAR;
                    clear_acc <= 1'b1;
                end
            end
            
            S_CLEAR: begin
                clear_acc <= 1'b0;
                apply_v   <= 1'b1;
                timer     <= 0;
                state     <= S_INTEGRATE;
            end
            
            S_INTEGRATE: begin
                if (timer == TIMESTEPS - 1) begin
                    apply_v <= 1'b0;
                    state   <= S_SAMPLE;
                end else begin
                    timer <= timer + 1;
                end
            end
            
            S_SAMPLE: begin
                sample_i <= 1'b1;
                state    <= S_DONE;
            end
            
            S_DONE: begin
                sample_i <= 1'b0;
                done     <= 1'b1;
                state    <= S_IDLE;
            end
            
            default: state <= S_IDLE;
        endcase
    end
endmodule

