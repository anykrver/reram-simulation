`timescale 1ns/1ps

module tb_neuro_edge;

    parameter TIMESTEPS = 10;
    parameter ROWS = 4;
    parameter COLS = 4;

    reg clk;
    reg rst_n;
    reg start;
    
    wire clear_acc;
    wire apply_v;
    wire sample_i;
    wire done;

    // Instantiate Controller
    crossbar_controller #(
        .TIMESTEPS(TIMESTEPS),
        .TS_WIDTH(4)
    ) dut_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .clear_acc(clear_acc),
        .apply_v(apply_v),
        .sample_i(sample_i),
        .done(done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test sequence
    initial begin
        $display("Starting hardware simulation...");
        rst_n = 0;
        start = 0;
        #20;
        rst_n = 1;
        #20;
        
        // Trigger integration
        start = 1;
        #10;
        start = 0;
        
        // Wait for done
        wait(done);
        #50;
        
        $display("Simulation complete. Controller cycled through %d timesteps.", TIMESTEPS);
        $finish;
    end

    // Waveform dump
    initial begin
        $dumpfile("neuro_edge_hw.vcd");
        $dumpvars(0, tb_neuro_edge);
    end

endmodule
