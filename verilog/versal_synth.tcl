# Vivado Synthesis Script for Neuro-Edge Verilog Modules
# Target: Xilinx Versal ACAP (e.g., vck190)

# 1. Setup project
create_project -force neuro_edge_versal ./verilog/vivado_project -part xcvc1902-vsva2197-2MP-e-S

# 2. Add source files
add_files g:/zserious project/reram-simulation-main/verilog/spike_encoder.sv
add_files g:/zserious project/reram-simulation-main/verilog/crossbar_controller.sv
add_files g:/zserious project/reram-simulation-main/verilog/accumulator.sv

# 3. Set top module (depending on use case)
set_property top crossbar_controller [current_fileset]

# 4. Run Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# 5. Open report
open_run synth_1 -name synth_1
report_utilization -file synthesis_utilization.rpt
report_timing_summary -file synthesis_timing.rpt

puts "Synthesis for Versal target complete."
