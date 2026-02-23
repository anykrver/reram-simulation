# Xilinx Versal Hardware Constraints (Template)
# Target: Xilinx Versal ACAP (e.g., VCK190)

# Main System Clock (300 MHz Differential Example)
# set_property PACKAGE_PIN AR20 [get_ports clk_p]
# set_property PACKAGE_PIN AR21 [get_ports clk_n]
# set_property IOSTANDARD DIFF_SSTL12 [get_ports clk_p]
# set_property IOSTANDARD DIFF_SSTL12 [get_ports clk_n]

# Single-Ended Clock Example (for simulation/emulation)
create_clock -period 4.000 -name sys_clk [get_ports clk]

# System Reset
# set_property PACKAGE_PIN J17 [get_ports rst_n]
# set_property IOSTANDARD LVCMOS12 [get_ports rst_n]

# IO constraints for output spikes (demo)
# set_property IOSTANDARD LVCMOS12 [get_ports {done}]
# set_property PACKAGE_PIN L14 [get_ports {done}]
