library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

entity neuromorphic_sca_top is
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;
        start    : in  std_logic;
        hw_class : out std_logic_vector(CLASS_WIDTH - 1 downto 0);
        valid    : out std_logic
    );
end entity;

architecture structural of neuromorphic_sca_top is
    signal addr_count_en_s  : std_logic;
    signal spike_count_en_s : std_logic;
    signal tick_count_en_s  : std_logic;
    signal rd_en_s          : std_logic;
    signal load_s           : std_logic;
    signal enable_s         : std_logic;
    signal clear_s          : std_logic;
    signal tick_s           : std_logic;
    signal latch_out_s      : std_logic;
    signal trace_done_s     : std_logic;
    signal inference_done_s : std_logic;
    signal hw_class_s       : class_t;
begin
    u_bc : entity work.controller_fsm
        port map (
            clk            => clk,
            rst            => rst,
            start          => start,
            trace_done     => trace_done_s,
            inference_done => inference_done_s,
            addr_count_en  => addr_count_en_s,
            spike_count_en => spike_count_en_s,
            tick_count_en  => tick_count_en_s,
            rd_en          => rd_en_s,
            load           => load_s,
            enable         => enable_s,
            clear          => clear_s,
            tick           => tick_s,
            latch_out      => latch_out_s,
            valid          => valid
        );

    u_bo : entity work.datapath
        port map (
            clk            => clk,
            rst            => rst,
            addr_count_en  => addr_count_en_s,
            spike_count_en => spike_count_en_s,
            tick_count_en  => tick_count_en_s,
            rd_en          => rd_en_s,
            load           => load_s,
            enable         => enable_s,
            clear          => clear_s,
            tick           => tick_s,
            latch_out      => latch_out_s,
            trace_done     => trace_done_s,
            inference_done => inference_done_s,
            hw_class       => hw_class_s
        );

    hw_class <= std_logic_vector(hw_class_s);
end architecture;
