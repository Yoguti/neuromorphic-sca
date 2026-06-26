library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

entity datapath is
    port (
        clk            : in  std_logic;
        rst            : in  std_logic;

        addr_count_en  : in  std_logic;
        spike_count_en : in  std_logic;
        tick_count_en  : in  std_logic;
        rd_en          : in  std_logic;
        load           : in  std_logic;
        enable         : in  std_logic;
        clear          : in  std_logic;
        tick           : in  std_logic;
        latch_out      : in  std_logic;

        trace_done     : out std_logic;
        inference_done : out std_logic;
        hw_class       : out class_t
    );
end entity;

architecture structural of datapath is
    signal addr       : unsigned(ADDR_WIDTH - 1 downto 0);
    signal rom_data   : feature_t;
    signal features   : feature_array_t(0 to NET_NUM_INPUTS - 1);
    signal spikes     : std_logic_vector(NUM_CLASSES - 1 downto 0);
    signal out_counts : class_count_array_t(0 to NUM_CLASSES - 1);
begin
    u_addr : entity work.addr_counter
        port map (
            clk        => clk,
            rst        => rst,
            clear      => clear,
            count_en   => addr_count_en,
            addr       => addr,
            trace_done => trace_done
        );

    u_mem : entity work.trace_rom
        port map (
            clk   => clk,
            rd_en => rd_en,
            addr  => addr,
            data  => rom_data
        );

    u_input_buffer : entity work.input_buffer
        port map (
            clk          => clk,
            rst          => rst,
            clear        => clear,
            load         => load,
            sample_index => addr,
            feature_in   => rom_data,
            features_out => features
        );

    u_core : entity work.snn_core
        port map (
            clk         => clk,
            rst         => rst,
            clear       => clear,
            enable      => enable,
            tick        => tick,
            features_in => features,
            out_spikes  => spikes
        );

    u_tick_counter : entity work.inference_counter
        port map (
            clk        => clk,
            rst        => rst,
            clear      => clear,
            count_en   => tick_count_en,
            ticks_done => inference_done
        );

    u_out_reg : entity work.output_register
        port map (
            clk        => clk,
            rst        => rst,
            clear      => clear,
            count_en   => spike_count_en,
            latch_out  => latch_out,
            spikes_in  => spikes,
            counts_out => out_counts
        );

    u_wta : entity work.wta
        port map (
            counts_in => out_counts,
            hw_class  => hw_class
        );
end architecture;
