library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

entity snn_core is
    port (
        clk         : in  std_logic;
        rst         : in  std_logic;
        clear       : in  std_logic;
        enable      : in  std_logic;
        tick        : in  std_logic;
        features_in : in  feature_array_t(0 to NET_NUM_INPUTS - 1);
        out_spikes  : out std_logic_vector(NUM_CLASSES - 1 downto 0)
    );
end entity;

architecture structural of snn_core is
    signal lif_enable   : std_logic;
    signal lif_spikes   : std_logic_vector(NET_NUM_LIF - 1 downto 0) := (others => '0');
    signal acc_currents : current_array_t(0 to NET_NUM_LIF - 1);
begin
    lif_enable <= enable and tick;

    u_synapses : entity work.synapse_accumulator
        port map (
            features_in   => features_in,
            neuron_spikes => lif_spikes,
            currents_out  => acc_currents
        );

    u_neurons : entity work.neuron_bank
        port map (
            clk         => clk,
            rst         => rst,
            clear       => clear,
            enable      => lif_enable,
            currents_in => acc_currents,
            lif_spikes  => lif_spikes,
            out_spikes  => out_spikes
        );
end architecture;
