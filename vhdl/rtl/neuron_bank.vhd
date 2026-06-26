library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;
use work.neuron_params_pkg.all;

entity neuron_bank is
    port (
        clk         : in  std_logic;
        rst         : in  std_logic;
        clear       : in  std_logic;
        enable      : in  std_logic;
        currents_in : in  current_array_t(0 to NET_NUM_LIF - 1);
        lif_spikes  : out std_logic_vector(NET_NUM_LIF - 1 downto 0);
        out_spikes  : out std_logic_vector(NUM_CLASSES - 1 downto 0)
    );
end entity;

architecture structural of neuron_bank is
    signal spikes_s : std_logic_vector(NET_NUM_LIF - 1 downto 0) := (others => '0');
begin
    ---------------------------------------------------------------------------
    -- Bloco de neuronios separado.
    -- As primeiras NET_NUM_OUTPUTS posicoes sao as classes HW 0..8.
    ---------------------------------------------------------------------------
    gen_neurons : for i in 0 to NET_NUM_LIF - 1 generate
        u_lif : entity work.lif_neuron
            generic map (
                THRESHOLD         => NEURON_THRESHOLD(i),
                LEAK_FACTOR       => natural(NEURON_LEAK_FACTOR(i)),
                RESET_POTENTIAL   => NEURON_RESET_POTENTIAL(i),
                RESTING_POTENTIAL => NEURON_RESTING_POTENTIAL(i),
                REFRACTORY_PERIOD => natural(NEURON_REFRACTORY_PERIOD(i))
            )
            port map (
                clk           => clk,
                rst           => rst,
                clear         => clear,
                enable        => enable,
                input_current => currents_in(i),
                fired         => spikes_s(i),
                potential_out => open
            );
    end generate;

    lif_spikes <= spikes_s;

    gen_outputs : for c in 0 to NUM_CLASSES - 1 generate
        out_spikes(c) <= spikes_s(c);
    end generate;
end architecture;
