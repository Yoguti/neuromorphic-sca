library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

package neuron_params_pkg is
    ---------------------------------------------------------------------------
    -- Parâmetros reais de snn/network-csvs/best_network_neurons.csv.
    -- Índice interno lif_idx = node_id - 20.
    -- 0..8  => saidas HD 0..8
    -- 9..12 => hidden neurons 29..32
    ---------------------------------------------------------------------------
    constant NEURON_THRESHOLD : int_array_t(0 to NET_NUM_LIF - 1) := (
        23, 20, 11, 26, 20, 20, 20, 20, 20, 18, 20, 20, 20
    );

    constant NEURON_LEAK_FACTOR : int_array_t(0 to NET_NUM_LIF - 1) := (
        3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3
    );

    constant NEURON_RESET_POTENTIAL : int_array_t(0 to NET_NUM_LIF - 1) := (
        -8, -5, 0, -5, -5, -5, -5, -9, -5, -5, -5, -5, -5
    );

    constant NEURON_RESTING_POTENTIAL : int_array_t(0 to NET_NUM_LIF - 1) := (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    );

    constant NEURON_REFRACTORY_PERIOD : int_array_t(0 to NET_NUM_LIF - 1) := (
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    );
end package;
