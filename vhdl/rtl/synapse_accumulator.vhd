library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;
use work.synapse_params_pkg.all;

entity synapse_accumulator is
    port (
        features_in   : in  feature_array_t(0 to NET_NUM_INPUTS - 1);
        neuron_spikes : in  std_logic_vector(NET_NUM_LIF - 1 downto 0);
        currents_out  : out current_array_t(0 to NET_NUM_LIF - 1)
    );
end entity;

architecture combinational of synapse_accumulator is
begin
    process(features_in, neuron_spikes)
        variable acc_v    : current_array_t(0 to NET_NUM_LIF - 1);
        variable src_node : integer;
        variable tgt_node : integer;
        variable src_idx  : integer;
        variable tgt_idx  : integer;
        variable raw_i    : integer;
        variable drive_i  : integer;
    begin
        for i in 0 to NET_NUM_LIF - 1 loop
            acc_v(i) := to_signed(0, CURRENT_WIDTH);
        end loop;

        for s in 0 to NET_NUM_SYNAPSES - 1 loop
            src_node := SYN_SRC(s);
            tgt_node := SYN_TGT(s);
            drive_i  := 0;

            if src_node >= 0 and src_node < integer(NET_NUM_INPUTS) then
                -- C original: (int32_t)((float)weight * input_value)
                -- VHDL: input_value esta em Q8.8, entao reescala para inteiro.
                raw_i   := to_integer(features_in(src_node)) * SYN_W(s);
                drive_i := trunc_div_pow2_i(raw_i, FEATURE_FRAC_BITS);
            else
                src_idx := src_node - NET_FIRST_LIF_NODE;
                if src_idx >= 0 and src_idx < integer(NET_NUM_LIF) then
                    if neuron_spikes(src_idx) = '1' then
                        drive_i := SYN_W(s);
                    end if;
                end if;
            end if;

            tgt_idx := tgt_node - NET_FIRST_LIF_NODE;
            if tgt_idx >= 0 and tgt_idx < integer(NET_NUM_LIF) then
                acc_v(tgt_idx) := acc_v(tgt_idx) + to_signed(drive_i, CURRENT_WIDTH);
            end if;
        end loop;

        currents_out <= acc_v;
    end process;
end architecture;
