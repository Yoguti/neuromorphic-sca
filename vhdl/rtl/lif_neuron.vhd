library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

entity lif_neuron is
    generic (
        THRESHOLD         : integer := LIF_DEFAULT_THRESHOLD;
        LEAK_FACTOR       : natural := LIF_DEFAULT_LEAK_FACTOR;
        RESET_POTENTIAL   : integer := LIF_DEFAULT_RESET;
        RESTING_POTENTIAL : integer := LIF_DEFAULT_RESTING;
        REFRACTORY_PERIOD : natural := LIF_DEFAULT_REFRACTORY_PERIOD
    );
    port (
        clk           : in  std_logic;
        rst           : in  std_logic;
        clear         : in  std_logic;
        enable        : in  std_logic;
        input_current : in  current_t;
        fired         : out std_logic;
        potential_out : out pot_t
    );
end entity;

architecture rtl of lif_neuron is
    signal membrane_potential : pot_t := to_signed(LIF_DEFAULT_RESTING, POT_WIDTH);
    signal fired_reg          : std_logic := '0';
    signal refractory_counter : natural range 0 to 255 := 0;
begin
    process(clk, rst)
        variable v_int     : integer;
        variable delta_int : integer;
        variable leak_int  : integer;
    begin
        if rst = '1' then
            membrane_potential <= to_signed(RESTING_POTENTIAL, POT_WIDTH);
            refractory_counter <= 0;
            fired_reg <= '0';
        elsif rising_edge(clk) then
            if clear = '1' then
                membrane_potential <= to_signed(RESTING_POTENTIAL, POT_WIDTH);
                refractory_counter <= 0;
                fired_reg <= '0';
            elsif enable = '1' then
                if refractory_counter > 0 then
                    refractory_counter <= refractory_counter - 1;
                    fired_reg <= '0';
                else
                    v_int := to_integer(membrane_potential) + to_integer(input_current);
                    v_int := clamp_i(v_int, -32768, 32767);

                    delta_int := v_int - RESTING_POTENTIAL;
                    leak_int := arith_shift_right_i(delta_int, LEAK_FACTOR);

                    -- Replica a protecao do simulador C: se ha delta, mas o shift zerou,
                    -- aplica vazamento minimo de +/-1.
                    if delta_int /= 0 and leak_int = 0 then
                        if delta_int > 0 then
                            leak_int := 1;
                        else
                            leak_int := -1;
                        end if;
                    end if;

                    v_int := v_int - leak_int;
                    v_int := clamp_i(v_int, -32768, 32767);

                    if v_int >= THRESHOLD then
                        fired_reg <= '1';
                        refractory_counter <= REFRACTORY_PERIOD;
                        membrane_potential <= to_signed(RESET_POTENTIAL, POT_WIDTH);
                    else
                        fired_reg <= '0';
                        membrane_potential <= to_signed(v_int, POT_WIDTH);
                    end if;
                end if;
            end if;
        end if;
    end process;

    fired <= fired_reg;
    potential_out <= membrane_potential;
end architecture;
