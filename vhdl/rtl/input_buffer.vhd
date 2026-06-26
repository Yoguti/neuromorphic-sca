library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

entity input_buffer is
    port (
        clk          : in  std_logic;
        rst          : in  std_logic;
        clear        : in  std_logic;
        load         : in  std_logic;
        sample_index : in  unsigned(ADDR_WIDTH - 1 downto 0);
        feature_in   : in  feature_t;
        features_out : out feature_array_t(0 to NET_NUM_INPUTS - 1)
    );
end entity;

architecture rtl of input_buffer is
    signal regs : feature_array_t(0 to NET_NUM_INPUTS - 1) := (others => (others => '0'));
begin
    process(clk, rst)
        variable idx : integer;
    begin
        if rst = '1' then
            regs <= (others => (others => '0'));
        elsif rising_edge(clk) then
            if clear = '1' then
                regs <= (others => (others => '0'));
            elsif load = '1' then
                idx := to_integer(sample_index);
                if idx >= 0 and idx < integer(NET_NUM_INPUTS) then
                    regs(idx) <= feature_in;
                end if;
            end if;
        end if;
    end process;

    features_out <= regs;
end architecture;
