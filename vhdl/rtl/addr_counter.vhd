library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

entity addr_counter is
    port (
        clk        : in  std_logic;
        rst        : in  std_logic;
        clear      : in  std_logic;
        count_en   : in  std_logic;
        addr       : out unsigned(ADDR_WIDTH - 1 downto 0);
        trace_done : out std_logic
    );
end entity;

architecture rtl of addr_counter is
    signal addr_reg : unsigned(ADDR_WIDTH - 1 downto 0) := (others => '0');
begin
    process(clk, rst)
    begin
        if rst = '1' then
            addr_reg <= (others => '0');
        elsif rising_edge(clk) then
            if clear = '1' then
                addr_reg <= (others => '0');
            elsif count_en = '1' then
                if to_integer(addr_reg) < integer(NET_NUM_INPUTS) - 1 then
                    addr_reg <= addr_reg + 1;
                end if;
            end if;
        end if;
    end process;

    addr <= addr_reg;
    trace_done <= '1' when to_integer(addr_reg) = integer(NET_NUM_INPUTS) - 1 else '0';
end architecture;
