library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

entity inference_counter is
    port (
        clk        : in  std_logic;
        rst        : in  std_logic;
        clear      : in  std_logic;
        count_en   : in  std_logic;
        ticks_done : out std_logic
    );
end entity;

architecture rtl of inference_counter is
    signal tick_reg : unsigned(TICK_WIDTH - 1 downto 0) := (others => '0');
begin
    process(clk, rst)
    begin
        if rst = '1' then
            tick_reg <= (others => '0');
        elsif rising_edge(clk) then
            if clear = '1' then
                tick_reg <= (others => '0');
            elsif count_en = '1' then
                if to_integer(tick_reg) < INFERENCE_TICKS - 1 then
                    tick_reg <= tick_reg + 1;
                end if;
            end if;
        end if;
    end process;

    ticks_done <= '1' when to_integer(tick_reg) = INFERENCE_TICKS - 1 else '0';
end architecture;
