library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

entity output_register is
    port (
        clk        : in  std_logic;
        rst        : in  std_logic;
        clear      : in  std_logic;
        count_en   : in  std_logic;
        latch_out  : in  std_logic;
        spikes_in  : in  std_logic_vector(NUM_CLASSES - 1 downto 0);
        counts_out : out class_count_array_t(0 to NUM_CLASSES - 1)
    );
end entity;

architecture rtl of output_register is
    constant MAX_COUNT : count_t := (others => '1');
    signal accum_counts   : class_count_array_t(0 to NUM_CLASSES - 1) := (others => (others => '0'));
    signal latched_counts : class_count_array_t(0 to NUM_CLASSES - 1) := (others => (others => '0'));
begin
    process(clk, rst)
    begin
        if rst = '1' then
            accum_counts <= (others => (others => '0'));
            latched_counts <= (others => (others => '0'));
        elsif rising_edge(clk) then
            if clear = '1' then
                accum_counts <= (others => (others => '0'));
                latched_counts <= (others => (others => '0'));
            else
                if count_en = '1' then
                    for i in 0 to NUM_CLASSES - 1 loop
                        if spikes_in(i) = '1' and accum_counts(i) /= MAX_COUNT then
                            accum_counts(i) <= accum_counts(i) + 1;
                        end if;
                    end loop;
                end if;

                if latch_out = '1' then
                    latched_counts <= accum_counts;
                end if;
            end if;
        end if;
    end process;

    counts_out <= latched_counts;
end architecture;
