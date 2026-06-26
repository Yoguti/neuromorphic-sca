library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.trace_data_pkg.all;

entity trace_rom is
    port (
        clk   : in  std_logic;
        rd_en : in  std_logic;
        addr  : in  unsigned(ADDR_WIDTH - 1 downto 0);
        data  : out feature_t
    );
end entity;

architecture rtl of trace_rom is
    signal data_reg : feature_t := (others => '0');
begin
    -- Leitura sincronizada para FPGA/Quartus inferir ROM.
    process(clk)
    begin
        if rising_edge(clk) then
            if rd_en = '1' then
                if to_integer(addr) < TRACE_LEN then
                    data_reg <= TRACE_FEATURES(to_integer(addr));
                else
                    data_reg <= (others => '0');
                end if;
            end if;
        end if;
    end process;

    data <= data_reg;
end architecture;
