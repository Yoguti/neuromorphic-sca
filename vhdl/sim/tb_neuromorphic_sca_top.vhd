library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

entity tb_neuromorphic_sca_top is
end entity;

architecture sim of tb_neuromorphic_sca_top is
    constant CLK_PERIOD : time := 10 ns;

    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal start    : std_logic := '0';
    signal hw_class : std_logic_vector(CLASS_WIDTH - 1 downto 0);
    signal valid    : std_logic;
begin
    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.neuromorphic_sca_top
        port map (
            clk      => clk,
            rst      => rst,
            start    => start,
            hw_class => hw_class,
            valid    => valid
        );

    process
    begin
        rst <= '1';
        wait for 5 * CLK_PERIOD;
        rst <= '0';
        wait for 2 * CLK_PERIOD;

        start <= '1';
        wait for CLK_PERIOD;
        start <= '0';

        wait until valid = '1';
        report "Classificacao concluida. HW class = " & integer'image(to_integer(unsigned(hw_class)));

        wait for 10 * CLK_PERIOD;
        assert false report "Fim da simulacao" severity failure;
    end process;
end architecture;
