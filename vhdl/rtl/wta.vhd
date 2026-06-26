library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

entity wta is
    port (
        counts_in : in  class_count_array_t(0 to NUM_CLASSES - 1);
        hw_class  : out class_t
    );
end entity;

architecture combinational of wta is
begin
    process(counts_in)
        variable best_idx   : natural range 0 to NUM_CLASSES - 1;
        variable best_count : count_t;
    begin
        best_idx := 0;
        best_count := counts_in(0);

        for i in 1 to NUM_CLASSES - 1 loop
            if counts_in(i) > best_count then
                best_count := counts_in(i);
                best_idx := i;
            end if;
        end loop;

        hw_class <= to_unsigned(best_idx, CLASS_WIDTH);
    end process;
end architecture;
