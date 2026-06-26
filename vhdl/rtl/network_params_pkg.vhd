library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

package network_params_pkg is
    ---------------------------------------------------------------------------
    -- Parametros reais do best_network exportado do repo.
    -- Node IDs originais do C:
    --   entradas/POIs: 0..19
    --   saidas HD:     20..28
    --   hidden:        29..32
    ---------------------------------------------------------------------------
    constant NET_NUM_INPUTS  : natural := NUM_POIS;     -- 20
    constant NET_NUM_OUTPUTS : natural := NUM_CLASSES;  -- 9
    constant NET_NUM_HIDDEN  : natural := 4;
    constant NET_NUM_LIF     : natural := NET_NUM_OUTPUTS + NET_NUM_HIDDEN; -- 13

    constant NET_FIRST_INPUT_NODE : integer := 0;
    constant NET_FIRST_LIF_NODE   : integer := integer(NET_NUM_INPUTS); -- 20
end package;
