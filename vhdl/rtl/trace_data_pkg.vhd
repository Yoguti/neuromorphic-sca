library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

package trace_data_pkg is
    ---------------------------------------------------------------------------
    -- Uma trace do modulated_dataset/aes_hd_snn_ready.h5 convertida para Q8.8.
    --
    -- Este arquivo deve ser gerado por:
    --   python3 tools/h5_to_trace_data_pkg.py \
    --     --h5 ../modulated_dataset/aes_hd_snn_ready.h5 \
    --     --group Profiling_traces \
    --     --index 0 \
    --     --out rtl/trace_data_pkg.vhd
    --
    -- Os zeros abaixo sao apenas fallback seguro para o projeto compilar quando
    -- o arquivo HDF5 ainda nao foi materialmente extraido no ambiente local.
    ---------------------------------------------------------------------------
    constant TRACE_EXPECTED_LABEL : integer := -1;
    constant TRACE_FEATURES : feature_array_t(0 to NUM_POIS - 1) := (
         0 => to_signed(0, FEATURE_WIDTH),
         1 => to_signed(0, FEATURE_WIDTH),
         2 => to_signed(0, FEATURE_WIDTH),
         3 => to_signed(0, FEATURE_WIDTH),
         4 => to_signed(0, FEATURE_WIDTH),
         5 => to_signed(0, FEATURE_WIDTH),
         6 => to_signed(0, FEATURE_WIDTH),
         7 => to_signed(0, FEATURE_WIDTH),
         8 => to_signed(0, FEATURE_WIDTH),
         9 => to_signed(0, FEATURE_WIDTH),
        10 => to_signed(0, FEATURE_WIDTH),
        11 => to_signed(0, FEATURE_WIDTH),
        12 => to_signed(0, FEATURE_WIDTH),
        13 => to_signed(0, FEATURE_WIDTH),
        14 => to_signed(0, FEATURE_WIDTH),
        15 => to_signed(0, FEATURE_WIDTH),
        16 => to_signed(0, FEATURE_WIDTH),
        17 => to_signed(0, FEATURE_WIDTH),
        18 => to_signed(0, FEATURE_WIDTH),
        19 => to_signed(0, FEATURE_WIDTH)
    );
end package;
