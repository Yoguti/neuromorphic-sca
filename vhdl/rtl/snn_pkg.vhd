library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package snn_pkg is
    ---------------------------------------------------------------------------
    -- Implementacao VHDL da SNN do repo Yoguti/neuromorphic-sca.
    --
    -- Fonte de entrada correta: modulated_dataset/aes_hd_snn_ready.h5.
    -- O HDF5 ja contem 20 POIs pre-processados e z-scoreados por trace.
    -- Portanto esta versao usa uma ROM de 20 features fixed-point Q8.8,
    -- nao a trace crua 700x8.
    ---------------------------------------------------------------------------
    constant NUM_POIS        : natural := 20;
    constant TRACE_LEN       : natural := NUM_POIS;
    constant ADDR_WIDTH      : natural := 5;  -- 2^5 = 32 >= 20

    constant NUM_CLASSES     : natural := 9;  -- HD 0..8
    constant CLASS_WIDTH     : natural := 4;  -- representa 0..8
    constant COUNT_WIDTH     : natural := 16; -- Reg. de Saida 9 x 16 bits

    -- O codigo atual do repo usa INFERENCE_TICKS = 100.
    constant INFERENCE_TICKS : natural := 100;
    constant TICK_WIDTH      : natural := 7;  -- 2^7 = 128 >= 100

    ---------------------------------------------------------------------------
    -- Fixed-point da entrada do HDF5
    -- O dataset guarda float32 z-scoreado. O conversor grava Q8.8:
    --   fixed = round(float_value * 2^FEATURE_FRAC_BITS)
    -- A sinapse calcula aproximadamente:
    --   drive = trunc_toward_zero(weight * fixed / 2^FEATURE_FRAC_BITS)
    ---------------------------------------------------------------------------
    constant FEATURE_WIDTH     : natural := 16;
    constant FEATURE_FRAC_BITS : natural := 8;

    ---------------------------------------------------------------------------
    -- Larguras internas da rede LIF
    ---------------------------------------------------------------------------
    constant POT_WIDTH       : natural := 16;
    constant CURRENT_WIDTH   : natural := 32;

    ---------------------------------------------------------------------------
    -- Parametros LIF padrao espelhados do simulador C
    ---------------------------------------------------------------------------
    constant LIF_DEFAULT_THRESHOLD         : integer := 20;
    constant LIF_DEFAULT_RESTING           : integer := 0;
    constant LIF_DEFAULT_RESET             : integer := -5;
    constant LIF_DEFAULT_LEAK_FACTOR       : natural := 3;
    constant LIF_DEFAULT_REFRACTORY_PERIOD : natural := 2;

    subtype feature_t is signed(FEATURE_WIDTH - 1 downto 0);
    subtype class_t   is unsigned(CLASS_WIDTH - 1 downto 0);
    subtype count_t   is unsigned(COUNT_WIDTH - 1 downto 0);
    subtype current_t is signed(CURRENT_WIDTH - 1 downto 0);
    subtype pot_t     is signed(POT_WIDTH - 1 downto 0);

    type feature_array_t     is array (natural range <>) of feature_t;
    type class_count_array_t is array (natural range <>) of count_t;
    type current_array_t     is array (natural range <>) of current_t;
    type int_array_t         is array (natural range <>) of integer;

    function clamp_i(x : integer; lo : integer; hi : integer) return integer;

    -- Equivalente pratico ao cast C (int32_t) de um valor escalado:
    -- divide por 2^shift truncando em direcao a zero, inclusive para negativos.
    function trunc_div_pow2_i(x : integer; shift : natural) return integer;

    -- Equivalente ao right shift aritmetico usado no simulador C/Python para leak.
    -- Para negativos, preserva sinal: -9 >> 3 = -2.
    function arith_shift_right_i(x : integer; shift : natural) return integer;
end package;

package body snn_pkg is
    function clamp_i(x : integer; lo : integer; hi : integer) return integer is
    begin
        if x < lo then
            return lo;
        elsif x > hi then
            return hi;
        else
            return x;
        end if;
    end function;

    function trunc_div_pow2_i(x : integer; shift : natural) return integer is
        variable denom : integer := 1;
    begin
        for i in 1 to shift loop
            denom := denom * 2;
        end loop;

        if x >= 0 then
            return x / denom;
        else
            return -((-x) / denom);
        end if;
    end function;

    function arith_shift_right_i(x : integer; shift : natural) return integer is
        variable v : signed(31 downto 0);
    begin
        v := to_signed(x, 32);
        return to_integer(shift_right(v, shift));
    end function;
end package body;
