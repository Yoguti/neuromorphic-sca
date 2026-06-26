library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity controller_fsm is
    port (
        clk           : in  std_logic;
        rst           : in  std_logic;
        start         : in  std_logic;
        trace_done    : in  std_logic;
        inference_done: in  std_logic;

        addr_count_en : out std_logic;
        spike_count_en: out std_logic;
        tick_count_en : out std_logic;
        rd_en         : out std_logic;
        load          : out std_logic;
        enable        : out std_logic;
        clear         : out std_logic;
        tick          : out std_logic;
        latch_out     : out std_logic;
        valid         : out std_logic
    );
end entity;

architecture rtl of controller_fsm is
    type state_t is (IDLE, INIT, READ_SAMPLE, CAPTURE_SAMPLE, NEXT_SAMPLE,
                     RUN_TICK, COUNT_SPIKES, LATCH_RESULT, DONE);
    signal state, next_state : state_t := IDLE;
begin
    process(clk, rst)
    begin
        if rst = '1' then
            state <= IDLE;
        elsif rising_edge(clk) then
            state <= next_state;
        end if;
    end process;

    process(state, start, trace_done, inference_done)
    begin
        next_state <= state;

        addr_count_en  <= '0';
        spike_count_en <= '0';
        tick_count_en  <= '0';
        rd_en          <= '0';
        load           <= '0';
        enable         <= '0';
        clear          <= '0';
        tick           <= '0';
        latch_out      <= '0';
        valid          <= '0';

        case state is
            when IDLE =>
                if start = '1' then
                    next_state <= INIT;
                end if;

            when INIT =>
                clear <= '1';
                next_state <= READ_SAMPLE;

            -- Leitura sincronizada da ROM: rd_en em um ciclo, captura no próximo.
            when READ_SAMPLE =>
                rd_en <= '1';
                next_state <= CAPTURE_SAMPLE;

            when CAPTURE_SAMPLE =>
                load <= '1';
                next_state <= NEXT_SAMPLE;

            when NEXT_SAMPLE =>
                if trace_done = '1' then
                    next_state <= RUN_TICK;
                else
                    addr_count_en <= '1';
                    next_state <= READ_SAMPLE;
                end if;

            -- Cada tick replica um passo snn_tick() do simulador C.
            when RUN_TICK =>
                enable <= '1';
                tick   <= '1';
                next_state <= COUNT_SPIKES;

            when COUNT_SPIKES =>
                spike_count_en <= '1';
                tick_count_en  <= '1';
                if inference_done = '1' then
                    next_state <= LATCH_RESULT;
                else
                    next_state <= RUN_TICK;
                end if;

            when LATCH_RESULT =>
                latch_out <= '1';
                next_state <= DONE;

            when DONE =>
                valid <= '1';
                if start = '0' then
                    next_state <= IDLE;
                end if;
        end case;
    end process;
end architecture;
