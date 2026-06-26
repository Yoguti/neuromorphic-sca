# Como usar no Quartus

## 1. Gerar a trace real do HDF5

No diretório do projeto VHDL:

```bash
python3 tools/h5_to_trace_data_pkg.py \
  --h5 ../modulated_dataset/aes_hd_snn_ready.h5 \
  --group Profiling_traces \
  --index 0 \
  --out rtl/trace_data_pkg.vhd
```

Para traces de ataque:

```bash
python3 tools/h5_to_trace_data_pkg.py \
  --h5 ../modulated_dataset/aes_hd_snn_ready.h5 \
  --group Attack_traces \
  --index 0 \
  --out rtl/trace_data_pkg.vhd
```

## 2. Validar os parâmetros importados

```bash
python3 tools/verify_against_repo_csv.py
```

Saída esperada:

```text
OK: neuron_params_pkg.vhd confere com best_network_neurons.csv
OK: synapse_params_pkg.vhd confere com best_network_synapses.csv
```

## 3. Criar projeto Quartus

Defina o top-level como:

```text
neuromorphic_sca_top
```

## 4. Adicionar arquivos

Use a ordem em:

```text
quartus_file_order.tcl
```

## 5. Sinais principais

```vhdl
clk      : clock
rst      : reset ativo em '1'
start    : inicia inferência
valid    : resultado pronto
hw_class : classe HD prevista, 0..8
```

## 6. Tempo de execução

A FSM faz:

1. limpa estado interno;
2. carrega os 20 POIs da ROM para o `input_buffer`;
3. roda a SNN por 100 ticks;
4. conta spikes das 9 saídas;
5. trava os contadores;
6. aplica WTA e levanta `valid`.

## Observação

Sem gerar `trace_data_pkg.vhd` a partir do `.h5`, o projeto compila com uma trace zerada de fallback, mas o resultado não representa uma classificação real.
