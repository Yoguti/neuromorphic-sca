# Neuromorphic SCA VHDL — conforme ao repo `Yoguti/neuromorphic-sca`

Esta entrega é a versão revisada para ficar coerente com o **README**, o **código C/Python atual**, os **CSVs reais da rede** e o **`modulated_dataset`** do repositório.

## Decisão técnica principal

O fluxo antigo de diagrama com memória `700x8` foi substituído, nesta versão final, pelo fluxo real do repositório:

```text
modulated_dataset/aes_hd_snn_ready.h5
  /Profiling_traces/traces[index] ou /Attack_traces/traces[index]
        ↓ float32[20]
conversor Q8.8
        ↓ signed[20]
trace_data_pkg.vhd
        ↓
trace_rom.vhd
        ↓
input_buffer.vhd
        ↓
synapse_accumulator.vhd
        ↓
neuron_bank.vhd
        ↓
output_register.vhd
        ↓
wta.vhd
        ↓
hw_class 0..8
```

O `modulated_dataset` já contém as **20 features/POIs pré-processadas**. Portanto, para ficar fiel ao repo, o VHDL não deve consumir a trace crua de 700 amostras diretamente.

## Parâmetros confirmados

| Item | Valor usado no VHDL |
|---|---:|
| Entradas / POIs | 20 |
| Classes de saída | 9, HD 0..8 |
| Neurônios LIF de saída | 9 |
| Neurônios hidden | 4 |
| Total de neurônios LIF | 13 |
| Sinapses reais | 150 |
| Ticks de inferência | 100 |
| Formato de entrada | Q8.8 signed, 16 bits |
| Contadores de saída | 9 x 16 bits |

## Arquitetura dos arquivos

```text
rtl/
  snn_pkg.vhd                # tipos, constantes e funções auxiliares
  network_params_pkg.vhd     # tamanhos da rede real
  neuron_params_pkg.vhd      # parâmetros reais dos 13 neurônios LIF
  synapse_params_pkg.vhd     # 150 sinapses reais: src, tgt, weight
  trace_data_pkg.vhd         # uma trace HDF5 convertida para Q8.8
  trace_rom.vhd              # ROM de 20 POIs
  addr_counter.vhd           # contador 0..19
  input_buffer.vhd           # carrega os 20 POIs em paralelo para a rede
  lif_neuron.vhd             # neurônio LIF individual
  synapse_accumulator.vhd    # bloco separado de sinapses
  neuron_bank.vhd            # bloco separado de neurônios
  snn_core.vhd               # conecta sinapses + neurônios
  inference_counter.vhd      # conta 100 ticks de inferência
  output_register.vhd        # conta spikes das 9 classes
  wta.vhd                    # winner-takes-all
  controller_fsm.vhd         # bloco de controle
  datapath.vhd               # bloco operacional
  neuromorphic_sca_top.vhd   # top-level
```

## Como gerar a ROM real a partir do HDF5

No diretório do projeto VHDL, com o arquivo `.h5` disponível localmente:

```bash
python3 tools/h5_to_trace_data_pkg.py \
  --h5 ../modulated_dataset/aes_hd_snn_ready.h5 \
  --group Profiling_traces \
  --index 0 \
  --out rtl/trace_data_pkg.vhd
```

Para usar uma trace de ataque:

```bash
python3 tools/h5_to_trace_data_pkg.py \
  --h5 ../modulated_dataset/aes_hd_snn_ready.h5 \
  --group Attack_traces \
  --index 0 \
  --out rtl/trace_data_pkg.vhd
```

O arquivo `trace_data_pkg.vhd` incluído no ZIP vem com zeros apenas para permitir compilação estrutural sem o HDF5 no ambiente local. Para obter uma classificação real, gere esse arquivo com o comando acima.

## Como validar contra os CSVs reais

Rode:

```bash
python3 tools/verify_against_repo_csv.py
```

Esse script confere se os packages VHDL batem com:

```text
data/best_network_neurons.csv
data/best_network_synapses.csv
```

## Ordem de compilação

Use:

```text
quartus_file_order.tcl
```

Top-level:

```text
neuromorphic_sca_top
```

## Entradas e saídas do top-level

```vhdl
clk      : in  std_logic;
rst      : in  std_logic; -- ativo em '1'
start    : in  std_logic;
hw_class : out std_logic_vector(3 downto 0); -- 0..8
valid    : out std_logic;
```

## Observação de conformidade

O README do repo ainda contém um trecho textual antigo citando uma janela fixa de 20 ticks, mas o código atual usado como referência (`engine.c` e `data-prep/sca.py`) define **100 ticks**. Esta entrega segue o código executável atual.
