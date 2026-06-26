# Arquitetura revisada conforme o repo

O repositório `Yoguti/neuromorphic-sca` treina/exporta uma SNN para classificar o Hamming Distance de AES em **9 classes HD 0..8**. A entrada final da SNN não é a trace crua inteira: o pré-processamento seleciona os **20 POIs** e grava as features no HDF5 `modulated_dataset/aes_hd_snn_ready.h5`.

## Fluxo de dados final

```text
HDF5 /Profiling_traces/traces[index] ou /Attack_traces/traces[index]
        ↓ float32[20]
h5_to_trace_data_pkg.py
        ↓ Q8.8 signed[20]
trace_data_pkg.vhd
        ↓
trace_rom.vhd
        ↓
input_buffer.vhd
        ↓
snn_core.vhd
   ├── synapse_accumulator.vhd
   └── neuron_bank.vhd
          └── lif_neuron.vhd
        ↓
output_register.vhd
        ↓
wta.vhd
        ↓
hw_class
```

## Separação pedida: neurônios e sinapses

### Sinapses

Arquivos:

```text
rtl/synapse_params_pkg.vhd
rtl/synapse_accumulator.vhd
```

Responsabilidade:

```text
Para cada sinapse s:
  origem = SYN_SRC(s)
  destino = SYN_TGT(s)
  peso = SYN_W(s)

Se origem é input 0..19:
  drive = trunc_toward_zero(peso * feature_q8_8 / 256)

Se origem é neurônio LIF 20..32:
  drive = peso quando o neurônio de origem disparou; caso contrário 0

Acumula drive no neurônio destino.
```

### Neurônios

Arquivos:

```text
rtl/neuron_params_pkg.vhd
rtl/neuron_bank.vhd
rtl/lif_neuron.vhd
```

Responsabilidade:

```text
membrane = membrane + corrente
aplica leak por shift aritmético
se membrane >= threshold:
    fired = 1
    membrane = reset
    entra em refractory
senão:
    fired = 0
```

## Rede real importada

| Item | Valor |
|---|---:|
| Input nodes | 0..19 |
| Output nodes | 20..28 |
| Hidden nodes | 29..32 |
| LIF internos | 13 |
| Sinapses | 150 |
| Ticks | 100 |

## Quantização

O simulador C usa `float` nas entradas. O VHDL usa Q8.8:

```text
feature_q = round(feature_float * 256)
drive = trunc_toward_zero(weight * feature_q / 256)
```

A função `trunc_div_pow2_i` em `snn_pkg.vhd` evita diferença de sinal em divisões negativas.

O leak do LIF usa `arith_shift_right_i`, porque o C/Python de referência usa shift aritmético para `delta >> leak_factor`.
