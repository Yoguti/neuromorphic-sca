# Mapeamento entre diagramas e repo

Os diagramas originais descrevem a separação didática em:

- BC: Bloco de Controle;
- BO: Bloco Operacional;
- memória;
- registrador de amostra;
- SNN core;
- registrador de saída 9x16;
- WTA.

Após revisar o repo, o mapeamento conforme fica:

| Diagrama original | Implementação final conforme repo |
|---|---|
| Memória 700x8 | `trace_rom.vhd` com 20 POIs Q8.8 extraídos do HDF5 |
| Reg. amostra | `input_buffer.vhd`, carregando 20 features antes da inferência |
| SNN Core | `snn_core.vhd` |
| Sinapses internas | `synapse_accumulator.vhd` + `synapse_params_pkg.vhd` |
| Neurônios internos | `neuron_bank.vhd` + `lif_neuron.vhd` + `neuron_params_pkg.vhd` |
| Reg. saída 9x16 | `output_register.vhd` |
| WTA | `wta.vhd` |
| BC/FSM | `controller_fsm.vhd` |
| BO/Datapath | `datapath.vhd` |

## Por que não usar 700x8 como entrada final?

Porque o repo já gera o dataset pronto da SNN como `float32 [N,20]`. A etapa de seleção de POIs e normalização acontece antes da inferência. Logo, usar 700 amostras diretamente no VHDL deixaria o hardware desalinhado com a rede treinada e exportada.
