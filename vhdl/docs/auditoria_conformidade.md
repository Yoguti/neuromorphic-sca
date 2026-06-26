# Auditoria de conformidade

## Referências conferidas no repo

- `README.MD`: define o objetivo da SNN para SCA/AES, 9 classes HD e fluxo com exportação para HDL.
- `snn/network/network.h`: define `SNN_NUM_POIS = 20`, `SNN_NUM_INPUTS = 20`, `SNN_NUM_OUTPUTS = 9`.
- `snn/EONS/engine.c`: define `INFERENCE_TICKS = 100` no código atual.
- `snn/libs/dataset.c`: carrega `traces` e `labels_hw` de `Profiling_traces` ou `Attack_traces`.
- `data-prep/sca.py`: confirma `NUM_INPUTS = 20`, `NUM_OUTPUTS = 9`, `TICKS = 100`, além dos caminhos de HDF5.
- `snn/network-csvs/best_network_neurons.csv`: fonte dos parâmetros LIF.
- `snn/network-csvs/best_network_synapses.csv`: fonte das 150 sinapses.

## Ajustes aplicados

1. Removido o fluxo antigo `700x8` como entrada principal.
2. Mantida uma ROM de 20 features Q8.8, coerente com o HDF5 pré-processado.
3. Separadas sinapses e neurônios em arquivos VHDL próprios.
4. Importados 13 neurônios LIF reais.
5. Importadas 150 sinapses reais.
6. Ajustado o número de ticks para 100.
7. Adicionada função de divisão truncada para o cálculo de drive fixed-point.
8. Adicionada função de shift aritmético para o leak LIF, evitando divergência em valores negativos.
9. Adicionado script de verificação contra os CSVs reais.
10. Atualizados README e docs para remover contradições com o repo.

## Limitação honesta

O arquivo `.h5` é binário. O pacote inclui o conversor `tools/h5_to_trace_data_pkg.py`; a ROM real deve ser gerada no ambiente local onde o `.h5` estiver disponível. O VHDL incluído no ZIP usa uma trace zerada de fallback apenas para compilação estrutural.
