# Ataque neuromórfico de canal lateral

## Grupo: AP3-02208B-Grupo-E
## Integrantes
- Anderson Porfírio (25250364)
- Juan Pablo Lopes (25105624)
- Klaus Schneider (23203332)
- Luiz Fernando Montalvan de Sousa (24202598)

# Descrição do Projeto
Descrição de uma SNN (Spiking Neural Network) para side-channel analysis em HDL. O objetivo é projetar uma rede capaz de reconhecer operações criptográficas a partir de padrões de consumo de energia emitidos por um microcontrolador (ATMEGA) executando um algoritmo de criptografia (AES), utilizando uma rede concisa e otimizada através de um algoritmo evolutivo (EONS). Idealmente, a rede final será um balanço ideal entre consumo de energia, uso de recursos de hardware e acurácia.

## Atualizações desta Entrega
Nesta fase, avançamos da definição teórica para a estrutura prática do sistema:

* **Conversão de Dados:** Implementamos a técnica de modulação delta assíncrona (ADM) para transformar os traços de energia brutos em *spike trains* utilizáveis pela rede.
* **Simulação:** Criamos o motor baseado em neurônios *Leaky Integrate-and-Fire* (LIF) e sinapses ponderadas, estruturando a lógica em C para suportar a futura transição para hardware.





## Nota de conformidade com o repositório revisado

Após revisar o repositório `Yoguti/neuromorphic-sca`, a implementação VHDL final desta entrega usa o `modulated_dataset/aes_hd_snn_ready.h5`, que já contém os 20 POIs normalizados por trace. Assim, a memória principal do datapath final armazena 20 features Q8.8 por inferência, e não uma trace crua completa de 700 amostras.

A separação final é:

* `synapse_accumulator.vhd`: sinapses reais exportadas em CSV.
* `neuron_bank.vhd` + `lif_neuron.vhd`: neurônios LIF reais exportados em CSV.
* `output_register.vhd`: contadores 9x16.
* `wta.vhd`: seleção da classe HD vencedora.
