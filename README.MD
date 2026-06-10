# Embedded Spiking Neural Network for Side-Channel Analysis

An SNN modeled using Evolutionary Optimization for Neuromorphic Systems (EONS) the ASCAD dataset for training.

This project explores the use of event-driven Spiking Neural Networks (SNNs) for side-channel analysis of AES power traces. The objective is to train an optimized neuromorphic model capable of identifying cryptographic operations from power consumption patterns, enabling future reconstruction or inference of encryption-related information from side-channel leakage. The project also focuses on developing a hardware-efficient SNN architecture suitable for embedded implementation in VHDL and deployment on FPGA platforms.

The planned workflow is:

1. Convert AES power traces into spike  and generate the Hamming Weights.
2. Use EONS to evolve network topologies and parameters.
3. Evaluate candidate networks using the simulation engine.
4. Generate a hardware description of the best network for FPGA implementation.

## Dataset

Power consumption traces collected from an 8-bit ATmega8515 microcontroller while executing the Advanced Encryption Standard (AES-128) algorithm alongside first-order Boolean masking. The selected data uses a single fixed secret key through all traces.

![Project Image](https://raw.githubusercontent.com/Yoguti/neuromorphic-sca/master/images/1.png)

## Preparation

Power traces were converted into event-driven spike trains using an asynchronous delta modulation (ADM) encoding scheme similar to the method described in *An Asynchronous Delta Modulator for Spike Encoding in Event-Driven Brain-Machine Interface*. Positive and negative spikes were generated whenever the difference between the input signal and an adaptive reference exceeded a predefined threshold.

![Project Image](https://raw.githubusercontent.com/Yoguti/neuromorphic-sca/master/images/2.png)

## Environment

The Python environment for data preparation is stored in `data-prep/.venv/`, which is excluded from version control.

To recreate the environment:

```bash
cd data-prep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Engine
### Leaky Integrate-and-Fire Neurons

The engine uses the Leaky Integrate-and-Fire (LIF) neuron model.

A LIF neuron accumulates incoming spikes in its membrane potential. Over time, this potential naturally decays toward a resting value (the leak effect). When the membrane potential exceeds a threshold, the neuron emits a spike and resets its state.

The neuron state is represented by the following structure:

```c
typedef struct {
    int16_t membrane_potential;
    int16_t threshold;

    int8_t leak_factor;

    int16_t reset_potential;
    int16_t resting_potential;

    uint8_t has_fired;

    uint8_t refractory_period;
    uint8_t refractory_counter;
} lif_neuron_t;
```

### Synapses

Communication between neurons is performed through weighted synapses.

```c
typedef struct {
    uint16_t source_node;
    uint16_t target_node;
    int8_t weight;
} synapse_t;
```
Each synapse connects a source neuron to a target neuron and determines how strongly a spike influences the target neuron's membrane potential.
```c
if (source_fired) {
    uint16_t tgt_idx = SNN_NODE_TO_LIF(tgt);
    net->accumulated_inputs[tgt_idx] += syn->weight * source_fired;
}
```1

### Simulation Step

Network execution is performed in discrete time steps.

For each simulation tick:

1. Input spikes are propagated through active synapses.
2. Synaptic weights are accumulated at the target neurons.
3. Each neuron updates its membrane potential.
4. Neurons that reach the firing threshold emit a spike.
5. Temporary accumulators are cleared for the next tick.