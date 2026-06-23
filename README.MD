# Embedded Spiking Neural Network for Side-Channel Analysis

An SNN modeled using Evolutionary Optimization for Neuromorphic Systems (EONS) and trained on the ASCAD dataset.

This project explores the use of event-driven Spiking Neural Networks (SNNs) for Side-Channel Analysis (SCA) of AES power traces. The objective is to train an optimized neuromorphic model capable of classifying side-channel leakage to recover secret cryptographic keys. 

**HW Classes:**
During cryptographic operations, microcontrollers leak power that correlates with the number of active bits (`1`s) being processed, a metric known as the **Hamming Weight (HW)**. Since AES processes 8-bit bytes, there are **9 possible Hamming Weight classes (HW 0 through HW 8)**. 

The SNN acts as a classifier. It takes a sequence of power consumption spikes as input and predicts which of the 9 HW classes the hidden intermediate state belongs to. By correctly predicting the Hamming Weight of a targeted AES operation (specifically, the S-Box substitution of a known plaintext byte combined with a secret key byte), the network effectively discovers the secret key.

The project also focuses on developing a hardware-efficient SNN architecture suitable for embedded implementation in VHDL and deployment on FPGA platforms.

The planned workflow is:

1. Convert AES power traces into spikes and calculate the corresponding Hamming Weight labels.
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

![Project Image](https://raw.githubusercontent.com/Yoguti/neuromorphic-sca/master/images/3.png)

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
```

### Simulation Step

Network execution is performed in discrete time steps.

For each simulation tick:

1. Input spikes are propagated through active synapses.
2. Synaptic weights are accumulated at the target neurons.
3. Each neuron updates its membrane potential.
4. Neurons that reach the firing threshold emit a spike.
5. Temporary accumulators are cleared for the next tick.

### Memory Management (Double-Buffered Arena)

Given the high throughput required by the evolutionary algorithm, standard dynamic memory allocation (`malloc`/`free`) introduces unacceptable overhead and fragmentation. The engine implements a double-buffered arena allocator.

Generations alternate between an active and inactive arena, allowing entire populations of networks to be wiped from memory in `O(1)` time through a single pointer reset at the end of an epoch.

### Fitness Evaluation & Escaping Local Optima

The ASCAD dataset is statistically imbalanced, with the Hamming Weight 4 class representing approximately 28.7% of the traces. During evolution, networks tend to fall into a local optimum, minimizing topology (e.g., 1 synapse, 0 hidden neurons) and exclusively predicting the majority class to maximize accuracy while avoiding complexity penalties.

To escape this gravity well, the fitness evaluation implements a diversity stepping-stone mechanic. The evaluation metric provides a calculated numerical bonus for networks that predict distinct classes, creating a mathematical gradient that incentivizes the addition of synapses and neurons.

```c
// Fitness prioritizes accuracy but rewards networks that predict multiple classes
pop[i].fitness_score = accuracy + (unique_classes_predicted * 0.005f); 
```

### Classifying the Hamming Weight is enough to break AES

Through the EONS evolutionary process, the engine successfully produces highly optimized Spiking Neural Networks that map asynchronous UP and DOWN power spikes to the correct Hamming Weight class of the cryptographic operation.

![Evolved SNN Architecture](https://raw.githubusercontent.com/Yoguti/neuromorphic-sca/master/images/image_snn_network.png)

In Side-Channel Analysis (SCA), the goal is to **extract the fixed Master Key hidden inside the hardware**

During the first round of AES, the chip performs a highly nonlinear operation using a public look-up table called the S-Box:
`Intermediate_State = SBox(Plaintext ⊕ Secret_Key)`

While the chip calculates the `Intermediate_State`, it leaks power. The evolved SNN analyzes this power trace and accurately predicts the Hamming Weight.

By knowing the `Plaintext` and the Hamming Weight of the `Intermediate_State`, it becomes possible to guess all 256 probable values for a single byte of the key, calculate the expected Hamming Weight *would* be for each guess, and compares it against the SNN's real-world prediction. The key guess that perfectly aligns with the SNN's output reveals the secret key byte. By repeating this process for all 16 bytes, the entire 128-bit AES master key is completely recovered.