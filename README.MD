# Embedded Spiking Neural Network for Side-Channel Analysis

An SNN modeled using Evolutionary Optimization for Neuromorphic Systems (EONS) and trained on the AES_HD dataset.

This project explores the use of event-driven Spiking Neural Networks (SNNs) for Side-Channel Analysis (SCA) of AES power traces. The objective is to train an optimized neuromorphic model capable of classifying side-channel leakage to recover secret cryptographic keys.

**HW Classes:**
During cryptographic operations, hardware implementations leak electromagnetic radiation that correlates with the number of active bit transitions being processed. For AES hardware implementations, the most exploitable leakage is the **Hamming Distance (HD)** between consecutive register states during the last round. Since AES processes 8-bit bytes, there are **9 possible Hamming Distance classes (HD 0 through HD 8)**.

The SNN acts as a classifier. It takes a vector of power amplitude features as input and predicts which of the 9 HD classes the targeted AES operation belongs to. By correctly predicting the Hamming Distance of the inverse S-Box operation on the ciphertext, the network effectively discovers the secret key.

The project also focuses on developing a hardware-efficient SNN architecture suitable for embedded implementation in VHDL and deployment on FPGA platforms.

## Workflow

1. Extract high-SNR POIs from traces, label by HD.
2. Z-score normalize amplitudes.
3. Evolve SNN with EONS.
4. Evaluate in C simulator.
5. Export to HDL.

## Dataset

AES_HD: 100k EM traces from an unprotected AES-128 core on a Xilinx Virtex-5 FPGA (SASEBO GII). Traces captured by a probe measuring the electromagnetic radiation of a capacitor on the power supply line.

Leakage target (last-round register write):

```
Y(C_i1, C_i2, k*) = Sbox_inv[C_i1 ⊕ k*] ⊕ C_i2
```

Where C_i1 and C_i2 are ciphertext bytes linked by inverse ShiftRows (i1=12, i2=8). No masking, clean signal.

Available: [https://github.com/AESHD/AES_HD_Dataset](https://github.com/AESHD/AES_HD_Dataset)

## Preparation

**SNR-based POI selection + z-score standardization.**

1. Compute SNR per time index across all 9 HD classes.
2. Take top-20 POIs (reduces 1250 → 20 features).
3. Z-score normalize each column independently (stats from profiling set only).
4. Undersample profiling set so all 9 classes have equal counts.
5. Export to HDF5: float32 features + uint8 labels.

Output: 20-dim float32 vector (z-scored power amplitudes).

## Environment

Setup for generating the modulated dataset and running python testbenches 

```bash
cd data-prep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

## Engine

## Engine: LIF Neurons

The engine uses the Leaky Integrate-and-Fire (LIF) neuron model.

A LIF neuron accumulates incoming weighted inputs in its membrane potential. Over time, this potential naturally decays toward a resting value (the leak effect). When the membrane potential exceeds a threshold, the neuron emits a spike and resets its state.

```c
typedef struct {
    int16_t membrane_potential;
    int16_t threshold;
    int8_t  leak_factor;
    int16_t reset_potential;
    int16_t resting_potential;
    uint8_t has_fired;
    uint8_t refractory_period;
    uint8_t refractory_counter;
} lif_neuron_t;
```

## Synapses

The z-scored amplitude value at each POI is multiplied by the synaptic weight, giving graded excitation proportional to the leakage magnitude. Hidden and output neuron spikes deliver binary drive as in standard SNNs.

```c
typedef struct {
    uint16_t source_node;
    uint16_t target_node;
    int8_t   weight;
} synapse_t;
```

## Input Encoding

Spatial encoding: each of 20 POIs maps to a dedicated input neuron. Feature identity is determined by *which* neuron receives the signal

Synaptic drive from inputs:
```c
drive = (int32_t)((float)syn->weight * net->input_values[src]);
```

Each trace is presented to the network for a fixed window of 20 inference ticks. The input is re-asserted every tick, allowing the recurrent and output dynamics time to settle and accumulate spike counts in the readout layer. 


## Simulation Step

Each tick:
1. Load z-scored POI amplitudes to input neurons.
2. Compute input drive: weight × input_value (float).
3. Compute hidden/output drive: weight × has_fired (binary).
4. Add accumulated drives to membrane potentials.
5. Fire neurons at threshold, enter refractory.
6. Clear accumulators.

## Fitness Evaluation
as per *EONS*: 

```c
float accuracy     = (float)correct / (float)batch;
float size_penalty = (float)(net->num_synapses + SNN_LIF_COUNT(net)) * alpha;
pop[i].fitness_score = accuracy - size_penalty;
```

Evaluate on 2k-trace minibatches. Random guessing baseline: 1/9 ≈ 0.111.

### Breaking AES
The evolved SNN maps z-scored power amplitude features to the correct Hamming Distance class of the targeted last-round register write.

During the last round of AES, the chip performs the inverse S-Box substitution:
`Intermediate_State = SBox_inv[Ciphertext_byte ⊕ Secret_Key_byte]`

The SNN analyzes the power trace at the top-SNR time samples and predicts the Hamming Distance between consecutive register states.

By knowing the Ciphertext and the predicted HD of the register transition, it becomes possible to enumerate all 256 hypotheses for a single key byte, compute the expected HD for each hypothesis, and compare it against the SNN's prediction. The hypothesis that consistently aligns with the SNN's output across many traces reveals the correct key byte. Repeating this process for all 16 bytes recovers the entire 128-bit AES master key.

---

## References and Inspirations

**Evolutionary Optimization for Neuromorphic Systems (EONS)**
Catherine D. Schuman, J. Parker Mitchell, Robert M. Patton, Thomas E. Potok, James S. Plank.
*Evolutionary Optimization for Neuromorphic Systems.*
NICE '20, Heidelberg, Germany, 2020.
[https://doi.org/10.1145/3381755.3381758](https://doi.org/10.1145/3381755.3381758)

The genetic algorithm in this engine—including the structural mutation operators, tournament selection, and the size-penalized fitness function—is heavily based on the EONS framework described here.

**Make Some Noise: Unleashing the Power of Convolutional Neural Networks for Profiled Side-Channel Analysis**
Jaehun Kim, Stjepan Picek, Annelie Heuser, Shivam Bhasin, Alan Hanjalic.
*IACR Transactions on Cryptographic Hardware and Embedded Systems*, 2019(3), 148–179.
[https://doi.org/10.13154/tches.v2019.i3.148-179](https://doi.org/10.13154/tches.v2019.i3.148-179)

This paper established AES_HD as a standard benchmark and introduced the CNN architecture typically used to attack it. The dataset parsing and leakage models used in this project follow their conventions.
