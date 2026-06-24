Here is a revised version of the README. I kept the technical structure and methodology exactly the same, but loosened up the phrasing, varied the sentence lengths, and removed some of the overly "perfect" transitional filler that gives AI-generated text away. It should sound much more like a developer documenting their own engine.

---

# Embedded Spiking Neural Network for Side-Channel Analysis

SNN trained on AES_HD using Evolutionary Optimization for Neuromorphic Systems (EONS).

## The Problem

Unprotected AES leaks through EM radiation during the final round. The Hamming Distance between consecutive register states (0–8) correlates with power consumption. Predict HD correctly, recover the key.

The network classifies 20 power features into 9 HD classes. Small and efficient enough for FPGA deployment.

## Workflow

1. Extract high-SNR POIs from traces, label by HD.
2. Z-score normalize amplitudes.
3. Evolve SNN with EONS.
4. Evaluate in C simulator.
5. Export to HDL.

## Dataset

AES_HD: 100k EM traces from an unprotected AES-128 core on a Xilinx Virtex-5 FPGA (SASEBO GII). Parallel architecture, 11 cycles per encryption. Near-field probe over decoupling cap on power line.

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

The Python environment for data prep is kept in `data-prep/.venv/` (excluded from git).

To spin it up:

```bash
cd data-prep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

## Engine

## Engine: LIF Neurons

Leaky Integrate-and-Fire. Membrane potential accumulates input, decays toward rest, fires at threshold, resets and refractory.

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

## Input Encoding

Spatial encoding: each of 20 POIs maps to a dedicated input neuron. Feature identity by position, not timing.

Synaptic drive from inputs:
```c
drive = (int32_t)((float)syn->weight * net->input_values[src]);
```

Amplitude-weighted multiplication. Hidden/output neurons deliver binary drive on spike.

Inference: present trace for 20 ticks with input held steady. Readout accumulates spike counts.

## Synapses

```c
typedef struct {
    uint16_t source_node;
    uint16_t target_node;
    int8_t   weight;
} synapse_t;
```

## Simulation Step

Each tick:
1. Load z-scored POI amplitudes to input neurons.
2. Compute input drive: weight × input_value (float).
3. Compute hidden/output drive: weight × has_fired (binary).
4. Add accumulated drives to membrane potentials.
5. Fire neurons at threshold, enter refractory.
6. Clear accumulators.

After 20 ticks: output class = neuron with most spikes.

## Memory Management

Double-buffered arena allocator. Generations swap between active/inactive buffers. Old population wiped in O(1) by resetting pointer. Avoids malloc/free overhead.

## Fitness Evaluation

```c
float accuracy     = (float)correct / (float)batch;
float size_penalty = (float)(net->num_synapses + SNN_LIF_COUNT(net)) * alpha;
pop[i].fitness_score = accuracy - size_penalty;
```

Evaluate on 2k-trace minibatches (not full 84k). Stochastic variance helps escape local optima. Random guessing baseline: 1/9 ≈ 0.111.

## Key Recovery

Last-round inverse S-Box:
```
Intermediate_State = SBox_inv[Ciphertext_byte ⊕ Secret_Key_byte]
```

Known ciphertext + predicted HD → enumerate 256 key byte hypotheses. Calculate expected HD per guess, compare to network output. Correct hypothesis matches across many traces.

16 bytes = 128-bit key.

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