# Embedded Spiking Neural Network for Side-Channel Analysis

An SNN modeled using Evolutionary Optimization for Neuromorphic Systems (EONS) the ASCAD dataset for training.

This project explores the use of event-driven Spiking Neural Networks (SNNs) for side-channel analysis of AES power traces. The objective is to train an optimized neuromorphic model capable of identifying cryptographic operations from power consumption patterns, enabling future reconstruction or inference of encryption-related information from side-channel leakage. The project also focuses on developing a hardware-efficient SNN architecture suitable for embedded implementation in VHDL and deployment on FPGA platforms.
## Dataset

Power consumption traces collected from an 8-bit ATmega8515 microcontroller while executing the Advanced Encryption Standard (AES-128) algorithm alongside first-order Boolean masking. The selected data uses a single fixed secret key through all traces.

## Preparation

Power traces were converted into event-driven spike trains using an asynchronous delta modulation (ADM) encoding scheme similar to the method described in *An Asynchronous Delta Modulator for Spike Encoding in Event-Driven Brain-Machine Interface*. Positive and negative spikes were generated whenever the difference between the input signal and an adaptive reference exceeded a predefined threshold.

## Environment

The Python environment for data preparation is stored in `data-prep/.venv/`, which is excluded from version control.

To recreate the environment:

```bash
cd data-prep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
