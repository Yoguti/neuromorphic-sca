#!/usr/bin/env python3
"""
Verifica se os packages VHDL gerados batem com os CSVs reais exportados
pelo repo Yoguti/neuromorphic-sca.

Arquivos esperados:
  data/best_network_neurons.csv
  data/best_network_synapses.csv
  rtl/neuron_params_pkg.vhd
  rtl/synapse_params_pkg.vhd
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv_ints(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_vhdl_array(text: str, name: str) -> list[int]:
    m = re.search(rf"constant\s+{re.escape(name)}\s*:\s*[^:]+:=\s*\((.*?)\)\s*;", text, re.S | re.I)
    if not m:
        raise ValueError(f"Array VHDL nao encontrado: {name}")
    body = re.sub(r"--.*", "", m.group(1))
    nums = re.findall(r"[-+]?\d+", body)
    return [int(x) for x in nums]


def main() -> None:
    neu_csv = ROOT / "data" / "best_network_neurons.csv"
    syn_csv = ROOT / "data" / "best_network_synapses.csv"
    neu_vhd = ROOT / "rtl" / "neuron_params_pkg.vhd"
    syn_vhd = ROOT / "rtl" / "synapse_params_pkg.vhd"

    neurons = read_csv_ints(neu_csv)
    synapses = read_csv_ints(syn_csv)

    neurons = sorted(neurons, key=lambda r: int(r["node_id"]))
    expected_threshold = [int(r["threshold"]) for r in neurons]
    expected_leak = [int(r["leak_factor"]) for r in neurons]
    expected_reset = [int(r["reset_potential"]) for r in neurons]
    expected_resting = [int(r["resting_potential"]) for r in neurons]
    expected_refrac = [int(r["refractory_period"]) for r in neurons]

    neu_text = neu_vhd.read_text(encoding="utf-8")
    checks = {
        "NEURON_THRESHOLD": expected_threshold,
        "NEURON_LEAK_FACTOR": expected_leak,
        "NEURON_RESET_POTENTIAL": expected_reset,
        "NEURON_RESTING_POTENTIAL": expected_resting,
        "NEURON_REFRACTORY_PERIOD": expected_refrac,
    }
    for name, expected in checks.items():
        got = parse_vhdl_array(neu_text, name)
        if got != expected:
            raise AssertionError(f"Mismatch em {name}:\n got={got}\n exp={expected}")

    syn_text = syn_vhd.read_text(encoding="utf-8")
    expected_src = [int(r["source_node"]) for r in synapses]
    expected_tgt = [int(r["target_node"]) for r in synapses]
    expected_w = [int(r["weight"]) for r in synapses]

    for name, expected in {
        "SYN_SRC": expected_src,
        "SYN_TGT": expected_tgt,
        "SYN_W": expected_w,
    }.items():
        got = parse_vhdl_array(syn_text, name)
        if got != expected:
            raise AssertionError(f"Mismatch em {name}:\n got={got}\n exp={expected}")

    if len(neurons) != 13:
        raise AssertionError(f"Esperado 13 neuronios LIF, encontrado {len(neurons)}")
    if len(synapses) != 150:
        raise AssertionError(f"Esperado 150 sinapses, encontrado {len(synapses)}")

    print("OK: neuron_params_pkg.vhd confere com best_network_neurons.csv")
    print("OK: synapse_params_pkg.vhd confere com best_network_synapses.csv")
    print(f"OK: {len(neurons)} neuronios LIF e {len(synapses)} sinapses")


if __name__ == "__main__":
    main()
