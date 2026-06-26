#!/usr/bin/env python3
"""
Gera neuron_params_pkg.vhd e synapse_params_pkg.vhd a partir dos CSVs exportados
pelo repo Yoguti/neuromorphic-sca.

Uso:
  python3 tools/csv_to_vhdl_params.py \
    --neurons data/best_network_neurons.csv \
    --synapses data/best_network_synapses.csv \
    --out-dir rtl
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_array(values: list[int], indent: str = "        ", per_line: int = 15) -> str:
    chunks = []
    for i in range(0, len(values), per_line):
        part = values[i:i + per_line]
        comma = "," if i + per_line < len(values) else ""
        chunks.append(indent + ", ".join(str(v) for v in part) + comma)
    return "\n".join(chunks)


def emit_neurons(rows: list[dict[str, str]], out: Path) -> None:
    rows = sorted(rows, key=lambda r: int(r["node_id"]))
    n = len(rows)
    values = {
        "NEURON_THRESHOLD": [int(r["threshold"]) for r in rows],
        "NEURON_LEAK_FACTOR": [int(r["leak_factor"]) for r in rows],
        "NEURON_RESET_POTENTIAL": [int(r["reset_potential"]) for r in rows],
        "NEURON_RESTING_POTENTIAL": [int(r["resting_potential"]) for r in rows],
        "NEURON_REFRACTORY_PERIOD": [int(r["refractory_period"]) for r in rows],
    }
    blocks = []
    for name, arr in values.items():
        blocks.append(f"""    constant {name} : int_array_t(0 to NET_NUM_LIF - 1) := (\n{fmt_array(arr)}\n    );""")
    text = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

package neuron_params_pkg is
    ---------------------------------------------------------------------------
    -- Gerado automaticamente a partir do CSV de neuronios.
    -- Total esperado: {n} neuronios LIF.
    ---------------------------------------------------------------------------
{chr(10).join(blocks)}
end package;
"""
    out.write_text(text, encoding="utf-8")


def emit_synapses(rows: list[dict[str, str]], out: Path) -> None:
    src = [int(r["source_node"]) for r in rows]
    tgt = [int(r["target_node"]) for r in rows]
    w = [int(r["weight"]) for r in rows]
    n = len(rows)
    text = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;
use work.network_params_pkg.all;

package synapse_params_pkg is
    ---------------------------------------------------------------------------
    -- Gerado automaticamente a partir do CSV de sinapses.
    ---------------------------------------------------------------------------
    constant NET_NUM_SYNAPSES : natural := {n};

    constant SYN_SRC : int_array_t(0 to NET_NUM_SYNAPSES - 1) := (\n{fmt_array(src)}\n    );

    constant SYN_TGT : int_array_t(0 to NET_NUM_SYNAPSES - 1) := (\n{fmt_array(tgt)}\n    );

    constant SYN_W : int_array_t(0 to NET_NUM_SYNAPSES - 1) := (\n{fmt_array(w)}\n    );
end package;
"""
    out.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--synapses", required=True)
    ap.add_argument("--out-dir", default="rtl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_neurons(read_rows(Path(args.neurons)), out_dir / "neuron_params_pkg.vhd")
    emit_synapses(read_rows(Path(args.synapses)), out_dir / "synapse_params_pkg.vhd")
    print(f"Gerado: {out_dir / 'neuron_params_pkg.vhd'}")
    print(f"Gerado: {out_dir / 'synapse_params_pkg.vhd'}")


if __name__ == "__main__":
    main()
