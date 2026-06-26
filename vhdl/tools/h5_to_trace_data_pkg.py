#!/usr/bin/env python3
"""
Converte uma trace do modulated_dataset/aes_hd_snn_ready.h5 para rtl/trace_data_pkg.vhd.

O HDF5 esperado segue o loader C do repo:
  /Profiling_traces/traces     -> float32 [N, 20]
  /Profiling_traces/labels_hw  -> uint8   [N]
  /Attack_traces/traces        -> float32 [N, 20]
  /Attack_traces/labels_hw     -> uint8   [N]

Formato VHDL gerado: Q8.8 assinado por padrão.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py


def qfmt(values: np.ndarray, frac_bits: int, width: int) -> np.ndarray:
    scale = 1 << frac_bits
    q = np.rint(values.astype(np.float64) * scale).astype(np.int64)
    lo = -(1 << (width - 1))
    hi = (1 << (width - 1)) - 1
    return np.clip(q, lo, hi)


def emit_pkg(q: np.ndarray, label: int | None, out: Path, source: str, group: str, index: int,
             frac_bits: int, width: int) -> None:
    if q.shape[0] != 20:
        raise ValueError(f"Esperado vetor com 20 POIs, recebido shape={q.shape}")

    lines = []
    for i, v in enumerate(q.tolist()):
        comma = "," if i < len(q) - 1 else ""
        lines.append(f"        {i:2d} => to_signed({int(v)}, FEATURE_WIDTH){comma}")

    label_v = -1 if label is None else int(label)
    text = f'''library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.snn_pkg.all;

package trace_data_pkg is
    ---------------------------------------------------------------------------
    -- Gerado automaticamente por tools/h5_to_trace_data_pkg.py
    -- Fonte: {source}
    -- Grupo: {group}
    -- Indice: {index}
    -- Quantizacao: Q{width - frac_bits - 1}.{frac_bits} signed
    -- Label HD esperada: {label_v}
    ---------------------------------------------------------------------------
    constant TRACE_EXPECTED_LABEL : integer := {label_v};
    constant TRACE_FEATURES : feature_array_t(0 to NUM_POIS - 1) := (
{chr(10).join(lines)}
    );
end package;
'''
    out.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Caminho para aes_hd_snn_ready.h5")
    ap.add_argument("--group", default="Profiling_traces", choices=["Profiling_traces", "Attack_traces"])
    ap.add_argument("--index", type=int, default=0, help="Indice da trace no HDF5")
    ap.add_argument("--out", default="rtl/trace_data_pkg.vhd")
    ap.add_argument("--frac-bits", type=int, default=8)
    ap.add_argument("--width", type=int, default=16)
    args = ap.parse_args()

    h5_path = Path(args.h5)
    out_path = Path(args.out)

    with h5py.File(h5_path, "r") as f:
        traces_path = f"{args.group}/traces"
        labels_path = f"{args.group}/labels_hw"
        if traces_path not in f:
            raise KeyError(f"Dataset ausente: {traces_path}")
        X = f[traces_path]
        if X.ndim != 2 or X.shape[1] != 20:
            raise ValueError(f"Esperado {traces_path} com shape [N,20], recebido {X.shape}")
        if args.index < 0 or args.index >= X.shape[0]:
            raise IndexError(f"Indice {args.index} fora do intervalo 0..{X.shape[0]-1}")
        x = X[args.index].astype(np.float32)
        label = None
        if labels_path in f:
            label = int(f[labels_path][args.index])

    q = qfmt(x, frac_bits=args.frac_bits, width=args.width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emit_pkg(q, label, out_path, str(h5_path), args.group, args.index, args.frac_bits, args.width)

    print(f"Gerado: {out_path}")
    print(f"Label HD esperada: {label if label is not None else 'indisponivel'}")
    print("POIs float32:", " ".join(f"{v:.6g}" for v in x.tolist()))
    print("POIs Q:", " ".join(str(int(v)) for v in q.tolist()))


if __name__ == "__main__":
    main()
