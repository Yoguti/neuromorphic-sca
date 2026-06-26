import numpy as np
import h5py

NUM_INPUTS  = 20
NUM_OUTPUTS = 9
TICKS       = 100       # engine.c INFERENCE_TICKS
ALPHA       = 10.0      # engine.c READOUT_ALPHA
NUM_KEYS    = 256

INV_SBOX = np.array([
0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d],
dtype=np.uint8)

HW_LUT = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def hd_hypotheses(ct, ct_hi, ct_lo):
    """HD class for every key hypothesis: HW(InvSbox[ct_hi ^ k] ^ ct_lo), shape [256, N]."""
    c_hi = ct[:, ct_hi].astype(np.uint8)
    c_lo = ct[:, ct_lo].astype(np.uint8)
    H = np.zeros((NUM_KEYS, ct.shape[0]), dtype=np.uint8)
    for k in range(NUM_KEYS):
        inner = INV_SBOX[np.bitwise_xor(c_hi, np.uint8(k))]
        H[k] = HW_LUT[np.bitwise_xor(inner, c_lo)]
    return H


def load_h5(path, group):
    """Read a trace group ('Profiling_traces' or 'Attack_traces') plus meta."""
    with h5py.File(path, "r") as f:
        X  = f[f"{group}/traces"][:].astype(np.float32)
        y  = f[f"{group}/labels_hw"][:].astype(np.int64)
        ct = f[f"{group}/ciphertext"][:].astype(np.uint8)
        meta = (int(f["meta/target_ct_hi"][()]),
                int(f["meta/target_ct_lo"][()]),
                int(f["meta/true_key_byte"][()]))
    return X, y, ct, meta


def load_network(basename):
    """Parse the CSV pair the C engine writes. Outputs are lif 0..8, hidden follow."""
    import pandas as pd
    neu = pd.read_csv(f"{basename}_neurons.csv").sort_values("node_id").reset_index(drop=True)
    syn = pd.read_csv(f"{basename}_synapses.csv")
    src = syn["source_node"].to_numpy(np.int64)
    tgt = syn["target_node"].to_numpy(np.int64) - NUM_INPUTS
    w   = syn["weight"].to_numpy(np.int64)
    is_in = src < NUM_INPUTS
    return {
        "num_lif": len(neu),
        "num_hidden": int((neu["type"] == "hidden").sum()),
        "num_synapses": len(syn),
        "threshold": neu["threshold"].to_numpy(np.int64)[:, None],
        "leak":      neu["leak_factor"].to_numpy(np.int64)[:, None],
        "reset":     neu["reset_potential"].to_numpy(np.int64)[:, None],
        "resting":   neu["resting_potential"].to_numpy(np.int64)[:, None],
        "refrac_p":  neu["refractory_period"].to_numpy(np.int64)[:, None],
        "in_src": src[is_in],          "in_tgt": tgt[is_in],          "in_w": w[is_in],
        "nn_src": src[~is_in] - NUM_INPUTS, "nn_tgt": tgt[~is_in], "nn_w": w[~is_in],
    }


def _wrap_int16(x):
    return ((x + 32768) & 0xFFFF) - 32768


def spike_counts(net, X):
    """Run the network over X for TICKS, return output spike counts [N, 9].

    Replicates neuron.c lif_step and network.c snn_tick: per-synapse float drive
    truncated toward zero, int16 membrane, arithmetic-shift leak with a +/-1 floor,
    refractory hold, threshold reset. Validated against the C engine (gen-0 fitness).
    """
    T, L = X.shape[0], net["num_lif"]

    input_acc = np.zeros((L, T), dtype=np.int64)
    if net["in_src"].size:
        drive = net["in_w"][:, None].astype(np.float32) * X[:, net["in_src"]].T.astype(np.float32)
        drive = drive.astype(np.int32).astype(np.int64)
        np.add.at(input_acc, net["in_tgt"], drive)

    membrane  = np.repeat(net["resting"], T, axis=1).astype(np.int64)
    has_fired = np.zeros((L, T), dtype=np.int64)
    refrac    = np.zeros((L, T), dtype=np.int64)
    counts    = np.zeros((L, T), dtype=np.int64)

    thr, lk = net["threshold"], net["leak"]
    rst, rest, rp = net["reset"], net["resting"], net["refrac_p"]

    for _ in range(TICKS):
        acc = input_acc.copy()
        if net["nn_src"].size:
            np.add.at(acc, net["nn_tgt"], has_fired[net["nn_src"]] * net["nn_w"][:, None])

        in_refr = refrac > 0
        v = membrane + acc
        np.clip(v, -32768, 32767, out=v)
        delta = v - rest
        leak_v = np.right_shift(delta, lk)
        leak_v = np.where((delta != 0) & (leak_v == 0), np.sign(delta), leak_v)
        v2 = _wrap_int16(v - leak_v)

        fired = v2 >= thr
        membrane  = np.where(in_refr, membrane, np.where(fired, rst, v2))
        has_fired = np.where(in_refr, 0, fired.astype(np.int64))
        refrac    = np.where(in_refr, refrac - 1, np.where(fired, rp, 0))
        counts[:NUM_OUTPUTS] += has_fired[:NUM_OUTPUTS]

    return counts[:NUM_OUTPUTS].T


def log_proba(counts):
    """Laplace-smoothed log P(class | trace) from spike counts."""
    total = counts.sum(axis=1, keepdims=True)
    return np.log((counts + ALPHA) / (total + NUM_OUTPUTS * ALPHA))


def guessing_entropy(logp, ct, meta, n_axis=60, n_avg=50, seed=1):
    """Key-byte guessing entropy over a growing number of attack traces.

    Returns final rank, recovered byte, the GE curve, its trace axis, and the
    trace count at which GE first drops to <= 0.5 (None if never).
    """
    ct_hi, ct_lo, true_k = meta
    H = hd_hypotheses(ct, ct_hi, ct_lo)
    N = logp.shape[0]
    idx = np.arange(N)
    per_trace = np.stack([logp[idx, H[k]] for k in range(NUM_KEYS)])

    final = per_trace.sum(axis=1)
    recovered = int(np.argmax(final))
    final_rank = int((final > final[true_k]).sum())

    rng = np.random.default_rng(seed)
    axis = np.unique(np.linspace(1, N, n_axis).astype(int))
    ge = np.zeros(len(axis))
    for _ in range(n_avg):
        c = np.cumsum(per_trace[:, rng.permutation(N)], axis=1)
        for ti, nt in enumerate(axis):
            col = c[:, nt - 1]
            ge[ti] += (col > col[true_k]).sum()
    ge /= n_avg

    reached = np.where(ge <= 0.5)[0]
    disclosure = int(axis[reached[0]]) if reached.size else None
    return final_rank, recovered, ge, axis, disclosure
