import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5_PATH      = "../modulated_dataset/aes_hd_snn_ready.h5"
NET_BASENAME = "network-csvs/best_network"
PLOT_DIR     = "../modulated_dataset/plots/validation"
NUM_INPUTS   = 20
NUM_OUTPUTS  = 9
TICKS        = 40
ALPHA        = 1.0
NUM_KEYS     = 256
NUM_RANKS_AVG = 50

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


def load_network(basename):
    neurons = pd.read_csv(f"{basename}_neurons.csv").sort_values("node_id").reset_index(drop=True)
    synapses = pd.read_csv(f"{basename}_synapses.csv")
    num_lif = len(neurons)
    threshold = neurons["threshold"].to_numpy(np.int64)
    leak      = neurons["leak_factor"].to_numpy(np.int64)
    reset     = neurons["reset_potential"].to_numpy(np.int64)
    resting   = neurons["resting_potential"].to_numpy(np.int64)
    refrac_p  = neurons["refractory_period"].to_numpy(np.int64)
    num_hidden = int((neurons["type"] == "hidden").sum())

    src = synapses["source_node"].to_numpy(np.int64)
    tgt = synapses["target_node"].to_numpy(np.int64)
    w   = synapses["weight"].to_numpy(np.int64)
    tgt_lif = tgt - NUM_INPUTS

    is_in = src < NUM_INPUTS
    return {
        "num_lif": num_lif, "num_hidden": num_hidden, "num_synapses": len(synapses),
        "threshold": threshold[:, None], "leak": leak[:, None],
        "reset": reset[:, None], "resting": resting[:, None], "refrac_p": refrac_p[:, None],
        "in_src": src[is_in], "in_tgt": tgt_lif[is_in], "in_w": w[is_in],
        "nn_src": src[~is_in] - NUM_INPUTS, "nn_tgt": tgt_lif[~is_in], "nn_w": w[~is_in],
    }


def wrap_int16(x):
    return ((x + 32768) & 0xFFFF) - 32768


def snn_counts(net, X):
    T = X.shape[0]
    L = net["num_lif"]

    input_acc = np.zeros((L, T), dtype=np.int64)
    if net["in_src"].size:
        drive = (net["in_w"][:, None].astype(np.float32) * X[:, net["in_src"]].T.astype(np.float32))
        drive = drive.astype(np.int32).astype(np.int64)
        np.add.at(input_acc, net["in_tgt"], drive)

    membrane = np.repeat(net["resting"], T, axis=1).astype(np.int64)
    has_fired = np.zeros((L, T), dtype=np.int64)
    refrac = np.zeros((L, T), dtype=np.int64)
    counts = np.zeros((L, T), dtype=np.int64)

    thr = net["threshold"]; lk = net["leak"]; rst = net["reset"]
    rest = net["resting"]; rp = net["refrac_p"]

    for _ in range(TICKS):
        acc = input_acc.copy()
        if net["nn_src"].size:
            contrib = has_fired[net["nn_src"]] * net["nn_w"][:, None]
            np.add.at(acc, net["nn_tgt"], contrib)

        in_refr = refrac > 0
        v = membrane + acc
        np.clip(v, -32768, 32767, out=v)
        delta = v - rest
        leak_v = np.right_shift(delta, lk)
        adj = (delta != 0) & (leak_v == 0)
        leak_v = np.where(adj, np.sign(delta), leak_v)
        v2 = wrap_int16(v - leak_v)

        fired = v2 >= thr
        membrane_new = np.where(fired, rst, v2)
        has_fired_new = fired.astype(np.int64)
        refrac_new = np.where(fired, rp, 0)

        membrane = np.where(in_refr, membrane, membrane_new)
        has_fired = np.where(in_refr, 0, has_fired_new)
        refrac = np.where(in_refr, refrac - 1, refrac_new)

        counts[:NUM_OUTPUTS] += has_fired[:NUM_OUTPUTS]

    return counts[:NUM_OUTPUTS].T


def load_attack(path):
    with h5py.File(path, "r") as f:
        X = f["Attack_traces/traces"][:].astype(np.float32)
        ct = f["Attack_traces/ciphertext"][:].astype(np.uint8)
        ct_hi = int(f["meta/target_ct_hi"][()])
        ct_lo = int(f["meta/target_ct_lo"][()])
        true_k = int(f["meta/true_key_byte"][()])
    return X, ct, ct_hi, ct_lo, true_k


def counts_to_logproba(counts):
    total = counts.sum(axis=1, keepdims=True)
    return np.log((counts + ALPHA) / (total + NUM_OUTPUTS * ALPHA))


def hyp_hw(ct, ct_hi, ct_lo):
    c_hi = ct[:, ct_hi].astype(np.uint8)
    c_lo = ct[:, ct_lo].astype(np.uint8)
    H = np.zeros((NUM_KEYS, ct.shape[0]), dtype=np.uint8)
    for k in range(NUM_KEYS):
        inner = INV_SBOX[np.bitwise_xor(c_hi, np.uint8(k))]
        H[k] = HW_LUT[np.bitwise_xor(inner, c_lo)]
    return H


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    net = load_network(NET_BASENAME)
    print(f"Network: {net['num_hidden']} hidden, {net['num_synapses']} synapses")

    X, ct, ct_hi, ct_lo, true_k = load_attack(H5_PATH)
    print(f"Attack traces: {len(X)} | target ct pair ({ct_hi},{ct_lo}) | true key byte: {true_k}")

    counts = snn_counts(net, X)
    logp = counts_to_logproba(counts)

    H = hyp_hw(ct, ct_hi, ct_lo)
    N = len(X)
    per_trace = np.empty((NUM_KEYS, N), dtype=np.float64)
    idx = np.arange(N)
    for k in range(NUM_KEYS):
        per_trace[k] = logp[idx, H[k]]

    cum_full = np.cumsum(per_trace, axis=1)
    final_scores = cum_full[:, -1]
    recovered = int(np.argmax(final_scores))
    final_rank = int((final_scores > final_scores[true_k]).sum())

    rng = np.random.default_rng(1)
    trace_axis = np.unique(np.linspace(1, N, 60).astype(int))
    ge = np.zeros(len(trace_axis))
    for _ in range(NUM_RANKS_AVG):
        perm = rng.permutation(N)
        c = np.cumsum(per_trace[:, perm], axis=1)
        for ti, nt in enumerate(trace_axis):
            col = c[:, nt - 1]
            ge[ti] += (col > col[true_k]).sum()
    ge /= NUM_RANKS_AVG

    reached0 = np.where(ge <= 0.5)[0]
    n_to_disclose = int(trace_axis[reached0[0]]) if reached0.size else None

    print(f"True key byte:      {true_k}")
    print(f"Recovered key byte: {recovered}")
    print(f"Final rank of true key (0=best): {final_rank} / 255")
    if n_to_disclose is not None:
        print(f"Traces to disclosure (GE<=0.5): {n_to_disclose}")
    else:
        print("Guessing entropy did not reach 0 within the attack set.")
    if recovered == true_k:
        print("SUCCESS: master key byte recovered.")
    elif final_rank <= 5:
        print(f"PARTIAL: true key in top {final_rank+1}; more traces likely needed.")
    else:
        print("FAIL: key not recovered with this model/trace budget.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trace_axis, ge, color="#1f77b4", lw=1.5)
    ax.axhline(0, color="green", ls="--", lw=1, label="GE=0 (recovered)")
    if n_to_disclose is not None:
        ax.axvline(n_to_disclose, color="orange", ls=":", lw=1.5,
                   label=f"Disclosure @ {n_to_disclose}")
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel("Guessing entropy (rank of true key)")
    ax.set_title("Key-byte guessing entropy vs traces")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_guessing_entropy.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(final_scores, lw=0.6, color="#888888")
    ax.scatter([true_k], [final_scores[true_k]], color="red", zorder=5,
               label=f"True key ({true_k})")
    ax.set_xlabel("Key hypothesis"); ax.set_ylabel("Accumulated log-likelihood")
    ax.set_title("Key hypothesis scores (full attack set)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06_key_scores.png"), dpi=120)
    plt.close(fig)
    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()