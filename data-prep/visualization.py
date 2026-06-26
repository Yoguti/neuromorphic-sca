import numpy as np
import pandas as pd
import h5py
import os
import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_DIR  = "../AES_HD_Dataset"
NPZ_PATH  = os.path.join(DATA_DIR, "aes_hd.npz")
OUT_H5    = "../modulated_dataset/aes_hd_snn_ready.h5"
PLOT_DIR  = "../modulated_dataset/plots"
NUM_POIS  = 20
TEST_FRAC = 0.20
SEED      = 42

os.makedirs(PLOT_DIR, exist_ok=True)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(PLOT_DIR, "visualization_results.txt"))

TARGET_CT_HI = 15
TARGET_CT_LO = 11
KEY_HEX      = "2b7e151628aed2a6abf7158809cf4f3c"

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

RCON = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

def _xtime(a):
    a <<= 1
    if a & 0x100:
        a ^= 0x11b
    return a & 0xff

def aes128_last_round_key(key16):
    SBOX = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        SBOX[INV_SBOX[i]] = i
    rk = [list(key16[0:4]), list(key16[4:8]), list(key16[8:12]), list(key16[12:16])]
    words = [rk[0][:], rk[1][:], rk[2][:], rk[3][:]]
    for rnd in range(1, 11):
        prev = words[-1][:]
        prev = prev[1:] + prev[:1]
        prev = [int(SBOX[b]) for b in prev]
        prev[0] ^= RCON[rnd - 1]
        neww = [words[-4][i] ^ prev[i] for i in range(4)]
        words.append(neww)
        for j in range(1, 4):
            neww = [words[-1][i] ^ words[-4][i] for i in range(4)]
            words.append(neww)
    last = words[40:44]
    out = []
    for w in last:
        out.extend(w)
    return np.array(out, dtype=np.uint8)

def compute_hd_labels(ct, key_last_byte_hi, ct_hi, ct_lo):
    inner = INV_SBOX[np.bitwise_xor(ct[:, ct_hi].astype(np.uint8),
                                    np.uint8(key_last_byte_hi))]
    reg = np.bitwise_xor(inner, ct[:, ct_lo].astype(np.uint8))
    return HW_LUT[reg]

print("Loading AES_HD from NPZ...")
npz = np.load(NPZ_PATH)
data = npz["data"].astype(np.uint8)
plaintext  = data[:, 0:16]
ciphertext = data[:, 16:32]
key_arr    = data[:, 32:48]

key16 = np.frombuffer(bytes.fromhex(KEY_HEX), dtype=np.uint8)
last_rk = aes128_last_round_key(key16)
print(f"Master key:     {KEY_HEX}")
print(f"Last round key: {bytes(last_rk).hex()}")
print(f"Target: HW( InvSbox[ct[{TARGET_CT_HI}] ^ rk_last[{TARGET_CT_HI}]] ^ ct[{TARGET_CT_LO}] )")
print(f"True rk_last[{TARGET_CT_HI}] = {int(last_rk[TARGET_CT_HI])}")

labels_all = compute_hd_labels(ciphertext, last_rk[TARGET_CT_HI], TARGET_CT_HI, TARGET_CT_LO)

old_csv = os.path.join(DATA_DIR, "labels.csv")
if os.path.exists(old_csv):
    old = pd.read_csv(old_csv, header=None).values.ravel().astype(np.uint8)
    n = min(len(old), len(labels_all))
    old_hw = HW_LUT[old[:n]] if old.max() > 8 else old[:n]
    agree = float(np.mean(old_hw == labels_all[:n]))
    print(f"Agreement with existing labels.csv: {agree*100:.1f}%")
    if agree < 0.99:
        print("WARNING: computed HD labels do not match labels.csv.")
        print("The (ct_hi, ct_lo) pair or key convention may differ. Inspect before training.")
    else:
        print("Leakage model CONFIRMED against existing labels.")
else:
    print("No existing labels.csv to cross-check (proceeding with computed labels).")

print("Loading traces...")
dfs = []
for i in range(1, 6):
    p = os.path.join(DATA_DIR, f"traces_{i}.csv")
    if os.path.exists(p):
        dfs.append(pd.read_csv(p, header=None, sep=r'\s+', dtype=np.float32))
if dfs:
    traces_all = np.vstack([df.values for df in dfs])
else:
    traces_all = npz["traces"].astype(np.float32) if "traces" in npz.files else None
    if traces_all is None:
        raise SystemExit("No traces found in CSVs or NPZ.")

n = min(traces_all.shape[0], labels_all.shape[0], ciphertext.shape[0])
traces_all = traces_all[:n]
labels_all = labels_all[:n]
ciphertext = ciphertext[:n]

print(f"Loaded: {traces_all.shape[0]} traces x {traces_all.shape[1]} samples")
print(f"Classes: {np.unique(labels_all)}, counts: {np.bincount(labels_all, minlength=9)}")

num_classes = 9
num_s = traces_all.shape[1]

mean_pc = np.zeros((num_classes, num_s), dtype=np.float64)
var_pc  = np.zeros((num_classes, num_s), dtype=np.float64)
for c in range(num_classes):
    idx = np.where(labels_all == c)[0]
    if len(idx) > 0:
        mean_pc[c] = np.mean(traces_all[idx], axis=0)
        var_pc[c]  = np.var(traces_all[idx],  axis=0)

snr = np.var(mean_pc, axis=0) / (np.mean(var_pc, axis=0) + 1e-9)
poi_indices = np.sort(np.argsort(snr)[-NUM_POIS:])
print(f"Top {NUM_POIS} POI indices: {poi_indices}")
print(f"SNR at selected POIs: max={snr[poi_indices].max():.6f}, min={snr[poi_indices].min():.6f}")

poi_raw  = traces_all[:, poi_indices].astype(np.float64)
poi_mean = poi_raw.mean(axis=0)
poi_std  = poi_raw.std(axis=0) + 1e-9
poi_z    = ((poi_raw - poi_mean) / poi_std).astype(np.float32)

def make_plots():
    os.makedirs(PLOT_DIR, exist_ok=True)

    sample_trace = traces_all[0]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample_trace, lw=0.6, color="#1f77b4")
    ax.set_title(f"Example trace (idx=0, HD-HW={labels_all[0]})")
    ax.set_xlabel("Sample (time)")
    ax.set_ylabel("Amplitude (power)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_sample_trace.png"), dpi=120)
    plt.close(fig)

    counts = np.bincount(labels_all, minlength=num_classes)
    binom = np.array([math.comb(8, k) for k in range(9)], dtype=np.float64)
    binom_expected = binom / binom.sum() * counts.sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(num_classes)
    ax.bar(xs, counts, color="#ff7f0e", alpha=0.85, label="Actual count")
    ax.plot(xs, binom_expected, "o--", color="#2c3e50", label="Expected binomial C(8,k)")
    ax.set_title("HD Hamming-Weight distribution")
    ax.set_xlabel("HW class"); ax.set_ylabel("Traces"); ax.set_xticks(xs)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_hw_distribution.png"), dpi=120)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(sample_trace, lw=0.6, color="#1f77b4", alpha=0.7)
    a1.scatter(poi_indices, sample_trace[poi_indices], color="red", s=40, zorder=5,
               label=f"{NUM_POIS} POIs")
    a1.set_title("Trace with POIs"); a1.set_ylabel("Amplitude"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(snr, lw=0.8, color="#2ca02c")
    a2.scatter(poi_indices, snr[poi_indices], color="red", s=40, zorder=5)
    a2.set_title("SNR"); a2.set_xlabel("Sample"); a2.set_ylabel("SNR"); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_trace_with_pois.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(num_classes):
        ax.plot(mean_pc[c][poi_indices], marker="o", lw=1, alpha=0.8, label=f"HW={c}")
    ax.set_title("Per-class POI means")
    ax.set_xlabel("POI index"); ax.set_ylabel("Mean amplitude")
    ax.set_xticks(range(NUM_POIS)); ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_poi_class_means.png"), dpi=120)
    plt.close(fig)
    print(f"Plots saved to {PLOT_DIR}/")

def poi_sweep(traces, labels, snr_full, train_idx, test_idx,
              poi_counts=(10, 20, 50, 100, 200, 400)):
    print("\n==================== POI SWEEP ====================")
    print(f"{'POIs':>6}{'LR_bal':>10}{'RF_bal':>10}")
    lr_scores, rf_scores, used = [], [], []
    rng_local = np.random.default_rng(SEED)
    for npoi in poi_counts:
        if npoi > traces.shape[1]:
            continue
        idx = np.sort(np.argsort(snr_full)[-npoi:])
        raw = traces[:, idx].astype(np.float64)
        mu = raw[train_idx].mean(axis=0)
        sd = raw[train_idx].std(axis=0) + 1e-9
        z = ((raw - mu) / sd).astype(np.float32)
        Xtr, ytr = z[train_idx], labels[train_idx]
        Xte, yte = z[test_idx],  labels[test_idx]

        lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        lr.fit(Xtr, ytr)
        from sklearn.metrics import balanced_accuracy_score
        lr_b = balanced_accuracy_score(yte, lr.predict(Xte))

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    n_jobs=-1, random_state=SEED)
        rf.fit(Xtr, ytr)
        rf_b = balanced_accuracy_score(yte, rf.predict(Xte))

        lr_scores.append(lr_b); rf_scores.append(rf_b); used.append(npoi)
        print(f"{npoi:>6}{lr_b:>10.4f}{rf_b:>10.4f}")
    print("==================================================")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(used, lr_scores, "o-", label="LogisticRegression", color="#1f77b4")
    ax.plot(used, rf_scores, "s-", label="RandomForest", color="#d62728")
    ax.axhline(1.0/9, color="gray", ls="--", lw=1, label="Random (1/9)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of POIs (log)"); ax.set_ylabel("Balanced accuracy (attack split)")
    ax.set_title("Information ceiling vs POI count")
    ax.legend(); ax.grid(alpha=0.3)
    for x, y in zip(used, lr_scores):
        ax.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_poi_sweep.png"), dpi=120)
    plt.close(fig)
    best = used[int(np.argmax(lr_scores))]
    print(f"Best LR balanced acc at {best} POIs ({max(lr_scores):.4f})")
    print(f"Plot: {PLOT_DIR}/05_poi_sweep.png")

rng = np.random.default_rng(SEED)
classes = list(range(num_classes))
train_parts, test_parts = [], []
for c in classes:
    idx_c = np.where(labels_all == c)[0]
    rng.shuffle(idx_c)
    n_test_c = int(round(TEST_FRAC * len(idx_c)))
    test_parts.append(idx_c[:n_test_c])
    train_parts.append(idx_c[n_test_c:])
keep_train = np.sort(np.concatenate(train_parts))
test_idx   = np.sort(np.concatenate(test_parts))

make_plots()
poi_sweep(traces_all, labels_all, snr, keep_train, test_idx)