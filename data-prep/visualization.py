import numpy as np
import pandas as pd
import h5py
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DATA_DIR  = "../AES_HD_Dataset"
OUT_H5    = "../modulated_dataset/aes_hd_snn_ready.h5"
NUM_POIS  = 10
print("Loading AES_HD traces...")
dfs = []
for i in range(1, 6):
    dfs.append(pd.read_csv(os.path.join(DATA_DIR, f"traces_{i}.csv"), header=None, sep=r'\s+', dtype=np.float32))
traces_all = np.vstack([df.values for df in dfs])
labels_raw = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"), header=None).values.ravel().astype(np.uint8)
HW_LUT = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
labels_all = HW_LUT[labels_raw]

if traces_all.shape[0] != labels_all.shape[0]:
    min_len = min(traces_all.shape[0], labels_all.shape[0])
    traces_all = traces_all[:min_len]
    labels_all = labels_all[:min_len]

print(f"Loaded: {traces_all.shape[0]} traces x {traces_all.shape[1]} samples")
print(f"Classes: {np.unique(labels_all)}, counts: {np.bincount(labels_all)}")

num_classes = len(np.unique(labels_all))
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

rng = np.random.default_rng(42)
classes = sorted(np.unique(labels_all))
min_c = min(np.sum(labels_all == c) for c in classes)
print(f"\nBalancing: {len(classes)} classes, min_count={min_c}, total={len(labels_all)}")

keep_train = np.concatenate([rng.choice(np.where(labels_all == c)[0], min_c, replace=False) for c in classes])
keep_train = np.sort(keep_train)

test_mask = np.ones(len(labels_all), dtype=bool)
test_mask[keep_train] = False
test_idx = np.where(test_mask)[0]

poi_z_train = poi_z[keep_train]
y_train     = labels_all[keep_train]
poi_z_test  = poi_z[test_idx]
y_test      = labels_all[test_idx]

print(f"Train: {len(keep_train)} traces ({min_c} per class)")
print(f"Test:  {len(test_idx)} traces (natural distribution)")

print("\nVerifying signal...")
if __name__ == '__main__':
    scores = cross_val_score(LogisticRegression(max_iter=2000, C=0.1), poi_z_train, y_train, cv=5, n_jobs=-1)
    print(f"LR CV on train (balanced): {scores.mean():.4f} (random={1.0/len(classes):.4f})")

    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)

    print(f"\nWriting {OUT_H5} ...")
    with h5py.File(OUT_H5, "w") as f:
        pg = f.create_group("Profiling_traces")
        pg.create_dataset("traces",    data=poi_z_train, dtype=np.float32, compression="gzip")
        pg.create_dataset("labels_hw", data=y_train,     dtype=np.uint8)

        ag = f.create_group("Attack_traces")
        ag.create_dataset("traces",    data=poi_z_test, dtype=np.float32, compression="gzip")
        ag.create_dataset("labels_hw", data=y_test,     dtype=np.uint8)

        m = f.create_group("meta")
        m.create_dataset("poi_indices",  data=poi_indices.astype(np.int32))
        m.create_dataset("poi_mean",     data=poi_mean.astype(np.float32))
        m.create_dataset("poi_std",      data=poi_std.astype(np.float32))
        m.create_dataset("num_pois",     data=np.int32(NUM_POIS))
        m.create_dataset("num_classes",  data=np.int32(len(classes)))

    print("Done.")