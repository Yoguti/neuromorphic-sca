import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sca

H5_PATH = "../modulated_dataset/aes_hd_snn_ready.h5"
NET_BASENAME = "../snn/network-csvs/best_network"
PLOT_DIR = "../modulated_dataset/plots/validation/key_recovery"

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    net = sca.load_network(NET_BASENAME)
    X, _, ct, meta = sca.load_h5(H5_PATH, "Attack_traces")
    ct_hi, ct_lo, true_k = meta

    logp = sca.log_proba(sca.spike_counts(net, X))
    final_rank, recovered, ge, axis, disclosure = sca.guessing_entropy(logp, ct, meta)

    H = sca.hd_hypotheses(ct, ct_hi, ct_lo)
    N = logp.shape[0]
    idx = np.arange(N)
    per_trace = np.stack([logp[idx, H[k]] for k in range(256)])
    
    cumulative_evidence = np.cumsum(per_trace, axis=1)
    final_scores = cumulative_evidence[:, -1]

    fig, ax = plt.subplots(figsize=(10, 6))
    rank_curve = ge + 1
    ax.plot(axis, rank_curve, color="#2ecc71", lw=3)
    ax.axhline(1, color="#27ae60", ls="--", lw=2)
    
    if disclosure is not None:
        ax.axvline(disclosure, color="#f39c12", ls=":", lw=2, label=f"Correct Key Reaches 1st Place ({disclosure})")
        ax.legend(loc="upper right", fontsize=11)
        
    ax.set_ylim(min(256, np.max(rank_curve) + 10), 0.5)
    ax.set_xlabel("Observed Encryptions", fontsize=12)
    ax.set_ylabel("Rank of the True Key", fontsize=12)
    ax.set_title("How Quickly the Correct Key Becomes the Best Candidate", fontsize=16, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_correct_key_rank.png"), dpi=150)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(12, 6))
    top_20_idx = np.argsort(final_scores)[-20:][::-1]
    top_20_scores = final_scores[top_20_idx]
    
    colors = []
    for k in top_20_idx:
        if k == true_k:
            colors.append("#2ecc71")
        elif k == recovered and k != true_k:
            colors.append("#e67e22")
        else:
            colors.append("#3498db")
            
    min_score = np.min(final_scores)
    display_scores = top_20_scores - min_score
    
    bars = ax.bar([f"0x{k:02X}" for k in top_20_idx], display_scores, color=colors)
    ax.set_xlabel("Candidate Key Byte", fontsize=12)
    ax.set_ylabel("Accumulated Evidence", fontsize=12)
    ax.set_title("Top 20 Candidate Keys", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_top_candidates.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()