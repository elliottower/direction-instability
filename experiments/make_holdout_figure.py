"""Generate 2-panel held-out prediction figure for the paper."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/03_core_defenses/holdout_prediction.json") as f:
    data = json.load(f)

dis = [r["mean_loo_di"] for r in data]
coss = [r["mean_heldout_cosine"] for r in data]
hdac = [r for r in data if r["moa"] and "HDAC inhibitor" in str(r["moa"])]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

# Panel A: scatter
ax = axes[0]
ax.scatter(dis, coss, alpha=0.08, s=2, color="#4878CF", rasterized=True)
if hdac:
    hdac_dis = [r["mean_loo_di"] for r in hdac]
    hdac_cos = [r["mean_heldout_cosine"] for r in hdac]
    ax.scatter(hdac_dis, hdac_cos, s=18, color="#C44E52", zorder=5,
               edgecolors="white", linewidths=0.3, label="HDAC inhibitors")
    for r in hdac:
        if r["mean_heldout_cosine"] > 0.62 or r["mean_heldout_cosine"] < 0.12:
            ax.annotate(r["drug"][:14], (r["mean_loo_di"], r["mean_heldout_cosine"]),
                        fontsize=4.5, alpha=0.8, xytext=(3, 2), textcoords="offset points")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
ax.set_xlabel("LOO direction instability", fontsize=8)
ax.set_ylabel("Mean held-out cosine", fontsize=8)
ax.set_title(r"(a) LOO prediction ($\rho = -0.98$, AUROC $= 0.986$)", fontsize=7.5)
ax.tick_params(labelsize=7)

# Panel B: histogram by quartile
ax = axes[1]
di_arr = np.array(dis)
cos_arr = np.array(coss)
q25, q75 = np.percentile(di_arr, [25, 75])
low = cos_arr[di_arr <= q25]
high = cos_arr[di_arr >= q75]
ax.hist(low, bins=35, alpha=0.65, color="#4878CF", density=True,
        label=f"Stable (Q1, $n={len(low)}$)")
ax.hist(high, bins=35, alpha=0.65, color="#C44E52", density=True,
        label=f"Unstable (Q4, $n={len(high)}$)")
ax.set_xlabel("Mean held-out cosine", fontsize=8)
ax.set_ylabel("Density", fontsize=8)
ax.set_title("(b) Held-out cosine by instability quartile", fontsize=7.5)
ax.legend(fontsize=6, framealpha=0.9)
ax.tick_params(labelsize=7)

plt.tight_layout(w_pad=1.5)
plt.savefig("paper/fig_holdout_prediction.pdf", dpi=300, bbox_inches="tight")
plt.savefig("paper/fig_holdout_prediction.png", dpi=200, bbox_inches="tight")
print("Saved paper/fig_holdout_prediction.pdf")
