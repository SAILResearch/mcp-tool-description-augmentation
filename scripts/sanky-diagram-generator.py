#!/usr/bin/env python3
# Two-panel composite figure, with model-colored legend in plot (b),
# no in-plot labels for (b), and original color scheme preserved for (a).
# Legends are in the upper right for both panels.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Values from your screenshot
models = ["GPT-4.1", "Qwen3-Coder", "GLM-4.5", "Qwen3-80B"]

pct_exceed_as = np.array([0.683982684, 0.839826839, 0.742424242, 0.2380952381]) * 100
pct_success_after_surpass = np.array([0.1385281385, 0.1515151515, 0.1515151515, 0.01298701299]) * 100
pct_ae_above_baseline = np.array([0.3766233766, 0.3593073593, 0.3080808080, 0.1212121212]) * 100
pct_ae_up_given_as_up = np.array([55.06329114, 42.78350515, 41.49659864, 50.90909091])
pct_sr_up_given_as_up = np.array([20.25316456, 18.04123711, 20.40816327, 5.454545455])

# -----------------------------
# Figure configuration
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,   # do not change legend size
})

FIGSIZE = (10.5, 4.8)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

# -----------------------------
# Panel (a): grouped bars (keep original color scheme)
x = np.arange(len(models))
w = 0.24

bars1 = ax1.bar(x - w, pct_exceed_as, width=w, label="% AS > baseline")
bars2 = ax1.bar(x,      pct_ae_up_given_as_up, width=w, label="AE↑ | AS↑")
bars3 = ax1.bar(x + w,  pct_sr_up_given_as_up, width=w, label="SR↑ | AS↑")

ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=0)
ax1.set_ylabel("Percent of tasks")
ax1.set_ylim(0, 100)
ax1.grid(axis="y", alpha=0.3)
ax1.legend(ncol=1, frameon=True, loc="upper right")   # legend in top-right

def add_labels(bars):
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 1.2, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9)
add_labels(bars1); add_labels(bars2); add_labels(bars3)

# -----------------------------
# Panel (b): conversion map (distinct colors per model, no in-plot names)
# Derive a color per model from the default prop cycle
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [palette[i % len(palette)] for i in range(len(models))]

sizes = 12 + (pct_ae_up_given_as_up - pct_ae_up_given_as_up.min()) / (pct_ae_up_given_as_up.ptp() + 1e-9) * 300

handles = []
labels = []
for i, m in enumerate(models):
    h = ax2.scatter(pct_exceed_as[i], pct_sr_up_given_as_up[i],
                    s=sizes[i], color=colors[i], alpha=0.9, label=m)
    # collect for legend
    handles.append(h)
    labels.append(m)

# No texts on the points in plot (b)
ax2.set_xlabel("% tasks with AS > baseline")
ax2.set_ylabel("% SR increased given AS increased")
ax2.set_xlim(-2, 102)
ax2.set_ylim(-2, 102)
ax2.grid(alpha=0.3)

# Legend in the upper right with default size
ax2.legend(handles, labels, title="Model", loc="upper right", frameon=True)

fig.tight_layout()
fig.savefig("two_panel_effort_vs_outcome_colored_legend_topright.pdf", bbox_inches="tight")
print("Saved: two_panel_effort_vs_outcome_colored_legend_topright.pdf")
