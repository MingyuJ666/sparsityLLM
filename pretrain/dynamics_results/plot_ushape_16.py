import json
import matplotlib.pyplot as plt
import numpy as np

with open('dynamics_llama-16-16_5000steps.json') as f:
    results = json.load(f)

steps = [r["step"] for r in results]
top5 = [r["id_top5pct_energy"] for r in results]
acc = [r["id_acc"] * 100 for r in results]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False,
})

fig, ax1 = plt.subplots(figsize=(8, 5))

color_energy = '#d95f02'
color_acc = '#1f78b4'

# Top-5% Energy (left y-axis)
ax1.plot(steps, top5, color=color_energy, marker='o', markersize=5,
         linewidth=2, label='Top-5% Energy')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Top-5% Energy Ratio', color=color_energy)
ax1.tick_params(axis='y', labelcolor=color_energy)

# Accuracy (right y-axis)
ax2 = ax1.twinx()
ax2.plot(steps, acc, color=color_acc, marker='s', markersize=5,
         linewidth=2, linestyle='--', label='Accuracy')
ax2.set_ylabel('Accuracy (%)', color=color_acc)
ax2.tick_params(axis='y', labelcolor=color_acc)

# Title
ax1.set_title('ID Sparsity Learning Dynamics (llama-16-16)')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('id_sparsity_ushape_16x16.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: id_sparsity_ushape_16x16.pdf")
