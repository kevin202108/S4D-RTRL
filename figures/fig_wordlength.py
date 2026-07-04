"""Paper figure: fixed-point wordlength sweeps (data from data_stage3_run6.log).

Left: uniform quantization vs split (state/params W16 + learning tensors @ W).
Right: deployed-datapath state wordlength, PTQ (no adaptation) vs online-adapted.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOAT_REF = -50.85  # adapted float reference ACLR (dB)

# sweep 1: uniform W, stochastic rounding (nearest unstable below W12)
uni_W = [16, 12, 10, 8]
uni_aclr = [-50.84, -50.58, -46.84, -42.27]
# sweep 3: split — state/par @W16, learning tensors @W
spl_W = [12, 10, 8]
spl_aclr = [-50.84, -50.83, -50.81]

# sweep 5: deployed state quantization, learning @W8 split
dep_W = [16, 12, 10]
dep_ptq = [-49.88, -48.84, -41.44]
dep_adapt = [-50.82, -49.48, -41.50]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2))

ax1.axhline(FLOAT_REF, color="0.4", ls=":", lw=1, label="float ref")
ax1.plot(uni_W, uni_aclr, "o-", color="tab:red", label="uniform W (all tensors)")
ax1.plot(spl_W, spl_aclr, "s-", color="tab:blue",
         label="split: learning @W,\nstate/params @16b")
ax1.set_xlabel("wordlength W (bits)")
ax1.set_ylabel("adapted ACLR (dB)")
ax1.set_xticks([8, 10, 12, 16])
ax1.invert_xaxis()
ax1.set_title("(a) Learning-path quantization")
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(alpha=0.3)
ax1.annotate("8-bit learning:\n+0.04 dB", xy=(8.15, -50.6), xytext=(9.3, -48.6),
             fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

x = range(len(dep_W))
w = 0.36
ax2.axhline(FLOAT_REF, color="0.4", ls=":", lw=1, label="float ref")
BOT = -52
ax2.bar([i - w / 2 for i in x], [v - BOT for v in dep_ptq], w, bottom=BOT,
        color="0.7", label="PTQ (frozen)")
ax2.bar([i + w / 2 for i in x], [v - BOT for v in dep_adapt], w, bottom=BOT,
        color="tab:blue", label="online-adapted")
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"W{v}" for v in dep_W])
ax2.set_xlabel("deployed state wordlength")
ax2.set_ylabel("ACLR (dB)")
ax2.set_ylim(-52, -38)
ax2.set_title("(b) Deployed-datapath state")
ax2.legend(fontsize=7.5, loc="upper left")
ax2.grid(alpha=0.3, axis="y")
for i, (p, a) in enumerate(zip(dep_ptq, dep_adapt)):
    ax2.text(i + w / 2, a + 0.3, f"{a:.1f}", ha="center", fontsize=7)
    ax2.text(i - w / 2, p + 0.3, f"{p:.1f}", ha="center", fontsize=7, color="0.35")

fig.tight_layout()
fig.savefig("figures/fig_wordlength.png", dpi=200)
print("wrote figures/fig_wordlength.png")
