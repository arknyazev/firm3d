"""matplotlib helpers shared by the three MC-comparison workflows."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_xy_rz(out_path, R, phi, Z, title, color="C0"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(R * np.cos(phi), R * np.sin(phi), s=2, alpha=0.3,
                   color=color)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("X [m]"); axs[0].set_ylabel("Y [m]")
    axs[0].set_title(f"{title} — XY")
    axs[1].scatter(R, Z, s=2, alpha=0.3, color=color)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("R [m]"); axs[1].set_ylabel("Z [m]")
    axs[1].set_title(f"{title} — RZ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_s_hist(out_path, s_values, title, nbins=40, overlay=None):
    """overlay: optional (s_grid, density, label) for a reference curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.linspace(0, 1, nbins + 1)
    s_fin = s_values[np.isfinite(s_values)]
    if s_fin.size:
        ax.hist(np.clip(s_fin, 0, 1), bins=edges, density=True, alpha=0.7,
                label=f"samples (n={s_fin.size})")
    if overlay is not None:
        s_grid, density, label = overlay
        ax.plot(s_grid, density, "k--", label=label)
    ax.set_xlim(0, 1)
    ax.set_xlabel("s (Boozer)"); ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weight_hist(out_path, w, title, ess=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    w_pos = np.asarray(w, dtype=np.float64)
    w_pos = w_pos[np.isfinite(w_pos) & (w_pos > 0)]
    if w_pos.size:
        lo = max(w_pos.min(), 1e-300)
        hi = w_pos.max()
        if hi <= lo:
            hi = lo * 10.0
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
        ax.hist(w_pos, bins=bins, alpha=0.8, color="C4")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("IS weight w")
    ax.set_ylabel("count")
    full_title = title if ess is None else f"{title} (ESS={ess:.1f})"
    ax.set_title(full_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)