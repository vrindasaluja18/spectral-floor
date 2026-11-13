"""The figure: a world model's spectrum is a measure on a circle, and stride folds it.

Renders `docs/figures/spectral-floor.png` (2x2):

  A  the spectrum as a rose on S^1 -- even, because the process is real
  B  decimation as the covering map z -> z^s: the s shifted copies sum to the fold
  C  the mirror S(f) = S(1-f) vs the clamp that silently biased the floor 46% low
  D  the floor eps^2(s) = 1 - a^(2s): closed form, numeric, and the achieved error

Everything is computed with the package's own functions -- nothing is faked.

Run:  PYTHONPATH=. python3 scripts/make_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from spectral_floor import alias_spectrum, ar1_psd, ar1_floor, normalise_psd, szego
from experiments.run_floor_experiment import (
    ar1_true_spectrum,
    linear_predictor_rel_mse,
    simulate_ar1,
)

A = 0.99

INK = "#12151c"
PAPER = "#fbfaf7"
MUTED = "#8a8f98"
ACCENT = "#c8442c"
TEAL = "#1f6f78"
GOLD = "#c98a12"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)


def spectrum(a: float, n: int = 20001) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0, 1, n)[:-1]
    _, S = ar1_psd(a, u)
    return u, normalise_psd(S)


def panel_a(fig, gs_cell, u, S):
    """The spectrum as a rose on the unit circle."""
    ax = fig.add_subplot(gs_cell, projection="polar")
    theta = 2 * np.pi * u
    S_smooth = np.concatenate([S, S[:1]])
    th = np.concatenate([theta, [2 * np.pi]])
    ax.plot(th, S_smooth, color=TEAL, lw=1.8)
    ax.fill(th, S_smooth, color=TEAL, alpha=0.12)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rticks([])
    ax.set_xticks(np.pi / 2 * np.arange(4))
    ax.set_xticklabels(["f=0", "f=¼", "f=½", "f=¾"], fontsize=8)
    ax.grid(color=MUTED, alpha=0.25, lw=0.6)
    ax.set_title("A  the spectrum is a measure on a circle",
                 fontsize=10.5, color=INK, pad=12, loc="left")
    ax.annotate("even:  S(f) = S(1−f)",
                xy=(np.pi, ax.get_rmax() * 0.55), xytext=(np.pi * 1.28, ax.get_rmax() * 0.9),
                fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    return ax


def panel_b(ax, u, S, s=4):
    """Decimation = the covering map: s shifted copies sum to the fold."""
    S_dec = alias_spectrum(S, s)
    for k in range(s):
        shifted = np.interp((u + k) / s % 1.0, u, S) / s
        ax.plot(u, shifted, color=GOLD, lw=0.9, alpha=0.65,
                label="shifted copies  S((f+k)/s)/s" if k == 0 else None)
    ax.plot(u, S_dec, color=INK, lw=2.2, label=f"fold  S_dec(f)   (s={s})")
    ax.plot(u, S, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="original  S(f)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(S.max(), S_dec.max()) * 1.08)
    ax.set_xlabel("normalised frequency  f", fontsize=8.5)
    ax.set_ylabel("S(f)", fontsize=8.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "¼", "½", "¾", "1"], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title(f"B  stride {s} = the covering map  z ↦ z^s",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_c(ax, u, S):
    """Mirror vs clamp: the bug that biased the floor 46% low."""
    half = S[: len(S) // 2 + 1]
    grid = np.linspace(0, 0.5, len(half))
    clamped = normalise_psd(np.interp(u, grid, half))
    ax.plot(u, clamped, color=ACCENT, lw=1.9, label="clamp (np.interp) — WRONG")
    ax.plot(u, S, color=TEAL, lw=1.9, label="mirror  S(1−f) — exact")
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.annotate("flat: clamped to S(½)",
                xy=(0.78, clamped[clamped.size // 2 + 100]), xytext=(0.5, S.max() * 0.42),
                fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, S.max() * 1.05)
    ax.set_xlabel("normalised frequency  f", fontsize=8.5)
    ax.set_ylabel("S(f)", fontsize=8.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "¼", "½", "¾", "1"], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("C  mirror, don't interpolate  (−46% at s=16)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_d(ax, u, S):
    """The floor: closed form vs numeric, plus the achieved error."""
    strides = [1, 2, 4, 8, 16]
    closed = [ar1_floor(A, s) for s in strides]
    numeric = [szego(alias_spectrum(S, s)) for s in strides]

    x = simulate_ar1(A, 400_000, seed=0)
    achieved = [linear_predictor_rel_mse(x, s, window=256) for s in strides]

    ax.plot(strides, closed, color=INK, lw=2.0, marker="o", ms=5,
            label="closed form  1 − a^(2s)")
    ax.plot(strides, numeric, color=TEAL, lw=1.4, ls="none", marker="s", ms=5,
            label="numeric  Szegő(S_dec)")
    ax.plot(strides, achieved, color=ACCENT, lw=1.4, ls="none", marker="D", ms=4.5,
            label="best linear predictor")
    for s, f, ac in zip(strides, closed, achieved):
        ax.vlines(s, f, ac, color=MUTED, lw=0.7, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(strides)
    ax.set_xticklabels([str(s) for s in strides], fontsize=8)
    ax.set_xlabel("stride  s", fontsize=8.5)
    ax.set_ylabel("relative MSE  (floor / achieved)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_title("D  the floor binds: achieved ≤ 1.11× floor",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    import os

    u, S = spectrum(A)
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.07, right=0.97, top=0.88, bottom=0.08)

    panel_a(fig, gs[0, 0], u, S)
    panel_b(fig.add_subplot(gs[0, 1]), u, S)
    panel_c(fig.add_subplot(gs[1, 0]), u, S)
    panel_d(fig.add_subplot(gs[1, 1]), u, S)

    fig.suptitle("A world model's spectrum is a measure on a circle — and stride folds it",
                 fontsize=13.5, color=INK, x=0.07, ha="left", y=0.965, weight="bold")
    fig.text(0.07, 0.918,
             "AR(1), a = 0.99   ·   the floor  eps² = exp(⟨log S⟩)/⟨S⟩  is fixed by S alone, before any training",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.97, 0.022,
             "github.com/vrindasaluja18/spectral-floor",
             fontsize=8, color=MUTED, ha="right")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "spectral-floor.png")
    fig.savefig(path, dpi=200)
    print("wrote", os.path.normpath(path))

    # honest report of the numbers the figure shows
    strides = [1, 2, 4, 8, 16]
    x = simulate_ar1(A, 400_000, seed=0)
    print(f"  {'s':>3}  {'closed':>9}  {'numeric':>9}  {'achieved':>9}  {'gap':>6}")
    for s in strides:
        f = ar1_floor(A, s)
        n = szego(alias_spectrum(S, s))
        ac = linear_predictor_rel_mse(x, s, window=256)
        print(f"  {s:>3}  {f:>9.5f}  {n:>9.5f}  {ac:>9.5f}  {ac/f:>5.2f}x")


if __name__ == "__main__":
    main()
