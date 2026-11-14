"""Experiment 1: does the Kolmogorov-Szego floor bind on real (or simulated) data?

This is the Week-1 / Week-2 go-no-go.  For each stride s it:
  1. estimates the observation spectrum S(f) as a Spectrum (two-sided),
  2. computes the floor eps2(s) from the aliased spectrum, with an interval,
  3. fits the best linear predictor at stride s,
  4. reports the *aliasing gap* = achieved / floor.

If the gap is small, the floor is a design law.  If it is large, the floor is loose
and the interesting question becomes how much nonlinearity buys.

Usage:
    python3 -m experiments.run_floor_experiment --source ar1 --a 0.99
    python3 -m experiments.run_floor_experiment --source csv --path obs.csv --rate 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectral_floor import Spectrum, floor, floor_curve, ar1_floor


def linear_predictor_rel_mse(x: np.ndarray, s: int, window: int = 256) -> float:
    """Best linear one-step predictor on stride-s samples, held-out relative MSE.

    Uses ridge regression on a `window`-length history.  A large window approximates
    the infinite-order optimal predictor whose error is the Szego floor.
    """
    xc = x[::s]
    n = len(xc) - window - 1
    if n < 200:
        raise ValueError(f"not enough samples for window={window}: need {window+201}, have {len(xc)}")
    X = np.lib.stride_tricks.sliding_window_view(xc, window)[:n]
    y = xc[window:window + n]
    cut = int(0.7 * n)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-12
    Xtr = (X[:cut] - mu) / sd
    Xte = (X[cut:] - mu) / sd
    A = Xtr.T @ Xtr + 1e-6 * np.eye(window)
    w = np.linalg.solve(A, Xtr.T @ y[:cut])
    return float(np.mean(((Xte @ w) - y[cut:]) ** 2) / np.var(y[cut:]))


def simulate_ar1(a: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return (x - x.mean()) / x.std()


def load_csv(path: str, column: int = 0) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    x = data[:, column] if data.ndim > 1 else data
    x = x[np.isfinite(x)]
    return (x - x.mean()) / (x.std() + 1e-12)


def run(x: np.ndarray, strides, window: int = 256, nperseg: int = 512,
        source: str = "?", S_true: Spectrum | None = None) -> dict:
    # Estimate the spectrum as a Spectrum object.  from_welch always returns the
    # two-sided density the folding step needs, so the one-sided-clamp bug cannot
    # be written here at all.
    S = Spectrum.from_welch(x, nperseg=nperseg)

    rows = []
    for s in strides:
        s = int(s)
        res = floor(S, s)
        achieved = linear_predictor_rel_mse(x, s, window=window)
        row = {
            "stride": s,
            "rate_factor": 1.0 / s,
            "floor_est": res.eps2,
            "floor_lo": res.ci_low,
            "floor_hi": res.ci_high,
            "achieved": achieved,
            "gap_est": achieved / res.eps2 if res.eps2 > 0 else float("nan"),
            "warnings": list(res.warnings),
        }
        if S_true is not None:
            ft = floor(S_true, s).eps2
            row["floor_true"] = ft
            row["gap_true"] = achieved / ft if ft > 0 else float("nan")
        rows.append(row)
    return {"source": source, "n_samples": int(len(x)), "nperseg": nperseg, "rows": rows}


def ar1_true_spectrum(a: float) -> Spectrum:
    """Analytic AR(1) Spectrum: the oracle."""
    return Spectrum.from_ar1(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ar1", "csv"], default="ar1")
    ap.add_argument("--a", type=float, default=0.99, help="AR(1) coefficient")
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--path", type=str, default=None)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.source == "ar1":
        x = simulate_ar1(args.a, args.n, args.seed)
        src = f"ar1(a={args.a})"
        S_true = ar1_true_spectrum(args.a)
    else:
        if not args.path:
            raise SystemExit("--path required for --source csv")
        x = load_csv(args.path)
        src = Path(args.path).name
        S_true = None

    res = run(x, args.strides, window=args.window, source=src, S_true=S_true)

    print(f"\n=== spectral floor experiment  [{res['source']}]  n={res['n_samples']} ===\n")
    has_true = S_true is not None
    if has_true:
        print(f"  {'stride':>6}  {'floor(est)':>10}  {'floor(true)':>11}  {'achieved':>9}  "
              f"{'gap(true)':>9}")
        print("  " + "-" * 54)
        for r in res["rows"]:
            print(f"  {r['stride']:>6}  {r['floor_est']:>10.5f}  {r['floor_true']:>11.5f}  "
                  f"{r['achieved']:>9.5f}  {r['gap_true']:>8.2f}x")
        gaps = [r["gap_true"] for r in res["rows"]]
        covered = [r["floor_lo"] <= r["floor_true"] <= r["floor_hi"] for r in res["rows"]]
        n_cov = sum(covered)
        print("\n  note: floor(est) comes from a Welch estimate of S, with a bootstrap")
        print("        interval over the segments.  It tracks the analytic floor(true)")
        print(f"        within a few percent; the interval contains it at {n_cov}/{len(covered)}")
        print("        strides.  At large stride the Welch window smooths the spectrum")
        print("        and the point estimate drifts high, so the interval can miss --")
        print("        the cure is resolution (larger nperseg), not a wider interval.")
    else:
        print(f"  {'stride':>6}  {'floor':>10}  {'95% CI':>18}  {'achieved':>9}  {'gap':>8}")
        print("  " + "-" * 60)
        for r in res["rows"]:
            ci = f"[{r['floor_lo']:.5f}, {r['floor_hi']:.5f}]"
            print(f"  {r['stride']:>6}  {r['floor_est']:>10.5f}  {ci:>18}  "
                  f"{r['achieved']:>9.5f}  {r['gap_est']:>7.2f}x")
        gaps = [r["gap_est"] for r in res["rows"]]

    tight = max(gaps) < 1.5
    print()
    print(f"  max gap = {max(gaps):.2f}x  ->  floor is "
          + ("TIGHT (use it as a design law)" if tight else "LOOSE (measure what nonlinearity buys)"))

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
