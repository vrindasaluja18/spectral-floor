"""Experiment 2: does the Kolmogorov-Szego floor bind on NONLINEAR processes?

The floor `eps2 = exp(<log S>)/<S>` is the *exact* optimal one-step relative MSE of
the best **linear** predictor (verified against a brute-force Yule-Walker solve).  It
is therefore a theorem that a linear predictor reaches it.  The repo's headline --
"the floor binds, use it as a design law" -- was tested only where that theorem
guarantees it: an AR(1) process (linear) predicted by a linear model.

This experiment re-tests the claim on processes that are NOT linear, and reports two
gaps per (process, stride):

    gap_linear = achieved_linear / floor     <- should stay ~1 (it is a theorem)
    gap_esn    = achieved_esn    / floor     <- < 1 means nonlinearity BEATS the floor

A gap_esn well below 1 means the floor is a lower bound for the *linear* class only,
and the design-law claim must be reworded.

Usage:
    PYTHONPATH=. python3 -m experiments.run_nonlinear_experiment
    PYTHONPATH=. python3 -m experiments.run_nonlinear_experiment --out nonlinear.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from spectral_floor import Spectrum, floor


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------
def ar1(a: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def logistic(n: int, r: float = 3.9, x0: float = 0.4, transient: int = 1000) -> np.ndarray:
    x = np.empty(n + transient)
    x[0] = x0
    for i in range(1, n + transient):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x[transient:]


def logistic_noisy(n: int, r: float = 3.8, noise: float = 0.02,
                   x0: float = 0.4, transient: int = 1000, seed: int = 0) -> np.ndarray:
    """Noisy logistic, clamped to [0,1] so the map cannot escape and overflow."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, noise, n + transient)
    x = np.empty(n + transient)
    x[0] = x0
    for i in range(1, n + transient):
        v = r * x[i - 1] * (1.0 - x[i - 1]) + e[i]
        x[i] = min(1.0, max(0.0, v))
    return x[transient:]


def nl_ar_tanh(n: int, seed: int = 0) -> np.ndarray:
    """Bounded stochastic nonlinear AR: x_t = tanh(1.5 x_{t-1}) + 0.3 e_t."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = np.tanh(1.5 * x[i - 1]) + 0.3 * e[i]
    return x


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (x - x.mean()) / (x.std() + 1e-12)


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------
def linear_rel_mse(x: np.ndarray, s: int, window: int = 256) -> float:
    """Best linear one-step predictor on stride-s samples, held-out relative MSE.

    Same construction as `run_floor_experiment.py` (ridge on a `window`-length
    history), so the AR(1) control reproduces the README table exactly.
    """
    xc = x[::s]
    n = len(xc) - window - 1
    if n < 200:
        raise ValueError("not enough samples")
    X = np.lib.stride_tricks.sliding_window_view(xc, window)[:n]
    y = xc[window:window + n]
    cut = int(0.7 * n)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-12
    Xtr = (X[:cut] - mu) / sd
    Xte = (X[cut:] - mu) / sd
    A = Xtr.T @ Xtr + 1e-6 * np.eye(window)
    w = np.linalg.solve(A, Xtr.T @ y[:cut])
    return float(np.mean(((Xte @ w) - y[cut:]) ** 2) / np.var(y[cut:]))


def esn_rel_mse(x: np.ndarray, s: int = 1, n_res: int = 150, sr: float = 0.9,
                in_scale: float = 0.5, ridge: float = 1e-6, washout: int = 200,
                train_frac: float = 0.7, seed: int = 0) -> float:
    """Echo state network one-step predictor, held-out relative MSE.

    A random reservoir (recurrent weights are never trained) plus a ridge readout.
    Deliberately untuned: the same hyperparameters are used for every process.
    """
    xc = x[::s]
    T = len(xc)
    rng = np.random.default_rng(seed)

    W = rng.normal(0, 1, (n_res, n_res))
    W[rng.random((n_res, n_res)) > 0.1] = 0.0            # 10% density
    W *= sr / (max(abs(np.linalg.eigvals(W))) + 1e-12)
    Win = rng.uniform(-1, 1, n_res) * in_scale

    R = np.empty((T, n_res))
    r = np.zeros(n_res)
    for t in range(T):
        r = np.tanh(Win * xc[t] + W @ r)
        R[t] = r

    Rt, y = R[washout:-1], xc[washout + 1:]
    cut = int(train_frac * len(Rt))
    Xtr, Xte, ytr, yte = Rt[:cut], Rt[cut:], y[:cut], y[cut:]
    A = Xtr.T @ Xtr + ridge * np.eye(n_res)
    Wout = np.linalg.solve(A, Xtr.T @ ytr)
    return float(np.mean(((Xte @ Wout) - yte) ** 2) / np.var(yte))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(n: int, strides, nperseg: int = 512, window: int = 96) -> dict:
    processes = {
        "ar1(0.99) [linear]": normalise(ar1(0.99, n)),
        "logistic 3.9 [chaotic]": normalise(logistic(n)),
        "logistic+noise": normalise(logistic_noisy(n, r=3.8, noise=0.02)),
        "nl_ar tanh": normalise(nl_ar_tanh(n)),
    }
    rows = []
    for name, x in processes.items():
        S = Spectrum.from_welch(x, nperseg=nperseg)
        for s in strides:
            f = floor(S, s, n_boot=0).eps2
            lin = linear_rel_mse(x, s, window=window)
            esn = esn_rel_mse(x, s)
            rows.append({
                "process": name, "stride": int(s), "floor": f,
                "achieved_linear": lin, "achieved_esn": esn,
                "gap_linear": lin / f if f > 0 else float("nan"),
                "gap_esn": esn / f if f > 0 else float("nan"),
            })
    return {"n_samples": n, "nperseg": nperseg, "window": window, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--nperseg", type=int, default=512)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    res = run(args.n, args.strides, args.nperseg, args.window)

    print(f"\n=== nonlinear floor experiment  n={res['n_samples']} "
          f"nperseg={res['nperseg']} window={res['window']} ===\n")
    print(f"  {'process':<24} {'s':>2} {'floor':>9} {'lin':>9} {'gap_lin':>8} "
          f"{'esn':>9} {'gap_esn':>8}")
    print("  " + "-" * 74)
    last = None
    for r in res["rows"]:
        if r["process"] != last:
            print()
            last = r["process"]
        print(f"  {r['process']:<24} {r['stride']:>2} {r['floor']:>9.5f} "
              f"{r['achieved_linear']:>9.5f} {r['gap_linear']:>7.2f}x "
              f"{r['achieved_esn']:>9.5f} {r['gap_esn']:>7.2f}x")
    print("\n  gap_lin ~ 1.0 : the linear predictor reaches the linear floor (a theorem).")
    print("  gap_esn < 1.0 : the ESN beats the linear floor -- the floor is a lower")
    print("                  bound for LINEAR predictors only.\n")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
