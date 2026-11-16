"""Week 1: does the spectral floor bind on a REAL environment?

The package so far is tested on simulated processes (AR(1), logistic map), where
the signal is handed to you.  This experiment takes the roadmap's Week 1 step: a
real Atari game, where it is not.

The hard part is not running the game -- that is four lines.  The hard part is
that the floor is defined for a *stationary 1-D time series*, and a game is
neither of those things until you decide what to look at.  Two decisions:

  1. WHICH OBSERVABLE is "the signal".  We report three, because the answer
     depends on the choice, and comparing them is the point:

       frame_mean   mean brightness of the whole frame
       motion       mean |frame_t - frame_{t-1}| -- motion energy
       ram_byte     the most variable byte of the emulator's own 128-byte RAM

     `ram_byte` exists because object detection on raw ALE frames is a trap: the
     background, both paddles, the score digits and the ball are all near-white,
     so "find the ball" silently finds a static bar, the policy then does nothing,
     the screen freezes, and the spectrum is destroyed.  The emulator already
     knows where everything is; read its state instead of re-deriving it from
     pixels.  (Same lesson as the package's own `Spectrum` design: carry the
     meaning with the data rather than inferring it.)

  2. WHICH SLICE is stationary.  The floor assumes the statistics do not drift.
     A game drifts constantly (points, resets, the ball speeding up).  We measure
     the drift and report it rather than hiding it.

Then, per observable and per stride s: the floor vs a linear predictor vs an
echo-state network, exactly as in `run_nonlinear_experiment.py`.

Usage:
    PYTHONPATH=. python3 -m experiments.run_atari_floor --game pong --steps 6000
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from spectral_floor import Spectrum, floor
from experiments.run_nonlinear_experiment import esn_rel_mse, linear_rel_mse


def rollout(game: str, steps: int, seed: int = 0, frameskip: int = 1):
    """Run until `steps` frames of *active* play are collected; return (frames, ram).

    frames : (steps, 210, 160) uint8 screen stack
    ram    : (steps, 128)      uint8 emulator RAM

    Two things matter and both were learned the hard way:

    * The policy is uniform-random.  A scripted policy that plays well reaches a
      steady state, which is *less* stationary; random play keeps the game moving
      and needs no fragile object detection.

    * The game is RESET on game over.  An Atari episode ends (in Pong, when a
      player reaches 21 points) and the emulator then shows a frozen game-over
      screen indefinitely.  A rollout that ignores this records a constant for its
      whole tail; the held-out variance goes to ~1e-32, every relative MSE becomes
      `inf`, and it looks like a spectacular result rather than a dead screen.  We
      therefore restart whenever `game_over()` is true, and record how many
      episodes that took.

    `frameskip=1` so the stack is a genuine per-frame series; a larger frameskip
    would decimate before we have measured anything.
    """
    from ale_py import ALEInterface, roms

    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("frame_skip", frameskip)
    ale.loadROM(roms.get_rom_path(game))
    action_set = list(ale.getMinimalActionSet())
    ale.reset_game()
    ale.act(action_set[1])                      # FIRE: serve / start the game

    rng = np.random.default_rng(seed)
    frames = np.empty((steps, 210, 160), dtype=np.uint8)
    ram = np.empty((steps, 128), dtype=np.uint8)
    episodes = 1
    for t in range(steps):
        if ale.game_over():
            ale.reset_game()
            ale.act(action_set[1])
            episodes += 1
        ale.act(action_set[int(rng.integers(len(action_set)))])
        frames[t] = ale.getScreenGrayscale()
        ram[t] = ale.getRAM()
    return frames, ram, episodes


def extract_signals(frames: np.ndarray, ram: np.ndarray | None = None) -> dict:
    """Three scalar observables from a rollout.  See module docstring."""
    f = frames.astype(np.float32)

    frame_mean = f.mean(axis=(1, 2))

    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    motion = np.concatenate([[0.0], diff])

    out = {"frame_mean": frame_mean, "motion": motion}

    if ram is not None:
        var = ram.astype(np.float64).var(axis=0)
        j = int(np.argmax(var))
        out["ram_byte"] = ram[:, j].astype(float)
        out["_ram_byte_index"] = j
    return out


def stationarity_report(x: np.ndarray, n_chunks: int = 8) -> dict:
    """How much does the mean/std drift across the rollout?

    Split into chunks, report the spread of each chunk's mean and std, relative to
    the series' overall std.  (Dividing by the *mean* of a centred signal divides
    by ~0 and returns a meaningless 1e11; the overall std is the right scale.)
    """
    x = np.asarray(x, float)
    scale = x.std() + 1e-12
    chunks = np.array_split(x, n_chunks)
    means = np.array([c.mean() for c in chunks])
    stds = np.array([c.std() for c in chunks])
    return {
        "mean_spread_rel": float(means.std() / scale),
        "std_spread_rel": float(stds.std() / (stds.mean() + 1e-12)),
    }


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = x.std()
    return (x - x.mean()) / (s + 1e-12) if s > 0 else x * 0.0


def _guarded_rel_mse(fn, x, s):
    """Run a predictor, but return nan when the held-out target is degenerate.

    A relative MSE divides by the held-out variance.  If the signal is mostly
    constant (frozen game), that denominator is ~0 and the result is `inf` --
    which looks like a profound finding and is really just division by nothing.
    Better to report nan and let the frozen-fraction column explain why.
    """
    try:
        v = fn(x, s)
    except Exception:
        return float("nan")
    if not np.isfinite(v) or v > 1e6:
        return float("nan")
    return v


def run(game: str, steps: int, strides, nperseg: int, seed: int = 0) -> dict:
    frames, ram, episodes = rollout(game, steps, seed=seed)
    signals = extract_signals(frames, ram)

    out = {"game": game, "steps": steps, "episodes": episodes,
           "strides": list(strides),
           "ram_byte_index": signals.pop("_ram_byte_index", None), "observables": {}}
    for name, raw in signals.items():
        x = normalise(raw)
        S = Spectrum.from_welch(x, nperseg=nperseg)
        rows = []
        for s in strides:
            fl = floor(S, s, n_boot=0).eps2
            lin = _guarded_rel_mse(linear_rel_mse, x, s)
            esn = _guarded_rel_mse(esn_rel_mse, x, s)
            rows.append({
                "stride": int(s), "floor": fl,
                "achieved_linear": lin, "achieved_esn": esn,
                "gap_linear": lin / fl if fl > 0 else float("nan"),
                "gap_esn": esn / fl if fl > 0 else float("nan"),
            })
        pos = S.values[S.values > 0]
        out["observables"][name] = {
            "frozen_fraction": float(np.mean(np.abs(np.diff(x)) < 1e-9)),
            "stationarity": stationarity_report(x),
            "dynamic_range_db": float(10 * np.log10(
                (pos.max() + 1e-30) / (pos.min() + 1e-30))),
            "rows": rows,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="pong")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run(args.game, args.steps, args.strides, args.nperseg, args.seed)

    print(f"\n=== Atari floor experiment  game={res['game']}  steps={res['steps']} "
          f"({res['episodes']} episodes) ===")
    print(f"    (ram observable = byte {res['ram_byte_index']})\n")
    for name, blk in res["observables"].items():
        st = blk["stationarity"]
        print(f"  [{name}]  frozen {blk['frozen_fraction']*100:.1f}%  "
              f"dyn-range {blk['dynamic_range_db']:.0f} dB  "
              f"drift: mean {st['mean_spread_rel']:.2f} std {st['std_spread_rel']:.2f}")
        print(f"    {'s':>2} {'floor':>9} {'lin':>9} {'gap_lin':>8} {'esn':>9} {'gap_esn':>8}")
        for r in blk["rows"]:
            print(f"    {r['stride']:>2} {r['floor']:>9.5f} {r['achieved_linear']:>9.5f} "
                  f"{r['gap_linear']:>7.2f}x {r['achieved_esn']:>9.5f} {r['gap_esn']:>7.2f}x")
        print()

    print("  Reading this honestly:")
    print("   - a large frozen fraction means the observable is mostly constant;")
    print("     its spectrum is then an artefact and the floor is not meaningful.")
    print("   - drift > ~1 means the series is not stationary; the floor's")
    print("     assumption is questionable on this slice. Use a single episode.")
    print("   - the floor depends on WHICH observable you chose. That is a finding,")
    print("     not a bug: 'the signal' is a modelling decision, not a given.\n")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
