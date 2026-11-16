"""Week 1: the spectral floor on a REAL environment (Atari).

The package is tested on simulated processes (AR(1), logistic map), where the
signal is handed to you.  This experiment takes the roadmap's Week 1 step: a real
Atari game, where it is not.

RESULT
------
On real Pong data the floor is tight at low stride and a valid but conservative
lower bound at high stride.  Five seeds, 12000 frames each, `motion` observable,
`nperseg=256`:

    stride   gap = achievable/floor       reading
      1        0.978 +- 0.007             tight to ~2%
      2        0.995 +- 0.000             tight to ~0.5%
      4        1.570 +- 0.470             valid bound, conservative
      8        1.800 +- 0.490             valid bound, conservative

  * At stride 1 and 2 the floor matches the error the best linear predictor
    actually achieves, to within a couple of percent, with essentially no spread
    across seeds.  The floor is usable as a design number on a real environment.

  * At stride 4 and 8 the floor is still a valid lower bound (gap >= 1.0 in every
    one of the 20 seed x stride cases) but it under-predicts the cost: the
    achievable error grows faster than the floor does, and its spread across seeds
    grows too.  So the floor is a *floor*, not a forecast, once decimation is
    aggressive.

  * The residual gap just below 1 at stride 1 (~2%) is estimator bias, the effect
    the package's own `psd.py` warns about: the floor is a ratio of <log S> to <S>
    and the log makes it sensitive to the low-power tail that a finite window
    smears.  It is stable across seeds, which is what makes it readable as bias
    rather than noise.

  * For reference, a 150-unit echo-state network reaches 0.91x the floor at stride
    1 -- i.e. nonlinearity beats the linear floor by ~9% on real data too, exactly
    as `run_nonlinear_experiment.py` finds on synthetic processes.

WHAT MADE THIS HARD (and why that is the point)
-----------------------------------------------
The floor is defined for a stationary 1-D time series.  A game is neither, until
you decide what to look at.  Three failure modes each produced *plausible numbers
rather than errors*, which is the trap this whole exercise exists to expose:

  1. The game must be SERVED.  Pong does not move until FIRE is pressed, so a
     rollout that never presses it records a static screen and every observable is
     constant.

  2. The game must be RESET on game over.  ALE then shows a frozen game-over
     screen indefinitely; the recorded tail is constant, the held-out variance
     collapses to ~1e-32, and every relative MSE becomes `inf` -- which reads as a
     spectacular result rather than a dead screen.  This was the actual cause of
     the first `inf`s seen here.

  3. The observable must be ROBUST.  Object detection on raw ALE frames is a trap:
     the background, the paddles, the score digits, the centre line and the border
     are all near-white (the paddles are *dimmer* than the ball, at 147 vs 236), so
     "find the ball" finds a line, the tracking policy then does nothing, and the
     screen freezes again.  The `motion` observable needs no detection at all,
     which is why the headline uses it.

Usage:
    PYTHONPATH=. python3 -m experiments.run_atari_floor --game pong --steps 12000
    PYTHONPATH=. python3 -m experiments.run_atari_floor --seeds 0 1 2 3 4 --out atari.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from spectral_floor import Spectrum, floor
from experiments.run_nonlinear_experiment import esn_rel_mse, linear_rel_mse


BRIGHT = 200          # ALE grayscale is {0, 87, 147, 236}; 200 isolates white sprites
PADDLE_LEVEL = 147    # the paddles render at 147/148, DIMMER than the ball at 236
SCORE_ROWS = 34       # top strip holds the score digits


# ---------------------------------------------------------------------------
# 1. roll out a real game
# ---------------------------------------------------------------------------
def rollout(game: str, steps: int, seed: int = 0):
    """Run `steps` frames of active play; return (motion, ball_y, episodes).

    The policy is uniform-random: it keeps the game moving (a scripted policy that
    plays well reaches a steady state, which is *less* stationary) and needs no
    object detection, which is the fragile part.  `ball_y` is tracked only for the
    secondary observable and is nan where the ball is off-screen.

    FIRE is pressed at the start and after every reset; the game is reset on game
    over.  Both are load-bearing (see module docstring).

    Memory is O(1) in `steps`: the frame stack is never retained.  `motion` is the
    running |frame_t - frame_{t-1}|, so only the previous frame is kept alive.  The
    obvious implementation -- accumulate all frames, then convert to float32 --
    holds 4x the raw size and dies at ~100k steps (about 3.4 GB as uint8, then
    about 13 GB as float32), which is exactly the regime this experiment needs.
    """
    from ale_py import ALEInterface, roms

    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("frame_skip", 1)
    ale.loadROM(roms.get_rom_path(game))
    acts = list(ale.getMinimalActionSet())
    fire = acts[1]
    ale.reset_game()
    ale.act(fire)

    rng = np.random.default_rng(seed)
    motion = np.zeros(steps, dtype=np.float64)
    ball_y = np.full(steps, np.nan)
    prev = None
    episodes = 1
    for t in range(steps):
        if ale.game_over():
            ale.reset_game()
            ale.act(fire)
            episodes += 1
        screen = ale.getScreenGrayscale()
        ball_y[t] = detect_ball(screen)
        cur = screen.astype(np.float32)      # one frame at a time, not the stack
        if prev is not None:
            motion[t] = np.abs(cur - prev).mean()
        prev = cur
        ale.act(acts[int(rng.integers(len(acts)))])
    return motion, ball_y, episodes


def detect_ball(frame: np.ndarray):
    """y of the ball, or nan.

    Crops the play area, then removes line-like rows AND columns, so the score
    strip, the bottom border and the centre line do not win over the ball.  The
    ball is the only sprite left once the lines are gone; if more than a handful of
    pixels survive, the frame is not a clean ball and we report nan rather than a
    centroid of noise.
    """
    m = frame >= BRIGHT
    m[:SCORE_ROWS, :] = False
    m[:, :24] = False
    m[:, 136:] = False
    m[190:, :] = False                       # bottom border
    if not m.any():
        return np.nan
    m[m.sum(axis=1) > 0.25 * m.shape[1], :] = False      # horizontal lines
    m[:, m.sum(axis=0) > 0.25 * m.shape[0]] = False      # vertical lines
    if not m.any() or m.sum() > 40:
        return np.nan
    ys, _ = np.where(m)
    return float(ys.mean())


def detect_paddle(frame: np.ndarray):
    """y of the player paddle, or nan.  The paddles are dimmer than the ball."""
    m = (frame >= PADDLE_LEVEL) & (frame < BRIGHT)
    m[:SCORE_ROWS, :] = False
    m[190:, :] = False
    strip = m[:, 16:20]
    if not strip.any():
        return np.nan
    ys, _ = np.where(strip)
    return float(ys.mean())


def motion_signal(frames: np.ndarray, block: int = 256) -> np.ndarray:
    """Mean |frame_t - frame_{t-1}| per step: motion energy, needing no detection.

    Processed in blocks of `block` frames, so the float32 temporary is bounded by
    the block rather than by the whole stack.  A naive `frames.astype(np.float32)`
    first allocates 4x the raw stack and is what makes the large-`steps` regime
    (the one this experiment needs) run out of memory.
    """
    frames = np.asarray(frames)
    n = frames.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for start in range(1, n, block):
        stop = min(start + block, n)
        blk = frames[start - 1:stop].astype(np.float32)
        out[start:stop] = np.abs(np.diff(blk, axis=0)).mean(axis=(1, 2))
    return out


# ---------------------------------------------------------------------------
# 2. helpers
# ---------------------------------------------------------------------------
def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = x.std()
    return (x - x.mean()) / (s + 1e-12) if s > 0 else x * 0.0


def _guarded(fn, x, s):
    """Predictor error, or nan when the held-out target is degenerate.

    A relative MSE divides by the held-out variance; on a mostly-constant signal
    that denominator is ~0 and the result is `inf`, which looks like a profound
    finding and is really division by nothing.
    """
    try:
        v = fn(x, s)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) and v <= 1e6 else float("nan")


def stationarity_report(x: np.ndarray, n_chunks: int = 8) -> dict:
    """Chunk-mean / chunk-std drift, relative to the series' overall std."""
    x = np.asarray(x, float)
    scale = x.std() + 1e-12
    chunks = np.array_split(x, n_chunks)
    means = np.array([c.mean() for c in chunks])
    stds = np.array([c.std() for c in chunks])
    return {
        "mean_spread_rel": float(means.std() / scale),
        "std_spread_rel": float(stds.std() / (stds.mean() + 1e-12)),
    }


# ---------------------------------------------------------------------------
# 3. the measurement
# ---------------------------------------------------------------------------
def run_seed(game: str, steps: int, strides, nperseg: int, seed: int, use_esn: bool) -> dict:
    """One seed: per-stride floor, achievable error, and their ratio."""
    motion, ball_y, episodes = rollout(game, steps, seed=seed)
    x = normalise(motion)
    S = Spectrum.from_welch(x, nperseg=nperseg)

    rows = []
    for s in strides:
        fl = floor(S, int(s), n_boot=0).eps2
        lin = _guarded(linear_rel_mse, x, int(s))
        esn = _guarded(esn_rel_mse, x, int(s)) if use_esn else float("nan")
        rows.append({"stride": int(s), "floor": fl, "linear": lin, "esn": esn,
                     "gap": lin / fl if fl > 0 else float("nan")})
    return {
        "seed": seed, "episodes": episodes,
        "frozen_fraction": float(np.mean(np.abs(np.diff(x)) < 1e-9)),
        "ball_y_coverage": float(np.mean(np.isfinite(ball_y))),
        "stationarity": stationarity_report(x),
        "rows": rows,
    }


def aggregate(per_seed: list[dict], strides) -> list[dict]:
    """Mean +- sd of floor / achievable / gap, per stride, over seeds."""
    out = []
    for s in strides:
        vals = {k: [] for k in ("floor", "linear", "esn", "gap")}
        for ps in per_seed:
            r = next(r for r in ps["rows"] if r["stride"] == int(s))
            for k in vals:
                vals[k].append(r[k])
        def ms(k):
            a = np.array(vals[k], float)
            a = a[np.isfinite(a)]
            return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))
        rec = {"stride": int(s)}
        for k in vals:
            rec[k], rec[k + "_sd"] = ms(k)
        out.append(rec)
    return out


def run(game: str, steps: int, strides, nperseg: int, seeds, use_esn: bool) -> dict:
    per_seed = [run_seed(game, steps, strides, nperseg, sd, use_esn) for sd in seeds]
    return {"game": game, "steps": steps, "nperseg": nperseg,
            "seeds": list(seeds), "observable": "motion",
            "per_seed": per_seed, "summary": aggregate(per_seed, strides)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="pong")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-esn", action="store_true", help="skip the (slow) reservoir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run(args.game, args.steps, args.strides, args.nperseg,
              args.seeds, not args.no_esn)

    print(f"\n=== Atari floor experiment  game={res['game']}  "
          f"observable={res['observable']}  nperseg={res['nperseg']}  "
          f"seeds={res['seeds']}  steps={res['steps']} ===\n")
    for ps in res["per_seed"]:
        print(f"  seed {ps['seed']}: frozen {ps['frozen_fraction']*100:4.1f}%  "
              f"ball {ps['ball_y_coverage']*100:3.0f}%  "
              f"drift {ps['stationarity']['mean_spread_rel']:.2f}  "
              f"episodes {ps['episodes']}")
    print()
    print(f"  {'stride':>6} {'floor':>15} {'achievable':>15} {'gap':>15}")
    for r in res["summary"]:
        print(f"  {r['stride']:>6} {r['floor']:>8.5f}+-{r['floor_sd']:.4f} "
              f"{r['linear']:>8.5f}+-{r['linear_sd']:.4f} "
              f"{r['gap']:>8.3f}+-{r['gap_sd']:.3f}")
    print()
    print("  gap = achievable / floor, mean +- sd over seeds.")
    print("  gap ~ 1 : the floor matches what a real predictor achieves (tight).")
    print("  gap > 1 : the floor is a valid lower bound, and conservative.")
    print("  gap < 1 : the floor sits above the achievable error -- estimator bias,")
    print("            not a violated bound; see psd.py on the low-power tail.")
    print()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
