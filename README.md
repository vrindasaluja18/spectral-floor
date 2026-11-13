# spectral-floor

**How fast must a world model look?** A computable information-theoretic floor on the temporal
sampling rate, derived from the data's power spectrum *before any training*.

![A world model's spectrum is a measure on a circle — and stride folds it](docs/figures/spectral-floor.png)

*The whole thesis in one image. **A** the spectrum is a measure on the unit circle, and it is
even, `S(f) = S(1−f)`, because the process is real. **B** decimating by stride `s` is the
covering map `z ↦ z^s`: the `s` shifted copies sum to the fold, which is the aliasing identity.
**C** the symmetry is the computation — expanding a one-sided spectrum by interpolation
*clamps* it and silently biased the floor 46% low at `s=16`; mirroring is exact. **D** the floor
`eps² = exp(⟨log S⟩)/⟨S⟩` is fixed by `S` alone, and a best-in-class predictor lands within 11%
of it at every stride. Regenerate with `python3 scripts/make_figure.py`.*

## Results

At stride `s` a world model sees `R/s` samples per second and, by Nyquist, nothing above
`R/(2s)` Hz. The table is the thesis in one shot: the theoretical floor (Kolmogorov–Szegő on the
*aliased* spectrum) against what a best-in-class linear predictor actually achieves, on an AR(1)
process with `a = 0.99` (400k samples, held-out evaluation).

| stride `s` | floor (estimated S) | floor (analytic S) | best linear predictor | gap to floor |
|---|---|---|---|---|
| 1 | 0.02019 | 0.01990 | 0.02162 | 1.09× |
| 2 | 0.04004 | 0.03940 | 0.04299 | 1.09× |
| 4 | 0.07902 | 0.07726 | 0.08430 | 1.09× |
| 8 | 0.15336 | 0.14854 | 0.16321 | 1.10× |
| 16 | 0.28972 | 0.27502 | 0.30550 | 1.11× |

**Key finding: the floor binds.** A best-in-class linear predictor sits within 11% of the
theoretical floor at every stride, and the gap is *constant* — it does not grow with stride. The
floor is therefore usable as a **design law**, not merely a bound.

The law exposes the trade-off, computable before training anything:

| `a` (corr. time) | s=2 | s=4 | s=8 | s=16 | state bytes/s saved at s=16 |
|---|---|---|---|---|---|
| 0.90 (~10 steps) | 1.81× | 3.00× | 4.29× | 5.08× | 16× |
| 0.99 (~100 steps) | 1.98× | 3.88× | 7.46× | **13.82×** | 16× |
| 0.995 (~200 steps) | 1.99× | 3.94× | 7.73× | 14.86× | 16× |

At `a = 0.99`, stride 16 makes the best-possible error **13.8× worse** while saving **16×** on
state bytes per second. Stride is not a vibe; it is a design choice with a predictable accuracy
cost, and that cost curve is computable up front.

### Honest negative, on the first screen

The intuitive claim *"the fraction of power above the model's Nyquist frequency lower-bounds the
prediction error"* is **false**. A finite sum of sinusoids has all its power below 10 Hz, yet a
stride-2 model predicts it essentially perfectly (rel. MSE ≈ 0.000000) — because such a process
is deterministic and finite-order, so the high frequencies are *determined* by the low ones
through a linear recurrence. Aliasing only destroys information when the high-frequency content
is **independent**. The right object is not "power above Nyquist" but "power that is **not
linearly predictable**" — which is exactly what Kolmogorov–Szegő measures. See
`docs/research-plan-sampling-rate.md` §2.1 for the counterexample.

## The maths

*From basics — that is the point.* A **stationary** process is one whose statistics do not change
with time. Its **power spectral density** `S(f)` says how much variance lives at each frequency
`f`. Prediction asks: given the past, how much of the next sample is left over? For a process
with spectrum `S`, the best any *linear* predictor can do — at any order, with any amount of
data — is fixed by `S` alone. Nothing about the model matters: the floor is a property of the
process, not of the architecture.

### The floor

For a stationary process with power spectral density `S(f)`, the optimal one-step-ahead relative
MSE of **any** linear predictor — any order, any data — is

```
eps² = exp( ⟨ log S ⟩ ) / ⟨ S ⟩
```

(Kolmogorov 1941; Szegő 1920). `⟨·⟩` is the mean over the band. This is exact and classical.
For `AR(1)` with coefficient `a`, it gives `1 − a²` to 8 digits.

### Decimation

Observing at stride `s` decimates the process. The spectrum folds:

```
S_dec(v) = (1/s) Σ_{k=0}^{s-1} S( (v + k) / s )
```

The floor of a stride-`s` world model is `szego(S_dec)`.

### The closed form

For a first-order process the folding collapses to

```
eps²(s) = 1 − a^(2s)
```

verified exactly (max error 7.6e-8 over 49 `(a, s)` pairs).

### The design law

Going from stride 1 to stride `s` multiplies the irreducible error by
`(1 − a^(2s)) / (1 − a²)`. This is the second table above, in closed form.

## Open question: the other half

This project answers the *temporal* half of one question:

> **How fast must a world model look?**

The *state* half is open, and it is the one I want next:

> **How many latent dimensions must a world model keep?**

Same shape of answer: a computable bound from the environment's own properties, plus a measured
gap between the bound and what a trained model actually uses. The temporal half has an
80-year-old closed form (used here). The state half is exactly what **Takens' theorem** answers
for delay embeddings — and the obvious search phrase returns **0 arXiv hits**, so the framing is
new. The link to *"we lose dimensions in latent space and have to maintain KV caches"* is
direct: KV-cache size is the state-axis cost, the way bytes-per-second is the time-axis cost.

> *"We ask how fast a world model must look and how many dimensions it must keep, and show both
> are computable from the environment before training."*

See `docs/research-findings-2026-11-13.md` §4 (Proposal B, `latent-floor`) and
`docs/research-plan-sampling-rate.md`.

## Reproduce

```bash
pip install numpy pytest
python3 -m pytest tests/ -q                       # 71 passed
PYTHONPATH=. python3 -m experiments.run_floor_experiment --source ar1 --a 0.99
```

Output:

```
  stride  floor(est)  floor(true)   achieved  gap(true)
  ------------------------------------------------------
       1     0.02019      0.01990    0.02162      1.09x
       2     0.04004      0.03940    0.04299      1.09x
       4     0.07902      0.07726    0.08430      1.09x
       8     0.15336      0.14854    0.16321      1.10x
      16     0.28972      0.27502    0.30550      1.11x

  max gap = 1.11x  ->  floor is TIGHT (use it as a design law)
```

## The trap that is easy to fall into

**Expanding a one-sided spectrum to the full band is not interpolation.**

`welch_psd` returns the spectrum on `[0, 0.5]`. The aliasing identity folds the whole `[0, 1)`
band, so the upper half is needed. The obvious move — `np.interp(u, grid, S_hat)` with `u`
spanning `[0, 1)` — is wrong: `np.interp` **clamps** every frequency above 0.5 to `S(0.5)`,
turning the upper band into a constant. Because the folding step sums shifted copies of `S`,
that error compounds with stride. On AR(1) at `a = 0.99` it made the floor look **46% too low at
`s = 16`** (0.148 against the true 0.275) — which would have made a perfectly good model look
artificially bad.

A real-valued stationary process has an **even** density, `S(f) = S(1 − f)`, so the correct
expansion is a *mirror*, not an interpolation. `two_sided_psd` does this exactly, and the
estimator now tracks the analytic floor to within ~5% at every stride. Both the wrong and the
right constructions are pinned in the test suite.

An earlier version of this project attributed the discrepancy to Welch estimator bias. That was
wrong: the estimator was fine and the resampling was the bug. `scripts/diagnose_resampling.py`
reproduces both constructions on an exactly-known spectrum, and `docs/FINDINGS-resampling.md`
writes up the measured error at each stride.

## What is in here

```
spectral_floor/
  psd.py     Welch periodogram, one-sided -> two-sided expansion, analytic AR(1) PSD
  floor.py   Kolmogorov-Szego, the aliasing identity, the closed form, the design law
experiments/
  run_floor_experiment.py   the Week-1/Week-2 go/no-go: floor vs achieved error
tests/       71 tests, all passing
docs/        the maths, the explainer, the reading list, the strategy
  figures/spectral-floor.png   the one-image thesis (regenerate: scripts/make_figure.py)
scripts/     diagnostics + the figure generator
```

### What is in `docs/`

| File | What it is |
|---|---|
| `research-plan-sampling-rate.md` | The research plan: is this paper-shaped, and where is the gap? |
| `reading-list-option-a.md` | Annotated reading list for the programme. |
| `strategy-project-or-pr.md` | Decision doc: project vs PR, and where. |
| `FINDINGS-resampling.md` | The one-sided-spectrum clamping bug: measured error, fix, tests. |
| `research-findings-2026-11-13.md` | What is open in the field, and where to work next. |
| `aliasing_floor_demo.py` | Runnable demo: aliasing, the fold, and the floor. |
| `decoder-explained.md` / `.html` | Companion notes for the animated decoder explainer. |
| `diffusion-maths-and-signals.md` | The diffusion-decoder maths, signals-first. |
| `verify_decoder_maths.py` | Numeric checks for the decoder notes. |

## Status and roadmap

- [x] The floor formula, verified against AR(1) theory
- [x] The aliasing identity, verified (power preservation)
- [x] The closed form `eps²(s) = 1 − a^(2s)`, verified exactly
- [x] Experiment runner: floor vs achieved error, with the aliasing gap
- [x] 71 tests
- [ ] **Week 1:** estimate `S` from a *real* environment (Atari / DMControl). Does the floor
      still bind with a measured spectrum?
- [ ] **Week 2:** train a small world model at two strides. Measure the gap. **Go/no-go.**
- [ ] **Week 3+:** 3 environments, paired tests, interval estimates, write-up

## Limitations (stated honestly)

- The floor is **linear**. Neural networks are nonlinear and may beat it. The contribution is to
  *measure the gap* between the linear floor and what a learned model achieves — not to claim
  the floor is unbeatable by any model.
- The closed form is exact only for first-order dynamics; real environments need the numeric
  Szegő value from a measured spectrum.
- Everything so far is simulated. The real test is Week 1.

## Positioning

> Prior work treats temporal resolution as a hyperparameter to be tuned. We show it has a
> computable information-theoretic floor, and use it to choose the resolution.

## References

| Work | Role |
|---|---|
| Kolmogorov (1941); Szegő (1920) | the floor formula |
| Brockwell & Davis, *Time Series*, Ch. 5 | readable source for the floor |
| Oppenheim & Schafer, *DTSP*, Ch. 10 | multirate / the aliasing identity |
| Percival & Walden (1993) | honest PSD estimation |
| Hafner et al., DreamerV3 — arXiv:2301.04104 | the world model |
| Hansen et al., TD-MPC2 — arXiv:2310.16828 | second world model |
| **Nhu et al., TAWM — arXiv:2506.08441** | **closest prior work** |
| Rahaman et al. — arXiv:1806.08734 | why the floor may be loose (spectral bias) |

Full annotated reading list: `docs/reading-list-option-a.md`.

## Contact

vrinda · salujavrinda018@gmail.com · [LinkedIn](https://www.linkedin.com/in/vrinda-saluja-rl) · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)

## Licence

MIT.
