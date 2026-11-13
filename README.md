# spectral-floor

**How fast must a world model look? A computable information-theoretic floor on temporal
sampling.**

vrinda · salujavrinda018@gmail.com · [LinkedIn](https://www.linkedin.com/in/vrinda-saluja-rl) · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)

---

## The idea in one paragraph

A world model observes the world at some rate. If it uses a frame skip (stride) of `s`, it
sees `R/s` samples per second and, by Nyquist, nothing above `R/(2s)` Hz. That sounds like a
loss — but *how much* of a loss, exactly? Prior work treats the observation rate as a
hyperparameter to be tuned (e.g. **TAWM**, arXiv:2506.08441, which conditions on `Δt` and
trains over many values). This project instead asks for the **floor**: the best possible
prediction error at a given stride, computable from the data's power spectrum *before any
training*. The answer is the classical **Kolmogorov–Szegő** formula applied to the *aliased*
spectrum, and it yields a closed form for first-order dynamics:

```
eps²(s) = 1 − a^(2s)
```

Stride is therefore not a vibe. It is a design choice with a predictable accuracy cost.

---

## What is in here

```
spectral_floor/
  psd.py     Welch periodogram, one-sided -> two-sided expansion, analytic AR(1) PSD
  floor.py   Kolmogorov-Szego, the aliasing identity, the closed form, the design law
experiments/
  run_floor_experiment.py   the Week-1/Week-2 go/no-go: floor vs achieved error
tests/       69 tests, all passing
docs/        the maths, the explainer, the reading list, the strategy
```

## Quickstart

```bash
pip install numpy pytest
python3 -m pytest tests/ -q                       # 69 passed
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

**The floor binds.** A best-in-class linear predictor sits within 11% of the theoretical
floor at every stride, and the gap is *constant* — it does not grow with stride.

---

## The maths

### The floor

For a stationary process with power spectral density `S(f)`, the optimal one-step-ahead
relative MSE of **any** linear predictor — any order, any data — is

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
`(1 − a^(2s)) / (1 − a²)`:

| `a` (corr. time) | s=2 | s=4 | s=8 | s=16 | bytes/s saved at s=16 |
|---|---|---|---|---|---|
| 0.90 (~10 steps) | 1.81× | 3.00× | 4.29× | 5.08× | 16× |
| 0.99 (~100 steps) | 1.98× | 3.88× | 7.46× | **13.82×** | 16× |
| 0.995 (~200 steps) | 1.99× | 3.94× | 7.73× | 14.86× | 16× |

At `a = 0.99`, stride 16 makes the best-possible error 13.8× worse while saving 16× on
state bytes per second. **That trade-off curve is computable before training anything.**

---

## The one finding that would have fooled me

The naive claim *"the fraction of power above the model's Nyquist frequency is a lower bound
on prediction error"* is **false**. A finite sum of sinusoids has all its power below 10 Hz,
yet a stride-2 model predicts it essentially perfectly (rel. MSE ≈ 0.000000) — because such
a process is deterministic and finite-order, so the high frequencies are *determined* by the
low ones through a linear recurrence. Aliasing only destroys information when the
high-frequency content is **independent**.

The right object is not "power above Nyquist" but "power that is **not linearly predictable**"
— which is exactly what Kolmogorov–Szegő measures. See `docs/research-plan-sampling-rate.md`
§2.1 for the counterexample.

A second, smaller trap: the floor computed from a **Welch estimate** of `S` is biased, and the
bias *grows with stride* (at `s = 16` it reports 0.153 against the true 0.275). Use the
analytic spectrum when you have one. This is locked in as a regression test.

---

## Status and roadmap

- [x] The floor formula, verified against AR(1) theory
- [x] The aliasing identity, verified (power preservation)
- [x] The closed form `eps²(s) = 1 − a^(2s)`, verified exactly
- [x] Experiment runner: floor vs achieved error, with the aliasing gap
- [x] 69 tests
- [ ] **Week 1:** estimate `S` from a *real* environment (Atari / DMControl). Does the floor
      still bind with a measured spectrum?
- [ ] **Week 2:** train a small world model at two strides. Measure the gap. **Go/no-go.**
- [ ] **Week 3+:** 3 environments, paired tests, interval estimates, write-up

## Limitations (stated honestly)

- The floor is **linear**. Neural networks are nonlinear and may beat it. The paper's
  contribution is to *measure the gap* between the linear floor and what a learned model
  achieves — not to claim the floor is unbeatable by any model.
- The closed form is exact only for first-order dynamics; real environments need the
  numeric Szegő value from a measured spectrum.
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

## Licence

MIT.
