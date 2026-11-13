# Should we write a paper? — an honest assessment

> On applying **signal processing** to the sampling rate of world models.
> Author: vrinda · salujavrinda018@gmail.com · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)
>
> Companion code: `aliasing_floor_demo.py` (runs on CPU, numpy only).

---

## 0. The short answer

**Yes, there is a real gap — but not where you'd first think.** The tempting framing
("aliasing in world models!") is mostly *taken or wrong*. The framing that survives
contact with the literature is narrower and more defensible:

> **The sampling rate (stride) of a world model sets a *computable* ceiling on prediction
> accuracy. Nobody has written that ceiling down for world models, or used it to pick the
> rate.** The classical tool for this is the Kolmogorov–Szegő formula, and it does not
> appear in the world-model literature at all.

That is a paper-sized contribution *if* you build it properly. It is not a paper if you
just say "Nyquist applies to video".

---

## 1. What I found — the honest version

I searched arXiv directly. The counts matter:

| Query | Results | Meaning |
|---|---|---|
| `"world model" AND "aliasing"` | several — **all about *state* aliasing** | different problem (GUI states that look identical), not temporal/spectral |
| `"temporal downsampling" AND "world model"` | **0** | the gap |
| `"Kolmogorov-Szego" AND "prediction"` | **0** | the tool is unused in this field |
| `"Nyquist" AND "world model"` | **0** | nobody frames it this way |
| `"sampling rate" AND "world model" AND "information theory"` | **0** | nobody frames it this way |
| `"rate-distortion" AND "world model"` | 2, both *compression* (token streaming) | adjacent, not the same |

So the *vocabulary* of this problem — Nyquist, aliasing, spectral floors — is absent from
the world-model literature. That is a genuine signal.

### What is already taken (do not re-derive these)

1. **"The optimal sampling rate depends on the system's dynamics."** This is already
   stated and exploited by **TAWM** — *Time-Aware World Model for Adaptive Prediction and
   Control* (arXiv:2506.08441, 2025). It conditions on `Δt` and trains over many `Δt`.
   **Do not claim this as your idea.** It is the closest prior work and you must cite it.
2. **Rate–distortion framing of what a model can represent.** Standard, and already
   applied to world-model *token streaming* (arXiv:2605.09886).
3. **Video frame interpolation.** A whole field (see the AceVFI survey, arXiv:2506.01061).
   The "3 missing frames" problem is solved many ways; do not re-invent it.
4. **Action chunking / temporal abstraction at low control rates.** Well covered
   (e.g. HiPolicy arXiv:2604.06067).

### What is genuinely open

- **No closed-form accuracy floor as a function of stride, for world models.**
- **No measurement of how much of a world model's error is *irreducible aliasing* versus
  model error.** This is the key empirical gap: nobody separates "the model is bad" from
  "the rate forbids it".
- **No principled rule for choosing the stride**, beyond TAWM's "train on many Δt".
- **No spectral diagnostic** that tells you *before* training what a given stride costs.

---

## 2. The maths we actually derived (and verified)

This is the part that makes it a contribution rather than an observation.

### 2.1 The right tool is not "power above Nyquist"

My first guess was wrong, and the counterexample is instructive.

**Wrong claim:** *"the fraction of the signal's power above the model's Nyquist frequency
is a lower bound on prediction error."*

**Why it's wrong:** a finite sum of sinusoids has all its power below 10 Hz, yet a
stride-2 model predicts it essentially perfectly (measured rel. MSE ≈ 0.000000). The high
frequencies are **determined** by the low ones through a linear recurrence. Nothing is lost.
Aliasing only destroys information when the high-frequency content is *independent*.

### 2.2 The right tool: Kolmogorov–Szegő

For a stationary process with power spectral density `S(f)`, the best achievable one-step
relative MSE of *any* linear predictor (any order, any data, any depth) is

```
eps² = exp( ⟨ log S ⟩ ) / ⟨ S ⟩
```

where `⟨·⟩` is the mean over the band. This is classical (Kolmogorov 1941, Szegő 1920) and
exact. For a **decimated** process, evaluate it on the *aliased* spectrum:

```
S_dec(v) = Σ_{k=0}^{s-1} S_full( (v + k) / s )
```

Verified: for AR(1) it reproduces the textbook `1 − a²` to 8 digits.

### 2.3 The closed form (this is the headline result)

For a first-order process with correlation coefficient `a`, the floor after stride `s` is

```
            eps²(s) = 1 − a^(2s)
```

**Verified exactly** — max error `7.6e-8` over 49 `(a, s)` pairs. And a real predictor
reaches it (measured 0.0202 vs floor 0.0199 at `a = 0.99`, `s = 1`).

### 2.4 The design law that falls out

Going from stride 1 to stride `s` multiplies the irreducible error by
`(1 − a^(2s)) / (1 − a²)`:

| `a` (corr. time) | s=2 | s=4 | s=8 | s=16 | bytes/s saved at s=16 |
|---|---|---|---|---|---|
| 0.90 (~10 steps) | 1.81× | 3.00× | 4.29× | 5.08× | 16× |
| 0.99 (~100 steps) | 1.98× | 3.88× | 7.46× | **13.82×** | 16× |
| 0.995 (~200 steps) | 1.99× | 3.94× | 7.73× | 14.86× | 16× |

**Read the `a = 0.99` row.** Running at stride 16 makes the *best possible* error 13.8×
worse, while saving 16× on state bytes per second. That is a real trade-off curve, and it is
computable from the data spectrum **before training anything**.

> This is your thesis statement, in closed form:
> **accuracy vs state-bytes/second has a computable frontier, and stride is the knob.**

---

## 3. Three contributions, ranked by risk/reward

### Option A — the diagnostic (lowest risk, real paper) ⭐ RECOMMENDED

**Title:** *"How Fast Must a World Model Look? A Spectral Floor on Temporal Sampling."*

**Claim:** For a given environment, the Kolmogorov–Szegő floor `eps²(s)` computed from the
observation spectrum predicts the best achievable prediction error at stride `s`. Measure
this on 2–3 standard world-model benchmarks, compare to actual trained models, and show the
gap between floor and achieved error.

**Why it's a paper:** it gives practitioners a *pre-training* number for a *design choice*
they currently make by vibes. It is cheap, honest, and falsifiable.

**Experiments:**
1. Compute `eps²(s)` from the data spectrum for each environment, at `s ∈ {1,2,4,8,16}`.
2. Train a small world model at each stride. Plot achieved error vs floor.
3. Report the "aliasing gap": how much error is irreducible.
4. Show the accuracy-vs-bytes/s frontier.

**Compute:** CPU-friendly if you use small models on Atari/DMControl-style state
observations. This is realistic for you.

**Risk:** low. Even a *negative* result ("the floor is loose; learned models beat it by
exploiting nonlinearity") is publishable as a measurement.

---

### Option B — the method (medium risk, higher reward)

**Title:** *"Spectrum-Aware Strides: Choosing Temporal Resolution from the Data."*

**Claim:** pick the stride per environment by matching it to the *predictability* of the
spectrum, not by grid search. Beat a fixed stride on the accuracy-per-byte frontier.

**Why it's a paper:** it is TAWM's idea (adapt `Δt`) but with a *principled, closed-form*
selection rule rather than "train on many Δt". You must cite TAWM and beat it on
data-efficiency, not just accuracy.

**Risk:** medium. TAWM is a strong baseline and you need a real win, not a tie.

---

### Option C — the theory (highest risk)

**Title:** *"Aliasing Bounds for Learned Predictive Representations."*

**Claim:** extend the Kolmogorov–Szegő floor from linear to *learned nonlinear* predictors:
show when nonlinearity can beat the linear floor (it can — that's exactly the sinusoid
counterexample) and characterise the regimes.

**Why it's hard:** it is a theory paper. You need the right negative result and clean
conditions. The sinusoid counterexample is a genuine hook, but turning it into theorems is
months of work.

**Risk:** high. Do not start here.

---

## 4. The experiment protocol (for Option A)

```
1. Pick 2-3 environments with high-rate observations.
     Atari (60 Hz), DMControl (high-rate), or a synthetic video benchmark.
2. Estimate the observation power spectrum S(f) (Welch periodogram).
3. For each stride s, compute the aliased spectrum and the Szego floor eps2(s).
4. Train the SAME world model at each stride (identical architecture, params, data).
5. Report, per stride:
       achieved_rel_mse / floor(s)      <- the "aliasing gap"
       state bytes per second           <- the budget axis
6. Plot the frontier: achieved error vs bytes/s, with the floor overlaid.
```

**Controls (non-negotiable):**
- identical parameter count across strides (else you measure capacity, not rate);
- identical wall-clock *and* identical samples;
- report the *irreducible* floor, not just the achieved error;
- keep a negative result if the floor is loose.

**The one plot that makes the paper:** achieved error vs state-bytes/s, with the theoretical
floor drawn on top. If the floor is tight, you have a design law. If it's loose, you have a
measurement of how much nonlinearity buys — also a result.

---

## 5. Related work (cite these, and how to position)

| Work | What it does | How you differ |
|---|---|---|
| **TAWM** — arXiv:2506.08441 | conditions on Δt, trains over many Δt | you give a *closed-form selection rule*, not a search; you cite it as the closest prior |
| **AceVFI survey** — arXiv:2506.01061 | frame interpolation (the "3 missing frames") | you are not interpolating; you are bounding *prediction* accuracy |
| Rate-distortion token streaming — arXiv:2605.09886 | compression budget for world-model tokens | adjacent budget framing; yours is about *temporal rate*, not token count |
| **Kolmogorov–Szegő** (1941 / 1920) | exact linear-prediction floor | classical; **your contribution is applying it to world-model strides** |
| Spectral bias — arXiv:2110.02424 | NNs fit low frequencies first | related mechanism, different question (you bound, they explain training dynamics) |
| V-JEPA 2.1 — arXiv:2603.14482 | the model in the passage | it uses a fixed frame skip; you ask what that skip costs |

**Positioning sentence:** *"Prior work treats temporal resolution as a hyperparameter to be
tuned. We show it has a computable information-theoretic floor, and use it to select the
resolution."*

---

## 6. What to do next, concretely

**Week 1:** reproduce the demo on a *real* dataset's spectrum (one Atari game). Replace the
AR(1) toy with a measured PSD. Does the floor still bind?

**Week 2:** train a tiny world model at 2 strides on that environment. Measure the gap. This
is the go/no-go: if the floor is tight, write the paper. If not, pivot to Option B.

**Week 3:** if go, add 2 more environments and write it up as a short paper (4–6 pages).
Submit to a workshop first (NeurIPS/ICLR workshops have a low bar and fast feedback).

**Do not:** start with theory (Option C), or write the paper before Week 2's measurement.

---

## 7. Honest summary

| Question | Answer |
|---|---|
| Is there a gap? | **Yes**, and it is verifiable (0 arXiv hits on the key phrases) |
| Is it a paper? | **Yes, if** framed as a computable floor + measurement, not "aliasing lol" |
| Is it novel? | The *application* is; the *tool* (Kolmogorov–Szegő) is 80 years old |
| Is it yours to take? | Yes — it is low-compute, matches your thesis, and nobody is on it |
| Biggest risk | the floor turns out loose (nonlinearity dominates) — then it's still a measurement |
| Closest prior work | TAWM (2506.08441) — cite it early and often |
| Realistic compute | CPU for Option A; this is why it fits you |
