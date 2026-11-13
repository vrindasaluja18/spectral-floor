# Option A — Reading list

> *"How Fast Must a World Model Look? A Spectral Floor on Temporal Sampling."*
> Every arXiv ID below was checked against the arXiv API. ★ = read first.
> Author: vrinda · salujavrinda018@gmail.com · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)

Read in this order. Each tier only needs the ones above it.

---

## Tier 0 — The tool (read this first, it is the whole paper)

You need *exactly one* classical result, and you need to own it completely.

★ **Kolmogorov, "Interpolation and Extrapolation of Stationary Random Sequences"** (1941).
   The original. Hard to find in English; skip unless you want the history.
   Use instead:

★ **Brockwell & Davis, *Time Series: Theory and Methods*, Ch. 5** (linear prediction,
   Szegő's formula, spectral density). This is the readable, correct source. The key result:

```
eps² = exp( ⟨ log S ⟩ ) / ⟨ S ⟩
```

   where `⟨·⟩` is the mean over the band and `S` is the spectral density. This is the floor.

★ **Percival & Walden, *Spectral Analysis for Physical Applications*** (1993), Ch. 6–7.
   Welch's method, bias, variance, windowing. **You need this for Week 1** — when you
   estimate `S` from real data, the estimator's bias becomes part of your error bar.

  - Related, for the debiasing you'll likely need:
    **"Debiasing Welch's Method for Spectral Density Estimation"** — arXiv:2312.13643

**What to get out of Tier 0:** the floor formula, and how to estimate `S` honestly.

---

## Tier 1 — Multirate signal processing (the aliasing machinery)

★ **Oppenheim & Schafer, *Discrete-Time Signal Processing*, Ch. 4 (sampling) and Ch. 10
   (multirate)**. Decimation, interpolation, and the aliasing formula you need:

```
S_dec(v) = Σ_{k=0}^{s-1} S( (v + k) / s )
```

**What to get:** why decimation folds the spectrum, and the exact folding formula.

---

## Tier 2 — World models (know your objects)

You must know what a world model *is* before you can bound one.

★ **Ha & Schmidhuber, "World Models"** — arXiv:1803.10122 (2018). The origin, and the
   clearest statement of the "learn a model, then dream in it" idea.

★ **Hafner et al., "Dream to Control" (Dreamer)** — arXiv:1912.01603 (2019).
   Latent imagination; the model you will most likely measure.

★ **Hafner et al., "DreamerV3"** — arXiv:2301.04104 (2023).
   The current standard, runs on many domains, **MIT licensed, actively maintained**
   (`danijar/dreamerv3`). This is your main experimental target.

  - **"Mastering Atari with Discrete World Models" (DreamerV2)** — arXiv:2010.02193
  - **"Learning Latent Dynamics for Planning from Pixels" (PlaNet)** — arXiv:1811.04551

★ **Hansen et al., "TD-MPC2"** — arXiv:2310.16828 (2023).
   Continuous-control world model, **MIT licensed, actively maintained**
   (`nicklashansen/tdmpc2`). Your second experimental target — it has clean control-rate
   knobs.

**What to get:** which part of the model consumes the observation rate, and where a stride
would be inserted.

---

## Tier 3 — The closest prior work (cite, do not re-derive)

★ **Nhu et al., "Time-Aware World Model for Adaptive Prediction and Control" (TAWM)** —
   arXiv:2506.08441 (2025). **The single most important paper for your positioning.**

   It already says: *"the optimal sampling rate depends on a system's underlying dynamics"*
   and conditions on `Δt`, training over many `Δt` values.

   **Your delta:** they *search* over `Δt`; you give a *closed-form selection rule* and
   measure the *irreducible* floor. Cite it in your first paragraph and your related-work
   section. Do not claim their idea.

  - Repo: `github.com/anh-nn01/Time-Aware-World-Model` (small, ~15 stars, has a `tawm/` module)

- **"Learning World Models With Hierarchical Temporal Abstractions"** — arXiv:2404.16078
- **"HiPolicy: Hierarchical Multi-Frequency Action Chunking"** — arXiv:2604.06067
  (the action-side version of the same question)

**What to get:** exactly how TAWM chooses `Δt`, so your rule can be compared to theirs.

---

## Tier 4 — Rate, compression, and evaluation

- **"Network-Efficient World Model Token Streaming"** — arXiv:2605.09886
  (rate–distortion framing for world-model tokens; adjacent, not identical)
- **"AceVFI: Survey of Video Frame Interpolation"** — arXiv:2506.01061
  (the "missing frames" problem as a solved field; cite so you don't reinvent it)
- **V-JEPA 2.1** — arXiv:2603.14482 (the model from the passage; fixed frame skip)
- **WBench** — arXiv:2605.25874 (world-model evaluation practice)

---

## Tier 5 — Why learned models *don't* hit the floor (the honest half)

Your floor is a **linear** floor. Neural networks are nonlinear and can beat it. You must
know *when*, or a reviewer will kill you with one counterexample.

★ **Rahaman et al., "On the Spectral Bias of Neural Networks"** — arXiv:1806.08734 (2018).
   Networks fit low frequencies first. This is *why* your floor may be loose, and it is
   also the mechanism you can cite for why the residual error is structured.

- **"Spectral Bias in Practice"** — arXiv:2110.02424
- **"Addressing Spectral Bias by Multi-Grade Deep Learning"** — arXiv:2410.16105

**What to get:** the difference between "irreducible by any linear predictor" (your floor)
and "irreducible by any predictor at all" (a much harder question). **Your paper should
claim the former and measure the gap to the latter.** That gap is the contribution.

---

## Tier 6 — Statistics, because you must not fool yourself

★ **Wilcox, *Introduction to Robust Estimation and Testing*** (or his papers).
   You have a small number of seeds. Do not report a mean without an interval.

- **Benjamini–Hochberg** for multiple comparisons across strides/domains.
- **Paired tests** across seeds (same seed, different stride) — much more powerful than
  independent-sample tests. You already have `rigorous_validation` in `credence`; reuse it.

**What to get:** report intervals and paired tests, not bare means. This is what separates a
credible measurement paper from a blog post.

---

## The one-paragraph summary of the reading

> The **floor** is Kolmogorov–Szegő (Tier 0). The **aliasing** that makes stride cost
> anything is multirate DSP (Tier 1). The **objects** are DreamerV3 and TD-MPC2 (Tier 2).
> The **prior art you must beat and cite** is TAWM (Tier 3). The **adjacent work** is
> token rate-distortion and frame interpolation (Tier 4). The **honest limitation** is that
> your floor is linear while models are nonlinear (Tier 5) — measure the gap. And the
> **statistics** must be paired and interval-based (Tier 6).

---

## Minimum viable reading (if you only have a week)

1. Brockwell & Davis Ch. 5 — the floor formula. **(½ day)**
2. Oppenheim & Schafer §10.1–10.3 — decimation + aliasing formula. **(½ day)**
3. DreamerV3 paper §2–3 — what you're measuring. **(½ day)**
4. TAWM paper, all of it — your positioning. **(½ day)**
5. Rahaman et al. — the counterexample you must survive. **(¼ day)**
6. Welch / Percival & Walden Ch. 6–7 — how to estimate `S`. **(½ day)**

That is enough to start Week 1.
