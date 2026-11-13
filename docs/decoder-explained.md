# The Decoder, Explained — maths first, compute as a budget

> Companion notes for the animated explainer `decoder-explained.html`.
> Source passage: V-JEPA 2.1 world model → diffusion decoder → Cosmos tokenizer latents.
> Author: vrinda · salujavrinda018@gmail.com · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)

The whole topic reduces to one claim:

**An encoder is many-to-one, so information is genuinely destroyed. A decoder cannot recover it.
It can only (a) blur, if deterministic, or (b) invent, if generative — and inventing costs compute.**

Everything below is a consequence of that.

---

## 1. Setup and notation

A video is a sequence of RGB frames

```
x_t ∈ R^{3×H×W},   t = 1 … T,   rate = 20 fps
```

**V-JEPA encoder** `E`, with frame skip `s = 4`:

```
z_i = E(x_{1 + 4(i-1)}) ∈ R^d,    i = 1 … floor(T/4)
```

so `z` lives at `20/4 = 5 Hz`. This is the world model's entire universe: it predicts in
`z`-space and never touches pixels.

**Cosmos tokenizer** `C`, no frame skip:

```
c_t = C(x_t) ∈ R^m,   t = 1 … T   (20 Hz)
```

**Decoder** `p_θ`:

```
c_{1:T} ~ p_θ( · | z_{1:T/4} )
```

i.e. *given 5 Hz features, produce a 20 Hz latent stream.* The rate mismatch `20 ≠ 5` is the
whole story.

---

## 2. Where the information dies

`z_i` is a deterministic function of exactly one frame, `x_{1+4(i-1)}`. So for the three frames
strictly between `z_i` and `z_{i+1}`:

```
I( x_{1+4(i-1)+k} ; z_i , z_{i+1} ) ≈ 0,     k = 1, 2, 3
```

Mutual information is ~0 because `E` never observed those frames. This is not a limitation of the
decoder — it is a property of the data. No function, however large, inverts information that was
never in the input.

Therefore the correct target is not a reconstruction but a **conditional sample**:

```
x_{1+4(i-1)+k} ~ p( x | x_{1+4(i-1)}, x_{1+4i} )
```

That distribution is genuinely non-degenerate: motion between two observed frames has many
plausible in-betweens.

**This is the entire reason a generative model is used.**

---

## 3. Why a deterministic decoder blurs

Minimise squared error over a decoder `f`:

```
min_f  E ‖ x − f(z) ‖²      ⟹      f*(z) = E[ x | z ]
```

The minimiser of MSE is the conditional expectation — a weighted average over all plausible
in-between frames. So a deterministic decoder outputs the **blur**.

A diffusion model instead learns `p_θ(x | z)` and **samples**. Bias is traded for variance:
sharper output, but the sample is one plausible world, not the true one.

---

## 4. The diffusion decoder, mathematically

Forward process (fixed, no learning):

```
x^τ = √ᾱ_τ · x⁰ + √(1 − ᾱ_τ) · ε,     ε ~ N(0, I),    τ = 0 … N
```

Reverse process (learned), conditioned on the features `z`:

```
L(θ) = E_{x,z,ε,τ} ‖ ε − ε_θ( √ᾱ_τ x⁰ + √(1 − ᾱ_τ) ε , τ , z ) ‖²
```

**Spatially conditioned** means `z` is injected patch-aligned: each output patch attends to the
feature at the same grid location. Conditioning is a *local* function of space, not one global
vector — which is what keeps the invented motion consistent across the frame.

Sampling needs `N` sequential passes (typically `N ~ 20`–`1000`). **That `N` is where the compute
goes.**

---

## 5. Compute as a budget

Let `P` = decoder params, `n` = tokens per call, `N` = denoising steps, `d` = reservoir state dim.

```
one decoder call:               FLOPs ≈ 2 · P · n · N
one second of 20 fps video:     FLOPs ≈ 5 · 2 · P · n · N = 10 P n N
linear readout, one step:       FLOPs = 2 d²

ratio ≈ (10 P n N) / (2 d²)
```

Plug in a realistic video-DiT: `P ≈ 3×10⁸`, `n ≈ 10³`, `N = 100`, `d = 10³`:

| Quantity | Value |
|---|---|
| decoder FLOPs / s of video | ≈ 3×10¹⁴ (300 TFLOP) |
| linear readout FLOPs / step | ≈ 2×10⁶ (2 MFLOP) |
| **ratio** | **≈ 1.5×10⁸** |
| GPU-seconds per second of video | ≈ 0.6 s @ 5×10¹⁴ FLOP/s dense |
| decoder weights (fp16) | ≈ 600 MB |
| readout weights (fp16) | ≈ 2 MB |

So generating video is roughly **break-even with watching it** on a top-tier GPU, and a generative
readout costs **~10⁸×** more per step than a linear one. State these as order-of-magnitude
estimates with their assumptions, not benchmarks.

### Where the cost lives

- **`N` multiplies everything.** Halving steps halves FLOPs *and* latency. This is why few-step
  samplers / distillation are an entire research field.
- **Latency is sequential.** The `N` steps cannot be parallelised across time, so wall-clock scales
  with `N`. FLOPs you can buy with GPUs; latency you cannot.
- **Memory** is `O(P)` weights plus per-step activations, versus `O(d²)` for a linear readout.

---

## 6. The trap

```
world model predicts:   p̂( z_{t+1} | z_t )      lives in feature space
decoder samples:        p̂_θ( x | z )            lives in pixel / latent space

"the video looks great" measures the SECOND thing, not the first.
```

The invented in-between frames come from the decoder's prior, not the world model. To separate
them: decode ground-truth features too and compare frame-by-frame, or evaluate in `z`-space.

| You can claim | You cannot claim |
|---|---|
| features are decodable at all → they carry visual structure | that predicted motion is accurate |
| decoded futures are consistent with actions | that in-between frames are correct |
| qualitative sanity | any quantitative world-model score |

---

## 7. The mapping to reservoir computing / your thesis

| V-JEPA pipeline | ESN / reservoir | Role |
|---|---|---|
| feature `z` | reservoir state `r` | compressed memory — the bottleneck |
| V-JEPA encoder `E` | reservoir dynamics `W, W_in` | many-to-one compression (info dies here) |
| decoder `p(x\|z)` | linear readout `W_out` | map back to observable space |
| frame skip `s=4` | fading memory / spectral radius `ρ` | how much past survives |
| conditional mean = blur | MSE readout = smooth average | same failure mode, same reason |
| diffusion, cost `10PnN` | generative readout, same ~10⁸× cost | buy sharpness with FLOPs |

The shared law, stated once:

```
a readout cannot emit bits that the state does not contain.
deterministic readout → pays in blur.
generative readout    → pays in compute.
```

"the maths is cheap; memory is precious" — this pipeline is a case study of exactly that.

---

## Assumptions behind the numbers

- Transformer forward pass ≈ `2·P·n` FLOPs (params × tokens × 2 for multiply-add).
- 20 fps video, V-JEPA frame skip 4 → 5 decoder output steps per second.
- `N = 100` denoising steps; `P = 3×10⁸`; `n = 10³` tokens per 4-frame chunk.
- H100-class dense throughput ≈ 5×10¹⁴ FLOP/s (fp16, ~50% MFU).
- fp16 weights → 2 bytes per parameter.

Change any of these and the ratio moves; the *order of magnitude* is the robust part.
