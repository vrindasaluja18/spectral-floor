"""
The sampling-rate / stride ceiling on world-model prediction - with real numbers.

THE HONEST STORY (including a wrong guess I had to abandon)
-----------------------------------------------------------
Guess 1 (WRONG): "the fraction of power above the model's Nyquist frequency is a lower
bound on prediction error."  Counterexample in section 4: a sum of sinusoids has all
power below 10 Hz, and a stride-2 model predicts it near-perfectly.  Reason: a finite
sum of sinusoids is a deterministic, finite-order process - the high frequencies are
*determined* by the low ones via a linear recurrence.  Nothing is lost.

Correct statement: what matters is not how much power sits above Nyquist, but how much
of the process is **not linearly predictable** from its own past at the rate you
observe.  That quantity is exact, classical, and is the Kolmogorov-Szego formula:

        eps2 = exp( <log S> ) / <S>

and for the decimated process you evaluate it on the ALIASED spectrum.

The payoff is a closed form for a first-order process:  eps2(s) = 1 - a^(2s).
That is a design law: stride s multiplies the irreducible error by a known factor.

Run:  python3 aliasing_floor_demo.py
"""

import numpy as np

W = 82
def hdr(t): print("\n" + "=" * W + "\n" + t + "\n" + "=" * W)

R = 20.0                                   # base rate of the world (Hz)
U = np.linspace(0, 1, 400001)[:-1]         # normalised frequency grid u = f/R

# ===========================================================================
hdr("1. The exact tool: Kolmogorov-Szego, validated on AR(1)")
# ===========================================================================
# AR(1):  x[n] = a x[n-1] + e[n],  S(u) = 1 / |1 - a e^{-i2pi u}|^2
# Optimal linear one-step relative MSE should be exactly 1 - a^2.
def szego(S):
    S = np.maximum(S, 1e-300)
    return np.exp(np.mean(np.log(S))) / np.mean(S)

def ar1_psd(a):
    return 1.0 / np.abs(1 - a * np.exp(-2j * np.pi * U)) ** 2

print("   a       Szego eps2      1 - a^2        match")
for a in (0.8, 0.9, 0.95, 0.99, 0.995):
    g = szego(ar1_psd(a)); t = 1 - a * a
    print(f"  {a:6.3f}   {g:.8f}    {t:.8f}     {'YES' if abs(g-t) < 1e-9 else 'NO'}")

# ===========================================================================
hdr("2. Decimation folds the spectrum; the floor follows")
# ===========================================================================
# Aliasing formula for stride s:
#       S_dec(v) = sum_{k=0}^{s-1} S_full( (v + k) / s )
def fold(S, s):
    out = np.zeros_like(U)
    for k in range(s):
        out += np.interp((U + k) / s % 1.0, U, S)
    return out

print("  AR(1) a=0.99  (correlation time ~100 samples; a real, absolutely-continuous")
print("  spectrum, unlike a sum of sinusoids):")
print("    stride s | model rate | Szego floor | 1 - a^(2s) | state KB/s @d=1024")
for a in (0.99,):
    S = ar1_psd(a)
    for s in (1, 2, 4, 8, 16):
        f = szego(fold(S, s))
        closed = 1 - a ** (2 * s)
        kbs = 1024 * 2 * (R / s) / 1024
        print(f"      {s:2d}    |  {R/s:6.2f} Hz |   {f:.6f}  |  {closed:.6f} |   {kbs:6.1f}")

# ===========================================================================
hdr("3. The closed form eps2(s) = 1 - a^(2s) holds exactly")
# ===========================================================================
print("  sweeping a and s; comparing Szego on the folded spectrum to the closed form")
maxerr = 0.0
for a in (0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999):
    S = ar1_psd(a)
    for s in (1, 2, 3, 4, 8, 16, 32):
        maxerr = max(maxerr, abs(szego(fold(S, s)) - (1 - a ** (2 * s))))
print(f"  max |Szego - (1 - a^(2s))| over 49 (a, s) pairs = {maxerr:.2e}")
print("  => EXACT.  The floor is  eps2(s) = 1 - a^(2s).")

# ===========================================================================
hdr("4. Why the naive 'power above Nyquist' bound is WRONG")
# ===========================================================================
# A sum of sinusoids is deterministic + finite-order => the Szego floor is 0.
# Its spectrum is a line spectrum, so <log S> = -infinity.  Nothing is lost.
print("  a) finite sum of 40 sinusoids (deterministic, finite-order):")
uu = np.linspace(0.01, 0.49, 40)
amp = np.sqrt(1 / (uu * R) ** 1.5)
ph = np.random.default_rng(0).uniform(0, 2 * np.pi, 40)
t = np.arange(200_000) / R
xdet = np.sum(amp[:, None] * np.cos(2 * np.pi * uu[:, None] * R * t[None, :] + ph[:, None]), axis=0)
xdet = (xdet - xdet.mean()) / xdet.std()

def rel_mse(x, s, Wwin=64):
    xc = x[::s]
    n = len(xc) - Wwin - 1
    X = np.lib.stride_tricks.sliding_window_view(xc, Wwin)[:n]
    y = xc[Wwin:Wwin + n]
    cut = int(0.7 * n)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-9
    Xtr = (X[:cut] - mu) / sd
    Xte = (X[cut:] - mu) / sd
    A = Xtr.T @ Xtr + 1e-8 * np.eye(Wwin)
    w = np.linalg.solve(A, Xtr.T @ y[:cut])
    return np.mean(((Xte @ w) - y[cut:]) ** 2) / np.var(y[cut:])

for s in (2, 4, 8):
    print(f"     stride {s}: measured rel.MSE = {rel_mse(xdet, s):.6f}")
print("     -> deterministic high frequencies are RECOVERABLE. Naive bound: WRONG.")

print("\n  b) AR(1) a=0.99 (absolutely continuous spectrum, real innovation):")
xr = np.zeros(200_000); e = np.random.default_rng(1).normal(0, 1, 200_000)
for i in range(1, len(xr)):
    xr[i] = 0.99 * xr[i - 1] + e[i]
xr = (xr - xr.mean()) / xr.std()
S = ar1_psd(0.99)
print("     stride s | measured rel.MSE | Szego floor | 1-a^(2s)")
for s in (1, 2, 4, 8):
    m = rel_mse(xr, s, Wwin=256)
    print(f"        {s:2d}    |     {m:.5f}      |   {szego(fold(S,s)):.5f}   | {1-0.99**(2*s):.5f}")
print("     -> the floor BINDS when the process has a genuinely continuous spectrum.")

# ===========================================================================
hdr("5. The design law, and what it means for a world model")
# ===========================================================================
print("  For a first-order process, going from stride 1 to stride s multiplies the")
print("  irreducible prediction error by  (1 - a^(2s)) / (1 - a^2).")
print()
print("   a     | s=1   | s=2   | s=4   | s=8   | s=16  | cost saved at s=16")
for a in (0.9, 0.99, 0.995):
    base = 1 - a * a
    row = [f"{(1-a**(2*s))/base:5.2f}x" for s in (1, 2, 4, 8, 16)]
    print(f"  {a:5.3f}  | " + " | ".join(row) + " |     16x")
print()
print("  READING: if your dynamics have a=0.99 (correlation ~100 steps), running the")
print("  world model at stride 16 makes the best-possible error 13.8x worse - but saves")
print("  16x on state bytes/s.  THAT is the trade-off curve, and it is computable")
print("  from the data spectrum BEFORE training anything.")
print()
print("  => accuracy-vs-state-bytes/s is not a vibe. It has a closed form for AR(1),")
print("     and a computable Kolmogorov-Szego value for any measured spectrum.")
