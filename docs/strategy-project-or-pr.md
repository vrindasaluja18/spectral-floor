# Option A — Project or PR? (and where)

> Decision doc for the sampling-rate research programme.
> Author: vrinda · salujavrinda018@gmail.com · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)

---

## Short answer

**Do both — but they are not equal.**

| | What | Why | Effort |
|---|---|---|---|
| **Primary** | A standalone **project/repo** (`spectral-floor`) | this *is* the research; it becomes the paper and the hiring artifact | weeks |
| **Secondary** | One **small PR** to DreamerV3 or TD-MPC2 | puts your name in a 4k-star repo; proves you can work in someone else's code | days |

**The trick that makes them one effort:** the PR should be a *byproduct* of the project.
You need a stride/time-skip config knob to run your experiments — so you build it, use it, and
upstream it. If they merge it, you get credit and adoption. If they don't, you keep it in a
fork and cite it. Either way you lose nothing.

**Do not** try to make the research itself a PR. There is no repo where "a spectral floor on
temporal sampling" is a patch. Research goes in your own repo.

---

## Why the project is primary

1. **It is the paper.** Option A needs a code release anyway. The repo *is* the artifact.
2. **It is the hiring signal.** A README with real numbers, a reproducible runner, and tests
   is exactly what a startup looks for. A merged typo fix is not.
3. **It is yours.** A PR to someone's repo is their repo. A project is your portfolio.
4. **It fits your thesis.** "accuracy vs state-bytes/second" is already your framing; this
   repo makes it quantitative.

## Why the PR is still worth doing

1. **Distribution.** `danijar/dreamerv3` has ~4,000 stars. A merged PR there is a public,
   verifiable line on your profile.
2. **Proof of collaboration.** Working inside a large unfamiliar codebase and getting a
   maintainer to accept your change is a distinct skill from solo research.
3. **You already have the pattern.** `hydrogym` and `reaf-sim` were exactly this: find a
   real bug, fix it, test it, PR it. Same playbook.
4. **It is cheap.** Days, not weeks.

---

## Where to PR — the evidence

I checked each candidate repo's actual merge behaviour (recent closed PRs):

| Repo | Stars | Licence | Maintained | External PRs merged | What gets merged |
|---|---|---|---|---|---|
| **`danijar/dreamerv3`** | ~4.0k | MIT | yes (2026-10) | **11 of 11** merged were external | small fixes: Dockerfile pins, syntax compat, bug fixes |
| **`nicklashansen/tdmpc2`** | ~1.0k | MIT | yes (2026-10) | 5 external | bug fixes **and** features (has `CONTRIBUTING.md`) |
| `anh-nn01/Time-Aware-World-Model` | ~15 | none | 2025-11 | 0 of 0 | nothing; effectively no PR path |
| `facebookresearch/vjepa2` | ~4.9k | MIT | yes | — | research code drop; PRs uncommon |

### Recommendation

**First choice: `nicklashansen/tdmpc2`.**

Reasons:
- It has a `CONTRIBUTING.md` — the maintainer *invites* contributions. That is rare and
  worth a lot.
- It merges both bug fixes *and* features, so a `time_stride` config knob has a real chance.
- It has clean control-rate handling, which is exactly the axis your research measures.

**Second choice: `danijar/dreamerv3`.**

- Higher profile (4k stars) and it merges external contributions reliably — but only *small*
  fixes. A feature PR is unlikely to land. Good for a bug fix, poor for a feature.

**Not a target:** the TAWM repo (no PR path, 0 merges) and V-JEPA 2 (research drop).
Cite them; don't PR them.

---

## The two concrete PRs to aim for

### PR candidate 1 — the stride knob (strategic, medium risk)

**Repo:** `tdmpc2`
**Title:** `feat: configurable observation time-stride`
**Content:** add a `time_stride` / `frame_skip` config option so the world model can be
trained at a reduced observation rate, with the env wrapper resampling accordingly.

**Why it's a good PR:** it is a genuine capability gap, it is *exactly* what your
experiments need, and it is small and self-contained.

**Risk:** a feature PR may be rejected as out of scope. **Mitigation: open an issue first,
ask if they want it, then PR.** This is the standard, respectful path and it converts a
likely rejection into a likely merge.

### PR candidate 2 — a real bug fix (safe, high merge probability)

**Repo:** `dreamerv3` or `tdmpc2`
**Content:** find an actual bug by *reading and running* the code. Do not invent a fix for a
non-bug.

**How to find one (your proven playbook):**
1. Clone, install, run the smallest config end-to-end.
2. Read the core files (`dreamerv3/`, `tdmpc2/`) looking for: off-by-one in slicing,
   wrong axis in a `mean`/`reshape`, dtype/device mismatches, a `no_grad` missing, an
   off-by-one in a buffer index, a seed not set, an edge case at episode reset.
3. Reproduce it as a failing test *first*.
4. Fix minimally, add the test, PR with a clear before/after.

**This is what you already did twice.** The difference here is the codebase is bigger.

---

## The order of operations

```
Week 0   Read the Tier 0-3 material (reading-list-option-a.md)
Week 1   Build the project skeleton; reproduce the floor on a real spectrum
Week 2   Train at 2 strides; measure the aliasing gap.  GO / NO-GO for the paper.
Week 3   Open the tdmpc2 issue proposing the stride knob
Week 3+  Meanwhile: hunt a real bug -> the safe PR
Week 4+  If go: 3 environments, statistics, write up
```

**Do the issue before the PR.** Every time.

---

## What NOT to do

- ❌ Don't PR a research contribution. It has no home.
- ❌ Don't PR a "fix" for code you haven't run. Maintainers spot this instantly and it
  burns the repo for you permanently.
- ❌ Don't open a PR without an issue for anything larger than a bug fix.
- ❌ Don't touch the TAWM repo (no PR path) or V-JEPA 2 (research drop).
- ❌ Don't let the PR distract from the project. The project is the deliverable.
- ❌ Don't write the paper before Week 2's measurement.

---

## Naming and packaging

**Repo name:** `spectral-floor` (short, describes the contribution, searchable).

**Structure:**
```
spectral-floor/
├── README.md                  <- explain everything; your email + LinkedIn
├── LICENSE                    <- MIT
├── pyproject.toml
├── spectral_floor/
│   ├── psd.py                 <- Welch + debiasing
│   ├── floor.py               <- Kolmogorov-Szego, aliasing
│   └── experiment.py          <- runner
├── experiments/               <- configs + results
├── docs/
│   ├── paper-draft.md
│   └── (your existing explainers)
├── tests/
└── scripts/
```

**README must contain (your stated preference):** what the project is, the maths, the
results with real numbers, how to reproduce, and `salujavrinda018@gmail.com` + your
LinkedIn link.

---

## The one-line positioning (memorise this)

> *"Prior work treats temporal resolution as a hyperparameter to be tuned. We show it has a
> computable information-theoretic floor, and use it to choose the resolution."*

That sentence is the project, the paper, and the pitch. Everything else is evidence.
