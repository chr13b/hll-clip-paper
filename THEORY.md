# Theory of the stream-length clip

This note derives, in closed form, the bias, variance, and RMSE of the clipped
estimator `min(n_hat, L)` under the standard Gaussian model of a cardinality sketch,
and shows the derivations reproduce the measurements (verify with
`python3 phase1/theory_check.py`). Notation: `phi`, `Phi` are the standard normal
pdf/cdf, `Phibar = 1 - Phi`.

## 0. Setup

All modern cardinality estimators (classic HLL, HLL++, HIP/martingale, CPC) are, to
leading order in `1/sqrt(m)`, approximately unbiased with Gaussian relative error:

    n_hat = n (1 + sigma Z),   Z ~ N(0,1),   sigma = c / sqrt(m),

where `m` is the number of registers/buckets and `c` is an estimator-specific
constant (classic HLL `c ≈ 1.04`; HIP `c ≈ 0.83`; CPC lower still). The system also
knows `L`, the total number of inserted items (with duplicates); the true distinct
count satisfies `n <= L` exactly. Write `d = L/n >= 1` (the **duplication factor**),
and define the **standardized clip threshold**

    a = (L - n) / (sigma n) = (d - 1) / sigma.

`a` is the number of standard errors by which the wall `L` sits above the truth.

The clip is `n_hat_clip = min(n_hat, L)`. In standardized terms the clip caps `Z` at
`a`:

    n_hat_clip - n = n sigma * min(Z, a).

Everything below is a moment of the **right-censored standard normal** `min(Z, a)`.

## 1. Never-worse (pointwise, distribution-free, exact)

**Claim.** For every realization, `|min(n_hat, L) - n| <= |n_hat - n|`, with strict
inequality whenever `n_hat > L`.

**Proof.** `n <= L`. If `n_hat <= L`, the clip is inactive and the errors are equal.
If `n_hat > L >= n`, then `min(n_hat, L) = L` and `n <= L < n_hat`, so
`0 <= L - n < n_hat - n`; the clipped point is strictly closer to `n`. ∎

This needs no distributional assumption — it holds for any estimator, any hash, any
stream. It immediately gives `MSE(clip) <= MSE(raw)` and the same for any monotone
loss of `|error|`. The Gaussian model below only quantifies *how much* is gained; the
"never hurts" guarantee is unconditional. (Measured: zero violations in all 437
pooled cells across the six experiments.)

## 2. Closed-form moments of the clipped estimator

Using the standard right-censoring identities for `Z ~ N(0,1)`:

    E[min(Z,a)]   = -(phi(a) - a Phibar(a))
    E[min(Z,a)^2] = Phi(a) - a phi(a) + a^2 Phibar(a)

we get (relative to `n`):

    bind(a)        = P(n_hat > L) = Phibar(a)
    rel_bias(a)    = sigma * E[min(Z,a)]   = -sigma (phi(a) - a Phibar(a))   <= 0
    MSE_ratio(a)   = MSE(clip)/MSE(raw)    = Phi(a) - a phi(a) + a^2 Phibar(a)
    RMSE_gain(a)   = 1 - sqrt(MSE_ratio(a))
    var_ratio(a)   = Var(clip)/sigma^2 n^2 = MSE_ratio(a) - (phi(a) - a Phibar(a))^2

The clip trades a small **downward bias** for a larger **variance** cut: it deletes
the upper tail (impossible region), which removes variance but pulls the mean below
`n`. `MSE_ratio` is monotone increasing from `1/2` (at `a=0`) to `1` (as `a -> inf`),
so the gain is largest on duplicate-free streams and vanishes as duplicates grow —
matching intuition and the data.

## 3. The duplicate-free headline (a = 0, L = n)

When the stream is duplicate-free, `L = n`, `a = 0`, and the clip caps the estimate
at the truth itself. Then:

    MSE_ratio(0) = 1/2                      => RMSE_gain = 1 - 1/sqrt(2) = 29.29%
    rel_bias(0)  = -sigma/sqrt(2 pi)        (= -0.399 sigma)
    var_ratio(0) = 1/2 - 1/(2 pi) = 0.3408  => raw variance cut by 2.93x

**This is the paper's central number and it is parameter-free.** Half of the raw
estimator's squared-error mass lives in the (impossible) overestimate tail; deleting
it scales RMSE by exactly `1/sqrt(2)`. This is the "~sqrt(2) one-sided error"
intuition, made exact.

| quantity | predicted | measured |
|---|---|---|
| RMSE gain at d=1.0 | **29.29%** | 30.04% ± 3.90% over the 104 exp1-4 cells, 29.99% over all 107 swept cells (20.1–40.2) |
| rel bias at d=1.0 (m=1024) | **−0.0130** | −0.0122 to −0.0130 (ideal arm) |
| variance reduction | **2.93×** | — (consistent with the bias/RMSE split) |

The measured gain sits ~0.75 pp **above** 29.29% on average. The direction is the skew: the
true HLL error law is mildly right-skewed (`gamma1 ≈ 0.12`), which raises the d=1 gain by
≈1 pp (30.4%, see the second-order section). The finite-`m` bias is small and of EITHER
sign (−0.11σ to +0.03σ across the cells, ≈0 on average over exp1–4) and moves the gain by
≈0.5 pp per 0.01σ in its own direction — which is why PyPI (24–27%, bias ≈ −0.07σ) and
UltraLogLog (26–29%, ≈ −0.03σ) land *below* 29.3%. The Gaussian value is the zero-bias,
zero-skew baseline, not a lower bound: 10 of the 20 calibrated d=1 cells fall under it.

The 29.29% prediction is *estimator-agnostic* — it follows only from approximate
normality of the relative error, not from any HLL-specific detail. Accordingly the
measured d=1.0 gain holds across all five families tested (classic FFGM, HLL++,
HIP/martingale 28.5–31.4%, CPC 28.3–30.8%, UltraLogLog 25.8–29.5%, all bracketing the prediction), and on a
real workload: the production DataSketches HLL (lg_k=14) over the 827,798-name PyPI
namespace overestimates by +0.66%, which the clip corrects to +0.00%, with 24–27%
RMSE reduction on real keys (exp5).

### Relation to prior art

This makes precise the qualitative claim asserted without proof in **Pettie, Wang &
Yin (2021), Remark 2** — "`min{λ̂_i, i}` is never worse ... and has a constant factor
lower variance." The constant is `1/2 - 1/(2π) ≈ 0.341` (a 2.93× variance reduction),
and the bias they noted is exactly `−sigma/sqrt(2π)`. Their remark is a three-line
aside with no derivation; §2–§3 here supply the closed forms and the agreement with
data.

## 4. The value window and the scaling collapse (a > 0)

Because every standardized quantity in §2 is a function of `a = (d−1)/sigma` **alone**,
the gain curves for all `m` and all `d` must collapse onto a single universal curve
when plotted against `a`. We measured this across `m ∈ {256…16384}` and a
fine `d`-grid: the 80 pooled cells lie on the theoretical `RMSE_gain(a)` curve with
**RMS deviation 1.96 pp** (mean deviation +0.24 pp). The collapse is the theory's
sharpest qualitative prediction and it holds.

Solving `RMSE_gain(a) = g` for the window edges:

| target gain | threshold a* | applicability rule |
|---|---|---|
| 10% | 1.17 | duplicate fraction `(d−1) ≲ 1.17 sigma ≈ 1.22/sqrt(m)` |
| **5%** | **1.57** | `(d−1) ≲ 1.57 sigma ≈ 1.63/sqrt(m)` |
| 1% | 2.28 | `(d−1) ≲ 2.28 sigma ≈ 2.37/sqrt(m)` |

Measured 5%-edge `(d*−1)/sigma` ranged 1.42–2.01 (mean ≈ 1.7) — the theory value 1.57
sits inside the band, measured slightly higher for the same skew reason. **Practical
takeaway for the paper:** the clip is worth ≥5% RMSE precisely while the fraction of
the stream that is duplicates stays below roughly `1.6/sqrt(m)`. With `m = 16384`
(BigQuery/Redis default precision territory) that is ≈1.3% duplicates; with `m = 256`
it is ≈10%. This is the honest, quantitative "when does it bind" characterization this needs, replacing any blanket "~30%" claim.

This narrow-window prediction at high precision is borne out on real data: the PyPI
workload at `m = 16384` (σ ≈ 0.0055) puts the theoretical 5%-edge at `d ≈ 1.009`, and
the measured real-data gain is ~27% at `d = 1.0`, ~2% at `d = 1.01`, and 0 by
`d = 1.02` (exp5) — i.e. at deployment precision the hard clip pays off only on the
most nearly-distinct streams. This is precisely the regime where the soft-information
TPM variant (§5) earns its keep, extending the useful range to a few-percent
duplication (14–16% at `d = 1.01` on the same real data).

## 4b. Second order: bias and skew make the theory exact

The first-order (Gaussian) curve `1 - sqrt(g(a))` leaves a small positive residual: the
measured gain runs ~0.2–0.8 pp above prediction, and individual cells scatter by up to
~2 pp. The clip's gain is, exactly, a *functional of the error distribution* —
`gain = 1 - sqrt(E[min(r,w)^2]/E[r^2])` with `r` the relative error and `w = d-1` — so the
residual must be a higher-moment effect. It is, and only the second and third moments
matter.

**Skew correction (closed form).** Model the standardized error with a first-order
Edgeworth (Gram–Charlier) expansion of skewness `gamma1`:
`f(z) = phi(z)[1 + (gamma1/6)(z^3 - 3z)]`. Carrying the right-censored moment through
(integrals of `z^k phi` via the recurrence `M_k(a) = a^{k-1}phi(a) + (k-1)M_{k-2}(a)`)
gives a clean correction to the MSE ratio:

    g2(a) = g(a) - (gamma1/3) (a^2 + 1) phi(a),         and    gain2(a) = 1 - sqrt(g2(a)).

At `a = 0` this is `g2(0) = 1/2 - gamma1/(3 sqrt(2 pi))`: **positive skew makes the clip
delete *more* than half** the error mass, raising the duplicate-free gain above 29.29%.
The HLL relative error is mildly right-skewed (measured `gamma1 ≈ 0.12` at `d = 1`), so the
second-order theory predicts a duplicate-free gain of **30.4%**, against the measured
~30.0% — versus the first-order 29.29%. The correction is largest at small `a` and
decays as `(a^2+1)phi(a)`, the shape of the residual once the per-cell bias is removed.

**Moment completeness (empirical).** Skew and bias jointly fix the systematic offset; the per-cell
scatter is the finite-`m` *bias* `beta` (which shifts the clip threshold to
`a' = (w - beta)/sigma`). Folding both measured moments `(beta, sigma, gamma1)` into the
censored-moment integral predicts the measured clip gain across all 80 swept cells to an
**RMS of 0.165 pp** (mean +0.015 pp), down from **1.835 pp** for the zeroth-order
Gaussian — a tenfold collapse of the residual (`skew_theory.py`). In other words, the
clip's value is determined, to under 0.2 pp, by just the first three moments of the
sketch's error law; no fourth-order (kurtosis) term is needed. The truncated-moment
framework is, for practical purposes, exact.

## 5. The truncated-posterior estimator (soft information)

The hard clip uses `L` as a wall. The **truncated-posterior-mean (TPM)** estimator
uses it as soft prior knowledge: model `n_hat | n ~ N(n, (sigma n)^2)` with a prior
`pi(n)` supported on `[1, L]`, and report the posterior mean

    n_TPM = E[n | n_hat]  =  ∫_1^L n * N(n_hat; n, (sigma n)^2) pi(n) dn  / (normalizer).

This is the Bayes estimator under squared-error loss, so it is the natural extension
once we accept `L` as information. Three findings:

1. **Calibration is essential (and was one wrinkle).** With the *nominal*
   HLL `sigma = 1.04/sqrt(m)`, TPM can be worse than raw on bias-corrected HLL++,
   whose true error is smaller than nominal — the posterior then over-shrinks. Using a
   `sigma` calibrated to the sketch's actual error (a held-out, one-time measurement)
   fixes this. With calibrated `sigma`, TPM is neutral at `d=1.0` and **dominates the
   hard clip in the soft window**: on the HIP/CPC sketches at `d=1.02` it delivers
   20–25% RMSE reduction versus the hard clip's 5–17% (experiment 3).

2. **Prior sensitivity is mild.** Over the family {log-flat `1/n`, flat `1`,
   `1/sqrt(n)`}, the posterior-mean gain moves by at most 1.7 pp over the `d` tested
   (`theory_check.py` §5). The estimator is dominated by the likelihood and the `[1,L]`
   truncation, not the prior shape; log-flat (scale-invariant) is the principled
   default.

3. **Behavior as d grows.** As `L` moves above the bulk of the likelihood
   (`a` large), the truncation becomes slack, the posterior mean returns to `n_hat`,
   and TPM → raw — gracefully, with no never-worse violation observed.

The honest framing: the **hard clip is parameter-free, provably never-worse, and the
right default**; **TPM is strictly more powerful in the lightly-duplicated window but
needs a calibrated noise model**, so it is the opt-in variant for systems that know
their sketch's error profile.

## 5b. The refined clip: a tighter wall from heavy hitters

The clip uses `L` as the wall. But `L` is only the loosest valid upper bound on `n`; a
tighter one is available almost for free. Write the **duplicate mass**
`D = sum_i (f_i - 1) = L - n`, so `n = L - D`. A Misra–Gries summary with `k` counters,
run in the same pass, returns, for the items it tracks, counters `c_i` that **never
exceed** the true frequency (`c_i <= f_i`). Define the estimated heavy-duplicate mass
`D_hat = sum_{tracked, c_i >= 2} (c_i - 1)`.

**Lemma (valid tighter wall).** `n <= L - D_hat <= L`.
*Proof.* The tracked items are distinct, and `c_i - 1 <= f_i - 1` for each; summing over
a subset of items, `D_hat <= sum_all (f_i - 1) = D`. Hence `U := L - D_hat >= L - D = n`,
and `U <= L` since `D_hat >= 0`. ∎

The **refined clip** is `min(n_hat, U) = min(n_hat, L - D_hat)`.

**Proposition (domination hierarchy).** Pointwise, for every realization,
`|min(n_hat, U) - n| <= |min(n_hat, L) - n| <= |n_hat - n|`.
*Proof.* Both inequalities are the never-worse argument (§1) applied at the walls `U`
and `L`: since `n <= U <= L`, clipping at `U` is at least as good as clipping at `L`,
which is at least as good as not clipping. Concretely, with `U <= L`: if `n_hat <= U`,
all three are equal; if `U < n_hat <= L`, the refined clip returns `U in [n, n_hat]`
(closer to `n` than `n_hat = ` basic); if `n_hat > L`, refined returns `U <= L = ` basic,
and `U >= n`, so refined is at least as close. ∎

So the refined clip is **never worse than raw, and never worse than the basic clip** —
it strictly generalizes the latter, recovering it exactly when `D_hat = 0` (no heavy
duplicates). Its extra value is governed by how tight the wall is, i.e. by `D_hat / D`:

- **Spread duplication** (every item lightly repeated, `f_i` below the MG threshold
  `~L/k`): MG tracks no heavy item, `D_hat ~ 0`, `U ~ L`, and the refined clip reduces to
  the basic clip. Measured: identical to basic (and to raw when `d` is past the window).
- **Concentrated duplication** (a few items carry the duplicate mass, `f_i >> L/k`): MG
  captures them, `D_hat ~ D`, `U ~ n`, and the wall bites even when `d >> 1` and the basic
  clip (wall `L >> n`) is useless (`exp10_refined_clip.py`).

**But this is not a free win — and the honest accounting matters.** The Misra-Gries
summary is a *second sketch*: at `k=256` it is ~3 KB, larger than the `m=4096` HLL (~2 KB)
it augments. At **matched total memory**, the refined clip does NOT beat simply enlarging
the HLL. At its most favorable point (one dominant item, `d=5`, `n=32000`), the refined
clip (HLL + MG, ~5 KB) reaches 0.87% rel-RMSE, while a plain HLL at `lg_k=13` (~4 KB,
*less* memory) reaches 0.76% and `lg_k=14` reaches 0.50%. Because HLL is duplicate-insensitive,
memory spent on heavy-hitter structure buys less than memory spent on registers
(verified directly; the earlier "24-29% gain over raw" figure compared a ~5 KB estimator
to a ~2 KB one — a free-memory artifact, now corrected).

**The honest, conditional takeaway:** *if a heavy-hitter summary is already maintained*
(hot-key detection, top-k), `L - D_hat` tightens the clip's wall at no additional cost and
the domination `refined ⪰ basic ⪰ raw` applies; as a fresh memory investment it is
dominated by a larger plain sketch. This delimits what `L` can offer a duplicate-insensitive
sketch: the value lives near `d=1` (the basic clip), and the only extension — a tighter
wall from auxiliary structure — does not pay for that structure on its own.

## 6. Closest Bayesian prior art

Beraha–Favaro–Sesia (2024) give a Bayesian posterior over the distinct count from a
count-min-type sketch whose support is inherently `⊆ {1..n}` (the sample size). It is
not HLL, the truncation is never isolated as the working mechanism, and no effect size
is reported. §5 differs in object (HLL-class register sketches), in treating `L` (the
stream length, not a known sample size) as the truncation, and in quantifying the
gain — but the connection is cited and differentiated in the paper.
