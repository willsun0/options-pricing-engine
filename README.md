# Options Pricing Engine

A from-scratch options pricing library in Python: closed-form analytics, lattice
methods, Monte Carlo simulation, and volatility surface modelling — built to be
read and explained, not to be fast.

Every non-obvious numerical choice is justified in a comment where it is made.
If something looks arbitrary, the reasoning is next to it.

**Status:** Phases 1–4 complete (Black-Scholes, Greeks, CRR binomial tree, Monte
Carlo with variance reduction and exotics, real-market implied volatility and the
smile). Phase 5 in progress.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```bash
pytest
```

Generate the Phase 1 figures into `plots/`:

```bash
python options_engine/notebooks/phase1_black_scholes.py
python options_engine/notebooks/phase2_binomial_tree.py
python options_engine/notebooks/phase3_monte_carlo.py
python options_engine/notebooks/phase4_market_comparison.py          # cached data
python options_engine/notebooks/phase4_market_comparison.py --fetch  # live data
```

Price something:

```python
from options_engine import black_scholes_price, all_greeks

price = black_scholes_price(
    spot=100.0, strike=105.0, time_to_expiry=0.5,
    rate=0.04, volatility=0.22, option_type="call",
)
greeks = all_greeks(100.0, 105.0, 0.5, 0.04, 0.22, "call")
print(float(price), float(greeks.delta))
```

Every function broadcasts, so a whole strike ladder is one call:

```python
import numpy as np
strikes = np.linspace(80, 120, 41)
prices = black_scholes_price(100.0, strikes, 0.5, 0.04, 0.22, "call")
```

---

## Project layout

```
options_engine/
  common.py              Shared types, the OptionType enum, input validation
  greeks.py              All five Greeks: analytical and finite-difference
  pricing/
    black_scholes.py     Closed-form European pricing  [Phase 1]
    binomial_tree.py     Cox-Ross-Rubinstein lattice    [Phase 2]
    monte_carlo.py       Simulation + variance reduction [Phase 3]
  data/
    market_data.py       yfinance fetching, cleaning, caching [Phase 4]
  vol_surface/
    implied_vol.py       Newton/Brent implied vol solver      [Phase 4]
  tests/                 pytest suite (+ committed market fixture)
  notebooks/             Scripts that generate the figures
plots/                   Generated figures
data_cache/              Cached market snapshots (gitignored)
```

---

## Phase 1 — Black-Scholes

### The math

Full derivation lives in the module docstring of
[`pricing/black_scholes.py`](options_engine/pricing/black_scholes.py). The short
version, in three steps:

**1. Model the stock as geometric Brownian motion.**

$$dS_t = (\mu - q) S_t\,dt + \sigma S_t\,dW_t$$

Both terms scale with $S_t$, so returns rather than prices are what behave
consistently — which is why the stock can never go negative. Applying Itô's lemma
to $\ln S_t$ produces a second-order term that ordinary calculus would miss:

$$d(\ln S) = \left(\mu - q - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dW$$

That $-\sigma^2/2$ is not a modelling choice — it falls out of $(dS)^2 = \sigma^2 S^2 dt$.
It is why the median of $S_T$ sits below its mean, and it's the step most people
fumble when asked to derive this.

**2. Hedge away the risk, and $\mu$ disappears.**

Hold one option and short $\Delta = \partial V/\partial S$ shares. The $dW$ terms
cancel, so over an instant the portfolio is riskless and must earn $r$. That gives
the Black-Scholes PDE:

$$\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \tfrac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} - rV = 0$$

Note what's missing: $\mu$. Two investors who disagree completely about the stock's
expected return must still agree on the option's price, because either could run
the hedge against the other. Equivalently (Feynman–Kac), the price is a discounted
expectation under a measure where the drift is $r-q$:

$$V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_T)]$$

This is *not* an assumption that investors are risk-neutral. It's a change of
measure — a pricing device that happens to give the same answer as the hedge.

**3. Evaluate the integral.**

$S_T$ is lognormal under $\mathbb{Q}$, so the expectation splits into two pieces
and yields:

$$C = S e^{-qT}N(d_1) - Ke^{-rT}N(d_2), \qquad P = Ke^{-rT}N(-d_2) - Se^{-qT}N(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

### How to read the two terms

- $N(d_2)$ **is** the risk-neutral probability of finishing in the money. So
  $Ke^{-rT}N(d_2)$ is "what you expect to pay, discounted."
- $N(d_1)$ is **not** a probability under $\mathbb{Q}$. It's the same event measured
  under a different numeraire — the stock instead of the bank account — i.e.
  probability reweighted by how much stock you hold in each state. It's also exactly
  $\partial C/\partial S$.
- So: **price = PV(what you get) − PV(what you pay)**, each weighted by the
  probability of exercise under its natural numeraire. Being able to say this
  cleanly is worth more in an interview than reciting the formula.
- $e^{-qT}$ appears because the option holder doesn't collect dividends before
  expiry. Setting $q=0$ recovers the original 1973 formula; the $q$ term is Merton's
  extension. It's included from the start here because Phase 4 prices SPY, which
  yields ~1.2% — ignoring it biases calls high and puts low.

### The Greeks

All five are implemented analytically **and** numerically in
[`greeks.py`](options_engine/greeks.py):

| Greek | Formula (call) | Meaning |
|---|---|---|
| Delta | $e^{-qT}N(d_1)$ | Shares to hold to hedge |
| Gamma | $\dfrac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt{T}}$ | How fast the hedge goes stale |
| Vega | $Se^{-qT}\varphi(d_1)\sqrt{T}$ | Exposure to volatility |
| Theta | $-\dfrac{Se^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}} + qSe^{-qT}N(d_1) - rKe^{-rT}N(d_2)$ | Decay per unit calendar time |
| Rho | $KTe^{-rT}N(d_2)$ | Exposure to rates |

Two structural facts the tests pin down, both consequences of put–call parity:
**gamma and vega are identical for calls and puts** (the parity difference
$Se^{-qT} - Ke^{-rT}$ is linear in $S$ and free of $\sigma$, so it vanishes under
$\partial^2/\partial S^2$ and $\partial/\partial\sigma$), and each Greek's call–put
difference equals the corresponding derivative of the parity relation.

**A note on units.** Every function returns the *raw* derivative. Desks quote
several of these rescaled, and silently mixing conventions is the most common bug
in Greeks code, so nothing is rescaled implicitly — use the named helpers:

| Greek | Returned as | Desk convention | Helper |
|---|---|---|---|
| Vega | per 1.00 of vol | per vol point (1%) | `vega_per_percent()` |
| Theta | per year | per calendar day | `theta_per_day()` |
| Rho | per 1.00 of rate | per basis point | `rho_per_basis_point()` |

A raw vega of 16.2 means "gains 16.2 if vol goes 20% → 120%." The desk number is
0.162.

### Why analytical *and* numerical Greeks

The analytical formulas come from hand-differentiating the price. The numerical
ones only ever call the pricer. **Agreement between two independent computations
validates the price formula, the derivative algebra, and the finite-difference
machinery simultaneously** — it's the strongest single test in the project.

The numerical path also generalises: pass any pricer with the standard signature
to `numerical_greeks(price_fn=...)` and you get its Greeks without new derivations.
That's how Phases 2 and 3 will get Greeks out of the tree and Monte Carlo engines,
where no closed form exists.

**Choosing the bump size** (explained fully in `greeks.py`). A central difference
trades two error sources: truncation $O(h^2)$ against roundoff $O(\epsilon/h)$.
Balancing them gives $h^* \sim \epsilon^{1/3} \approx 6\times10^{-6}$. But gamma is a
*second* difference, dividing by $h^2$, so cancellation bites harder and its optimum
is $h^* \sim \epsilon^{1/4} \approx 10^{-4}$ — two orders of magnitude larger. The two
therefore get separate bump constants; sharing one would be wrong for at least one
of them. Both defaults were confirmed by sweeping $h$ and checking the error curve
has the predicted V-shape.

### Handling degenerate inputs

At $T=0$ or $\sigma=0$ the formula divides by zero. Rather than branching, the code
clamps to `TINY = 1e-12` and lets the formula's own limit do the work: $d_1,d_2 \to
\pm\infty$, the normal CDFs saturate at 0 or 1, and the price collapses to discounted
intrinsic value. This keeps every pricer branch-free and fully vectorised.

The cost is a bounded residual: exactly at expiry *and* exactly at the money, the
price retains $\approx \varphi(0)S\sigma\sqrt{\texttt{TINY}} \approx 8\times10^{-6}$
of time value for a \$100 name — four orders of magnitude below a one-cent tick, and
exactly zero once $S$ moves off $K$. That trade is documented at the constant and
asserted in the tests rather than left implicit.

### Test suite

7,424 tests, layered weakest-assumption-first:

1. **Known values** — Hull's Example 15.6 (call 4.76, put 0.81), plus the exact
   at-the-money-forward identity $C = Se^{-qT}(2N(\sigma\sqrt{T}/2)-1)$.
2. **Put–call parity** — model-free, so it catches sign and discounting errors
   even where both legs individually look plausible. Checked across a 1,080-point
   grid including negative rates.
3. **Arbitrage bounds and monotonicity** — properties any correct pricer must have,
   checked over a grid rather than at one point.
4. **Limits** — zero vol, zero time, deep ITM/OTM, infinite vol.
5. **Analytical vs numerical Greeks** — the core validation described above.
6. **Interface** — broadcasting, validation, enum/string handling.

Some tests deliberately assert *counterintuitive* behaviour, because those are the
cases that reveal whether the model is understood:

- A deep-ITM European **put loses value as maturity extends** (you wait longer to
  collect the strike). This is exactly the configuration where early exercise has
  value — and therefore the case Phase 2's American put must price higher.
- Theta is **positive** for that same deep-ITM European put.
- At-the-money-forward delta is **slightly above 0.5**, not equal to it: $d_1 =
  \sigma\sqrt{T}/2 > 0$ at the forward. "ATM delta is 50" is an approximation, and
  knowing why is a good interview answer.

### Figures

| File | What it shows |
|---|---|
| `plots/phase1_greeks.png` | Price and all five Greeks vs spot at 1 month / 6 months / 2 years |
| `plots/phase1_analytical_vs_numerical.png` | Absolute error between the two Greek implementations |

The Greeks figure is worth being able to sketch from memory: gamma and vega peak
at the strike and *invert* with maturity (short-dated gamma is tall and narrow,
long-dated is low and broad), while theta's trough is deepest for short-dated
at-the-money options. Those three facts together explain why short-dated ATM options
are expensive to hedge.

The validation figure shows errors at $10^{-15}$–$10^{-8}$ with the ragged look of
roundoff-dominated noise — which is what should appear when the algebra is right.

---

## Phase 2 — Binomial tree (Cox–Ross–Rubinstein)

Black-Scholes has a closed form, but only for European exercise. The moment a
holder can exercise early there *is* no formula — the value at each instant depends
on a decision, and decisions have to be solved backwards. That's what a lattice does.

Full write-up in [`pricing/binomial_tree.py`](options_engine/pricing/binomial_tree.py).

### Why these parameters

Three unknowns ($u$, $d$, $p$), three conditions:

1. **Match the risk-neutral mean.** $pu + (1-p)d = e^{(r-q)\Delta t}$, giving
   $p = \dfrac{e^{(r-q)\Delta t} - d}{u - d}$. Same change of measure as Black-Scholes —
   $\mu$ never appears.
2. **Match the variance.** $u = e^{\sigma\sqrt{\Delta t}}$. The $\sqrt{\Delta t}$ is the
   signature of Brownian motion.
3. **Recombine.** CRR's distinctive choice, $d = 1/u$, means an up-then-down move
   returns exactly to the start. After $N$ steps there are $N+1$ distinct prices
   rather than $2^N$ paths — this is what turns an intractable exponential problem
   into an $O(N^2)$ one, and it's the most important implementation fact about the
   method.

Worth being able to name the alternatives: **Jarrow–Rudd** sets $p=1/2$ and pushes
drift into $u,d$; **Tian** matches a third moment; **Leisen–Reimer** places a node
exactly on the strike. CRR is used here because $d=1/u$ is the easiest to verify
by hand.

`d = 1/u` also buys a concrete implementation win: every node price in the whole
tree is $S u^k$ for integer $k \in [-N, N]$, so one geometric ladder is precomputed
once and each time level is a **stride-2 slice** of it. No per-step `exp()` calls.

### The arbitrage guard

$p$ is only a probability if $|r - q|\sqrt{\Delta t} < \sigma$. Violate it — low vol,
high rates, too few steps — and the tree admits arbitrage and returns silently
wrong prices. `crr_parameters` raises instead, and the message reports the minimum
step count needed. A test extracts that number from the message and confirms it
actually works.

### Convergence, and why it wobbles

Error decays as $O(1/N)$ — log-log fits give slopes of −0.83 to −1.07 across
parameter sets — but **not smoothly**.

The mechanism is purely geometric. Terminal nodes sit at $Su^k$ for
$k = -N, -N+2, \ldots, N$, so **$k$ always shares the parity of $N$**. The strike
sits at a generally non-integer level $k^* = \ln(K/S)/(\sigma\sqrt{\Delta t})$, so
which nodes are available to straddle it flips with the parity of $N$. Since the
payoff kink at $K$ is exactly where discretisation error concentrates, even and odd
$N$ converge along two separate smooth curves. This is visible in
`plots/phase2_convergence.png` and asserted directly in the tests.

Two remedies, and their measured results:

| Method | Measured effect |
|---|---|
| Average $V_N$ and $V_{N+1}$ | **2.5–3× better.** Mixes one point from each branch. Implemented as `binomial_price_averaged`. |
| Richardson $2V_{2N} - V_N$ | **3–6× worse.** Deliberately not implemented. |

Richardson is the textbook fix for an $O(1/N)$ error, and it fails here because it
assumes the error admits a *smooth* expansion in $1/N$ — the parity oscillation
violates that outright, so differencing across it amplifies the wobble instead of
cancelling it. There's a test asserting this failure, so the claim stays honest if
the code changes. Knowing why a standard technique doesn't apply is worth more than
applying it reflexively; the principled fix is Leisen–Reimer.

### American exercise

At every node the holder takes $\max(\text{continuation}, \text{intrinsic})$. That
one line is the whole reason to use a lattice, and it turns pricing into a *free
boundary* problem with no closed form.

The tests pin down the inequalities that define early-exercise value:

- American ≥ European always, and American ≥ intrinsic always.
- **American call = European call exactly when $q = 0$.** This is the sharpest test
  of correctness: early exercise of a call on a non-payer is never optimal (you'd
  throw away time value *and* the interest earned by deferring the strike), so a
  `max()` applied even slightly too eagerly would push the American price above the
  European one and fail. With a 10% dividend yield the same test confirms the logic
  isn't merely inert.
- The early-exercise premium is **increasing in $r$** and vanishes at $r=0$, because
  the premium is driven by interest forgone on the strike.
- The deep-ITM American put case continues directly from Phase 1: that's exactly the
  configuration where the European put has *positive theta* — it loses value by
  waiting to collect the strike. The American holder simply doesn't wait.

`early_exercise_boundary()` traces the free boundary through time. It starts at the
strike at expiry and falls away as time to expiry grows: with more time left you
demand to be deeper in the money before surrendering the optionality. The lattice
resolves it only to the nearest node, so the returned curve is a staircase.

### Validation strategy

Beyond convergence, two things guard against a plausible-but-wrong induction:

- **Hand-computed trees.** One- and two-step trees worked out longhand in the test
  file, node by node. A 2000-step tree converges to Black-Scholes even with subtle
  flaws because errors average out; a two-step tree forgives nothing.
- **An independent reference implementation.** A deliberately naive nested-loop
  version sharing no code with the vectorised one — nested lists, repeated
  multiplication for node prices, no NumPy. Agreement to 1e-11 across a 96-point
  grid validates the stride-2 slicing tricks.

Put–call parity is also asserted at **3, 17, and 200 steps** — on the tree it holds
at *any* step count, not just in the limit, because both legs use the same nodes and
the same $p$. A failure would mean the induction itself is broken.

### Figures

| File | What it shows |
|---|---|
| `plots/phase2_convergence.png` | Price vs step count; even/odd branches coloured separately; raw vs averaged |
| `plots/phase2_convergence_rate.png` | Log-log error with fitted slope (−0.99) against an $O(1/N)$ reference |
| `plots/phase2_american_early_exercise.png` | American vs European put, the premium, and the free boundary |

Note in the first figure how the European put trades **below intrinsic** — that is
not an error, and it's the visual statement of why the American put is worth more.

---

## Phase 3 — Monte Carlo

Black-Scholes gives a formula; the tree handles early exercise. Both run out of road
when the payoff depends on the *whole path* (an average, a barrier touch) or on
several underlyings. Monte Carlo doesn't care: if you can simulate the path and
evaluate the payoff, you can price it.

Full write-up in [`pricing/monte_carlo.py`](options_engine/pricing/monte_carlo.py).

### The rate is both bad and good

$$\text{SE} = \sigma_\text{payoff} / \sqrt{N}$$

**Bad:** halving the error needs 4× the paths; one more digit needs 100×. Against the
tree's O(1/N) this is terrible for a vanilla, and nobody prices one this way.

**Good:** the rate is *independent of dimension*. A lattice costs O(steps^d) and dies
past three or four underlyings; $\sqrt{N}$ doesn't know what $d$ is. That single
property is why desks run simulations for basket and path-dependent products.

Since the exponent can't be improved by cleverness, the only lever is the numerator.
That's what variance reduction means — and it shows up visually in
`plots/phase3_variance_reduction.png` as **four parallel lines**: every technique
shifts the level, none changes the slope.

### No discretisation bias

Most Monte Carlo write-ups reach for an Euler scheme and inherit O(dt) bias. That's
unnecessary for GBM, which has an exact solution *and* an exact transition between
any two dates. So we jump straight to expiry for European payoffs and step exactly
between monitoring dates for path-dependent ones. Every price here is unbiased —
the only error is statistical, and the reported standard error genuinely bounds it.
A test verifies each monitoring date's marginal distribution is exactly right on a
deliberately coarse 4-step grid, which an Euler scheme would fail.

### Antithetic variates: not a free win

For every draw $Z$, also use $-Z$. At equal computational cost,

$$\text{Var}_\text{antithetic} / \text{Var}_\text{plain} = 1 + \rho, \qquad \rho = \text{Corr}(f(Z), f(-Z))$$

So the benefit is entirely determined by how monotone the payoff is in $Z$. Measured:

| Payoff | ρ | Improvement |
|---|---|---|
| Deep ITM call (K=40) | → −1 | **4.1×** |
| ATM call (K=100) | negative | 1.35× |
| Deep OTM call (K=250) | → 0 | 1.03× |
| **Straddle** $\|S_T-K\|$ | **positive** | **worse than plain** |

The straddle is the case that proves it isn't magic: a symmetric payoff makes
mirrored pairs *positively* correlated, and the technique actively hurts. That's the
third panel of the variance-reduction figure, and it's asserted in the tests.

### Control variates

With a correlated $X$ whose $E[X]$ is known exactly, $Y^* = Y + c(X - E[X])$ has the
same mean and, at the optimal $c^* = -\text{Cov}(Y,X)/\text{Var}(X)$, variance scaled
by $(1-\rho^2)$.

The intuition is bookkeeping: we know the true answer for $X$, so we can *see* the
error the draws produced on the control, and subtract off the part of $Y$'s error
that moved with it.

The showcase is the arithmetic Asian. It has no closed form (a sum of lognormals
isn't lognormal), but the **geometric**-average version does — the geometric mean of
lognormals *is* lognormal, so it's a Black-Scholes problem in disguise. Rather than
derive a second pricing formula, `geometric_asian_price` maps it onto the Phase 1
pricer via an effective volatility and a synthetic dividend yield that reproduces
$E[G]$. Measured correlation with the arithmetic average: **ρ = 0.9992**, giving

> **24.5× smaller standard error = 600× less variance** — the same accuracy from 600×
> fewer paths.

**One honestly-stated caveat:** fitting $c^*$ on the same sample makes the estimator
slightly biased. The bias is O(1/N) while the noise is O(1/√N), so it's dominated and
vanishes faster. Purists use a pilot run; this accepts it, which is standard practice.

### The techniques do *not* simply stack

| Setup | Control alone | + antithetic |
|---|---|---|
| European call, terminal-stock control | 0.0158 | **0.0079** ✓ |
| Asian, European-option control (weak) | 0.0148 | **0.0133** ✓ |
| Asian, geometric control (ρ=0.9992) | **0.00159** | 0.00164 ✗ |

Once a near-perfect control has stripped out the variance, what survives is the
*difference* between the arithmetic and geometric averages — a spread that's large
when the path is volatile regardless of direction, i.e. **not monotone in $Z$**.
Mirroring has nothing left to cancel while still halving the sample count. The rule
worth carrying: *antithetic sampling helps to the extent the remaining variance is
monotone in $Z$, and a strong control may already have spent that.*

### Exotics

**Asian** (arithmetic average). Averaging damps effective volatility toward
$\sigma/\sqrt{3}$, so Asians are cheaper than vanillas. Tests assert AM ≥ GM path by
path, that one averaging date recovers the vanilla exactly, and that price decreases
monotonically in the number of averaging dates.

**Barrier** (knock-in/knock-out, both directions). The sharpest test here is **in-out
parity**: every path either touches or doesn't, so in + out = vanilla *path by path*,
holding to **1.8e-15** — no statistics involved.

Discrete vs continuous monitoring is a **real distinction, not an approximation
error**: a daily-monitored knock-out is genuinely worth more than a continuous one,
because the price can cross intraday and come back. The simulation prices the discrete
contract exactly and *should not* converge to the continuous one.

If you do want the continuous price, `barrier_price_analytic` gives it in closed form
via the **reflection principle** — every path touching $H$ and ending at $x$ mirrors to
one ending at $2H-x$, with a $(H/S)^{2\lambda}$ Girsanov reweighting for the drift.
The raw discrete price approaches it only as O(1/√m), but **Broadie–Glasserman–Kou**
showed the fix is a barrier *shift* of $\exp(\pm 0.5826\,\sigma\sqrt{dt})$ rather than
more dates. Measured at 250 monitoring dates: raw error **+0.177**, corrected error
**−0.004** — about **40× better for free**. The constant is $-\zeta(1/2)/\sqrt{2\pi}$,
from the expected overshoot of a random walk past a level.

### Testing a randomised algorithm

Two rules throughout: **every test is seeded**, and **tolerances are derived from the
standard error, never invented** — "matches to 0.01" is meaningless without knowing
the noise level.

The most important test isn't a price at all. `test_confidence_intervals_have_the_coverage_they_claim`
runs 200 independent simulations and counts how many 95% intervals actually contain
the Black-Scholes value. The count is binomial(200, 0.95) with σ≈3.1, so the accepted
band is ±3σ. **An estimator whose error bars lie is more dangerous than one that's
merely imprecise.**

Two subtleties the implementation gets right:

- **Standard errors are computed over antithetic *pairs*, not paths.** Mirrored paths
  are deliberately *not* independent — that's the whole point of them — so treating
  2N of them as 2N samples produces a confidence interval that's simply wrong.
- **Sharing a seed does not share a sample.** `simulate_terminal_prices` draws an
  (n,1) array and `simulate_paths` draws (n,m); the generator is consumed in different
  shapes, so identical seeds give different numbers. The barrier parity tests
  regenerate the vanilla from the *path* simulation to make the comparison exact.

### A memory guard

Path storage is O(n_paths × n_steps) and grows faster than intuition suggests: 400,000
paths over 2,000 dates is ~12 GiB across the live buffers. Unguarded, that doesn't
fail — it swaps, and the process crawls for minutes looking like a hang. `simulate_paths`
now computes in place (halving peak allocation) and refuses oversized requests with the
actual figure and a path count that would fit. A test extracts that suggested number and
confirms it works.

### Figures

| File | What it shows |
|---|---|
| `plots/phase3_convergence.png` | Price with 95% bands, and error vs the O(1/√N) reference |
| `plots/phase3_variance_reduction.png` | Four parallel lines; the Asian control; the straddle counterexample |
| `plots/phase3_exotics.png` | Asian vs vanilla, in-out parity, and BGK landing on the exact continuous price |

---

## Phase 4 — Real market comparison

Phases 1–3 assume clean inputs. This phase is where the model meets actual quotes,
and it produces the project's central result: **Black-Scholes is visibly, measurably
wrong, and the market knows it.**

Data snapshot: SPY, spot 756.20, r = 3.70%, q = 1.23%, 7 expiries from 11 to 331 days.

### Implied volatility

Every Black-Scholes input is observable except volatility, so practitioners run the
formula backwards. Two things worth being precise about:

- **It is not a forecast.** It's the number that makes a particular model reproduce
  a particular price. When the model is wrong, implied vol absorbs *everything* the
  model gets wrong — not just volatility.
- **It's a quoting convention**, like yield for bonds. Nobody believes the stock has
  a different volatility depending on which strike you look at. Yet the implied vols
  differ across strikes — and that discrepancy *is* the smile.

The classic summary: *"the wrong number to put into the wrong formula to get the
right price."*

**The problem is well posed** because vega > 0, so price is strictly monotone in σ.
A solution exists and is unique exactly when the quote sits inside
`[discounted intrinsic, S·e^{-qT}]`. Outside that, no σ works — the solver checks
first and reports which bound failed and by how much, rather than letting a
root-finder wander.

**Newton first, Brent as backstop.** Newton is quadratic and we already have vega in
closed form, but it breaks where vega collapses (the wings — precisely where a smile
needs data). Measured over a realistic grid: **93% Newton, 7% Brent**, with Brent
taking the deep-ITM short-dated cases. Both halves are asserted, so neither the fast
path regressing nor the fallback rotting goes unnoticed.

Starting point is **Manaster–Koenig** (`σ₀ = √(|ln(S/K) + (r−q)T| · 2/T)`), the
volatility of maximum vega for that moneyness, from which Newton provably converges
monotonically.

**Verification:** round-tripping 1,120 price/vol combinations recovers σ, and on the
real chain **all 1,070 quotes invert with zero failures**, repricing the market mid
to within **1e-8**.

### The tolerance had to be derived, not chosen

This took three attempts and is the most interesting numerical detail in the phase.

A flat absolute tolerance of 1e-8 sounds strict — a millionth of a tick. But a deep
OTM option can be *worth* 1e-9, so the tolerance exceeds the entire price, and the
solver "converges" to σ = 0.830 against a true 0.800 while reporting success.

Scaling with price fixes that end but not the other: a 50-strike call on a 100 spot
is worth ~50, of which the *time value* might be 1e-9 — and only time value responds
to σ. So the tolerance scales with **extrinsic value**, with an absolute floor near
machine precision. For OTM options the two definitions coincide.

The tests then needed the same treatment. Volatility is recovered *through* price,
so the finest resolvable σ difference is the one that moves the price by one
representable unit:

$$\delta\sigma \approx \epsilon \cdot \text{price} / \text{vega}$$

That's ~1e-15 near the money and ~1e-7 for a deep ITM option. A flat `rel=1e-6`
assertion fails on perfectly correct answers — as it did on a 130-strike put whose
*price error was exactly zero* while σ was off by 1.2e-6. The test tolerance is now
computed from that formula, and a separate test asserts the property that *does* hold
everywhere: the solved vol reprices the input exactly.

### Data cleaning is most of the work

Nearly every surprising result in an options project is a data problem, not a
modelling one. Filters, each targeting a documented failure:

| Filter | Prevents |
|---|---|
| Mid, **never** `lastPrice` | Stale prints. **39% of quotes are stale by >10%**; one 630-call showed `lastPrice` 107.38 against a 124.62/127.43 market |
| Zero bid | Uninvestable; mid is half the ask, i.e. fiction |
| Relative spread ≤ 25% | A 0.05/0.60 market has ±85% uncertainty in its mid |
| Open interest ≥ 10 | Contracts nobody trades |
| **OTM only** | Deep ITM options are ~all intrinsic; σ is recovered from pennies of time value against a huge price |
| Min maturity | Sub-week options are dominated by pinning, not theory |

The OTM filter costs nothing: put–call parity means the OTM put and ITM call at the
same strike carry identical information, and the OTM one carries it far better
conditioned. **2,291 raw quotes → 990 usable (43%).**

Two input traps worth naming:

- **Moneyness must be measured against the forward**, not spot. An option is ATM
  when `K = F`. Using spot tilts the whole smile by the cost of carry.
- **`yfinance`'s dividend fields disagree on units**: `info["dividendYield"]` returns
  `1.01` (percent) while `info["yield"]` returns `0.0101` (decimal). Picking wrong is
  a 100× error in an input that directly shifts the forward. This computes the yield
  from actual trailing dividends instead.

### Validating against the vendor

yfinance publishes its own implied vol. Mine differ by a median of **0.8 vol points**
— which sounds like a discrepancy until you recompute under *their* assumptions:
setting `r = 0, q = 0` collapses the gap to **0.0014**. So the difference is entirely
explained by inputs, not math, and the vendor's numbers are the less correct ones.
That's a much stronger validation than agreement would have been.

### What the smile shows

| Expiry | ATM vol | Skew slope |
|---|---|---|
| 11d | 11.2% | −1.31 |
| 88d | 14.6% | −0.61 |
| 331d | 18.3% | −0.28 |

**1. Implied vol is not constant across strikes.** At 88 days it ranges from **11.8%
to 44.1% — a 32 vol point spread**. Black-Scholes says this should be a horizontal
line.

**2. The skew is downward-sloping**, not a symmetric smile. Low strikes are far more
expensive. This is a post-1987 phenomenon: pre-crash equity smiles were roughly
symmetric.

**3. Skew decays as T^−0.446**, against the theoretical **1/√T** (−0.5). Rescaling
the x-axis by σ√T collapses all seven expiries onto essentially one curve — visible
in the right panel of `phase4_smile.png`, and a genuinely satisfying empirical
confirmation.

### Why Black-Scholes deviates

The model assumes returns are **lognormal with constant volatility**. Reality
violates that in three specific ways, and each maps onto a feature of the smile:

**Fat tails.** Real returns have far more extreme moves than a normal distribution
allows. A 5σ daily move should happen once per ~7,000 years; equity indices deliver
them every few years. Options are bets on tails, so the market prices the tails it
actually observes — lifting both wings.

**Negative skewness (why it's a skew, not a smile).** Equities crash down, not up.
Leverage rises mechanically as prices fall, and correlations spike in a sell-off, so
the left tail is genuinely fatter than the right. Add persistent demand for crash
protection from institutions who must hedge, and the put wing carries a risk premium
on top of the physical probability.

**Volatility is stochastic and clusters.** σ is not a constant — it's a process, and
a mean-reverting one that jumps in crises and is *negatively correlated with returns*
(the leverage effect). Mixing over an uncertain σ produces fatter tails than any
single σ can, which is exactly what a smile encodes.

**The 1/√T decay follows from the central limit theorem.** Over long horizons,
returns aggregate towards normality, so the risk-neutral distribution's excess
skewness shrinks and the smile flattens. Short-dated options are dominated by
jump risk, which cannot average out.

### What this costs in dollars

Calibrating one volatility at the money (14.6% at 88 days) and pricing every strike
with it:

- Worst dollar error: **−$4.27** on an option the market prices at $6.24 — a 68%
  underpricing, on a contract whose bid-ask spread is a few cents.
- Deep OTM puts (moneyness < 0.75): market averages **$0.975**, the model says
  **$0.0000325**. The model is **30,000× too cheap.**

That last figure is the phase in one number. The model doesn't just misprice crash
protection — it says it's essentially free. Anyone who sold those options at model
value would be handing out disaster insurance for nothing, which is roughly the
trade that ended several funds in 1987 and 2008.

**The practical resolution:** nobody uses one volatility. Desks quote *in* implied
vol precisely because it's the language for describing where Black-Scholes needs
correcting, and then interpolate a surface across it. Phase 5 fits that surface.

### Figures

| File | What it shows |
|---|---|
| `plots/phase4_smile.png` | The smile at each expiry; rescaled by σ√T the curves collapse |
| `plots/phase4_surface.png` | 3-D surface, ATM term structure, and skew decay vs a 1/√T reference |
| `plots/phase4_model_vs_market.png` | Constant-vol prices vs market, in dollars and percent |
| `plots/phase4_data_quality.png` | The cleaning funnel, spread distribution, and `lastPrice` staleness |

### Reproducibility and offline tests

Every fetch is cached to disk, because option chains change every second and
otherwise no figure here could be regenerated or checked. The test suite runs
entirely against a **committed fixture** — deliberately including the messy rows the
filters exist to remove. The one live test is marked `network` and deselected by
default:

```bash
pytest -m network
```

---

## Phase 5

- **Phase 5 — ML extension.** A small PyTorch model for the vol surface, compared
  against a parametric SVI/SABR fit.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Merton, R. C. (1973). *Theory of Rational Option Pricing.*
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.*
- Hull, J. C. *Options, Futures, and Other Derivatives.*
- Glasserman, P. *Monte Carlo Methods in Financial Engineering.*
