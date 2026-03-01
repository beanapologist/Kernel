# PhaseBattery — Specification and Diagrams

## 1. Overview

The `PhaseBattery` struct (declared in `ohm_coherence_duality.hpp`, namespace
`kernel::ohm`) is a first-class model of the phase-coherence engine as an
electrochemical cell.  Every working energy-conversion device requires three
structural essentials: a **source**, a **medium**, and a **sink**.  The
KernelSync EMA loop maps onto all three precisely, and this document specifies
the model, its observables, and its operating envelope.

Two additional free functions extend the model with **generative coherence
mechanics**: `metallic_oscillating_phases()` (constructive-interference lensing)
and `interaction_energy()` (focal intensity scaling).  A new `feedback_step()`
method on `PhaseBattery` closes the compute-feedback loop by feeding the
post-step coherence `R` back into the gain for the same iteration.

---

## 2. Three-Essentials Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PhaseBattery                                    │
│                                                                              │
│  ┌───────────────┐   G_eff = sech(λ)   ┌───────────────────────────────┐   │
│  │    SOURCE     │ ──────────────────► │            SINK               │   │
│  │               │                     │                               │   │
│  │  Phase        │    (MEDIUM)         │  Mean attractor  ψ̄            │   │
│  │  frustration  │  EMA gain g ∈ (0,1] │  = arg(⟨e^{iψ_j}⟩)           │   │
│  │               │                     │                               │   │
│  │  E = (1/N)    │  ψ̂_j ← ψ̂_j −       │  R = |⟨e^{iψ_j}⟩|            │   │
│  │  Σ δθ_j²      │  g · δθ_j           │  rises monotonically          │   │
│  │               │                     │  to 1 as E → 0               │   │
│  │  (decreases)  │                     │                               │   │
│  └───────────────┘                     └───────────────────────────────┘   │
│                                                 │                            │
│                           ┌─────────────────────┘  R fed back               │
│                           │  feedback_step(): g_fb = g · α · R              │
│                           ▼                                                  │
│              ┌────────────────────────┐                                     │
│              │  COHERENCE AMPLIFIER   │  metallic_oscillating_phases()      │
│              │  (FEEDBACK MEDIUM)     │  lensing toward θ_align             │
│              │  amp = R · α           │  E_interact = R² · N · g            │
│              └────────────────────────┘                                     │
│                                                                              │
│  δθ_j = wrap(ψ̂_j − ψ̄)    N nodes    λ = Lyapunov exponent                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Electrochemical Analogy Map

| Battery concept       | PhaseBattery equivalent                    | Variable / formula               |
|-----------------------|--------------------------------------------|----------------------------------|
| Chemical potential    | Phase frustration                          | `E = (1/N) Σ δθ_j²`             |
| Electrolyte / medium  | EMA damping                                | `g = G_eff = sech(λ)`            |
| Electrode potential   | Mean circular phase                        | `ψ̄ = arg(⟨e^{iψ_j}⟩)`           |
| Terminal voltage      | Circular coherence                         | `R = |⟨e^{iψ_j}⟩| ∈ [0, 1]`    |
| Current flow          | Frustration released per step              | `ΔE = E_before − E_after`        |
| Internal resistance   | EMA gain reciprocal                        | `R_eff = cosh(λ) = 1/G_eff`      |
| Open circuit          | `g = 0`                                    | nothing moves, R constant        |
| Short circuit         | `g > 2` (i.e. `|1−g| > 1`)               | deviations amplify, E grows      |
| Useful operating zone | `g ∈ (0, 1]`                               | R non-decreasing, E decreasing   |
| **Lensing / focusing**| `metallic_oscillating_phases()`            | `amp = R · α`, output spread ≤ input |
| **Focal intensity**   | `interaction_energy(R, N, g)`             | `E_interact = R² · N · g`        |
| **Feedback loop**     | `feedback_step(α)`                         | `g_fb = g · α · R`, coherence feeds back into gain |

---

## 3. Base Specification

### 3.1 State

| Field    | Type                  | Meaning                                      | Invariant          |
|----------|-----------------------|----------------------------------------------|--------------------|
| `N`      | `int`                 | Number of phase nodes                        | `N ≥ 1`            |
| `g`      | `double`              | EMA gain = G_eff = sech(λ)                   | `g ≥ 0`            |
| `phases` | `vector<double>` (N) | Current phase estimate ψ̂_j for each node (rad) | updated each step |

### 3.2 Observables

| Method          | Returns  | Formula                                  | Monotonicity (g ∈ (0,1]) |
|-----------------|----------|------------------------------------------|--------------------------|
| `frustration()` | `double` | `(1/N) Σ wrap(ψ̂_j − ψ̄)²`               | non-increasing           |
| `circular_r()`  | `double` | `|Σ e^{iψ̂_j}| / N ∈ [0, 1]`            | non-decreasing           |
| `mean_phase()`  | `double` | `atan2(Σ sin ψ̂_j,  Σ cos ψ̂_j)` (rad)  | conserved (invariant)    |
| `step()`        | `double` | runs one EMA update; returns `ΔE ≥ 0`   | —                        |
| `feedback_step(α)` | `double` | two-sub-step EMA + coherence feedback; returns total `ΔE ≥ 0` | — |

### 3.3 Update Rule

```
For each node j = 0 … N−1:
    δθ_j  ← wrap(ψ̂_j − ψ̄)          // deviation from mean
    ψ̂_j  ← ψ̂_j − g · δθ_j           // EMA contraction toward ψ̄
```

Where `wrap(a) = a − 2π · floor((a + π) / (2π))` folds any angle into (−π, π].

### 3.4 Feedback Update Rule (`feedback_step`)

```
// Sub-step 1 — standard EMA contraction
For each node j:
    δθ_j  ← wrap(ψ̂_j − ψ̄)
    ψ̂_j  ← ψ̂_j − g · δθ_j

// Measure post-step coherence
R  ← |⟨e^{iψ̂_j}⟩|   // circular_r()

// Sub-step 2 — coherence-driven amplification pass
g_fb ← g · α · R      // feedback gain ∝ coherence (α is amplification factor)
For each node j:
    δθ_j  ← wrap(ψ̂_j − ψ̄)
    ψ̂_j  ← ψ̂_j − g_fb · δθ_j
```

Both sub-steps are independently dissipative contractions for `g ∈ (0, 1]` and
`α ≤ 1` (since `g_fb = g · α · R ≤ g · 1 · 1 ≤ 1`).  The system therefore
converges at least as fast as the standard `step()` and never diverges within
the stable operating zone.

**Key insight:** `g_fb` is largest when `R` is already high, meaning the
feedback amplification is strongest when phases are most aligned — exactly the
"lensing" regime where constructive interference is most effective.

---

## 3a. Generative Coherence Mechanics

### 3a.1 `metallic_oscillating_phases` (constructive-interference lensing)

Projects an input phase ensemble toward a focal alignment angle with an
amplitude proportional to the current circular coherence `R`:

```
amp  ← R · α                              // lensing amplitude = coherence × gain
For each node j:
    δ_j      ← wrap(ψ_j − θ_align)        // deviation from focal angle
    ψ_j^out  ← ψ_j − amp · δ_j            // project toward θ_align
```

| Parameter    | Type     | Meaning                                          |
|--------------|----------|--------------------------------------------------|
| `phases`     | `vector<double>` | Input phase ensemble                   |
| `align_angle`| `double` | Focal alignment angle θ_align (rad)              |
| `alpha`      | `double` | Coherence amplification factor α ≥ 0 (default 1)|

**Properties:**
- At `R = 1` (perfect coherence) and `α = 1`: phases already at focus, output = input.
- Output phase spread ≤ input phase spread (the function is a contraction).
- Higher `R` → stronger lensing (positive feedback of coherence into focusing power).

### 3a.2 `interaction_energy` (focal intensity scaling)

Computes the constructive-interference energy generated at the phase-medium
focal point, analogous to optical lensing intensity:

```
E_interact(R, N, g) = R² · N · g
```

| Value  | Meaning                                        |
|--------|------------------------------------------------|
| `R = 0` | No coherence — `E_interact = 0` (dark, no focus) |
| `R = 1` | Perfect coherence — `E_interact = N·g` (maximum focal energy) |

The `R²` factor mirrors the Born rule: energy/power scales as the square of the
coherence amplitude.  As `R → 1` the interaction energy saturates at `N·g`,
providing a predictable upper bound that scales linearly with the number of
nodes and the medium gain.

**Convergence profile (N=20, g=0.3):**

```
  R      E_interact
  0.1    0.06
  0.2    0.24
  0.3    0.54
  0.4    0.96
  0.5    1.50
  0.6    2.16
  0.7    2.94
  0.8    3.84
  0.9    4.86
  1.0    6.00   ← N·g = 20 × 0.3
```

---

## 4. Operating Regimes

```
g value       │ Regime        │ E trajectory  │ R trajectory  │ Analogy
──────────────┼───────────────┼───────────────┼───────────────┼────────────────────
g = 0         │ Open circuit  │ unchanged     │ unchanged     │ disconnected cell
0 < g < 1     │ Dissipative   │ decreasing    │ non-decreasing│ controlled discharge
g = 1         │ Full collapse │ → 0 one step  │ → 1 one step  │ all phases snap to ψ̄ (align, not invert)
1 < g ≤ 2    │ Over-shoot    │ oscillates    │ oscillates    │ ringing / resonance
g > 2         │ Unstable      │ growing       │ decreasing    │ short-circuit / runaway
```

### 4.1 Stability Boundary

The per-node contraction factor is `|1 − g|`.

- Stable (converging) when `|1 − g| < 1`, i.e. **`g ∈ (0, 2)`**.
- Marginally stable at `g = 2` (deviations flip sign but do not grow).
- Unstable when `g > 2`; each step multiplies the deviation by `|1 − g| > 1`.

### 4.2 Convergence Rate

Under the stable regime the frustration decays geometrically:

```
E(t) = E(0) · (1 − g)^{2t}
```

Steps to reach `R > threshold` scale as:

```
t_threshold  ≈  −log(1 − threshold) / (−2 log(1 − g))
            =   log(1 − threshold) / (2 log(1 − g))
```

For `g = sech(λ)`, faster convergence requires **smaller λ** (higher
conductance / lower Lyapunov exponent):

```
λ = 0.3  →  g ≈ 0.956  →  full coherence in ~1 step (N≥2 uniform spread)
λ = 2.0  →  g ≈ 0.266  →  requires ~11 steps to reach R > 0.95
```

---

## 5. Diagrams

### 5.1 Energy Flow Diagram

```
                   ┌───────────┐
  t=0              │  SOURCE   │   E(0) > 0  (phase nodes spread)
                   │  E(0)     │
                   └─────┬─────┘
                         │ g · E(0)  released each step
                         ▼
                   ┌───────────┐
  per step         │  MEDIUM   │   G_eff = sech(λ)
                   │  g        │   controls flow rate
                   └─────┬─────┘
                         │
                         ▼
                   ┌───────────┐
                   │   SINK    │   R(t) non-decreasing
                   │  ψ̄, R(t) │   ψ̄ invariant
                   └───────────┘
  t→∞              E → 0,  R → 1  (all nodes aligned with ψ̄)
```

### 5.2 State Trajectory (Phase Nodes)

```
  t = 0       t = 1       t = 2       t → ∞
  ψ₁ ●        ψ₁ ●        ψ₁●         ●
  ψ₂   ●      ψ₂  ●       ψ₂ ●        ●
  ψ₃  ●       ψ₃  ●       ψ₃ ●       ● ← all at ψ̄
  ψ₄    ●     ψ₄   ●      ψ₄  ●       ●
       ψ̄ ─── ψ̄ ──── ψ̄ ──── ψ̄   (conserved)
  R≈0.4        R≈0.65      R≈0.82     R≈1.0
```

### 5.3 Gain Sweep — Three Regimes

```
  Frustration E(t)

  E │
    │■■■■■■■■■■■■■■■■■■■■   g = 0    (open circuit, flat)
    │                        
    │■■                      g = 1.0  (full collapse to ψ̄ in 1 step)
    │                        
    │■■■■■                   g = 0.3  (gradual decay)
    │        ■■■■■■■■■■■■■   g = 2.5  (unstable — E grows)
    └──────────────────────► t (steps)
```

### 5.4 Circular Coherence R(t) — gain comparison

```
  R(t)
  1.0 ─────────────────────────────────────────────
      ·                                   g=1.0 ●──
  0.9 ·                          g=0.7 ●──────
      ·               g=0.3 ●───────
  0.5 ·        g=0.1 ●────────────────────────────
      ·
  0.0 ●────────────────────────────────────────────►
      0        5        10       15       20    t (steps)
```

---

## 6. G_eff Mapping (sech curve)

The medium parameter `g = G_eff = sech(λ)` maps the Lyapunov exponent λ to
a gain in `(0, 1]`:

```
  G_eff
  1.0 ●
      │ ●
  0.8 │   ●
      │      ●
  0.6 │          ●
      │               ●
  0.4 │                    ●
      │                          ●
  0.2 │                                  ●
      │                                              ●
  0.0 └─────────────────────────────────────────────►
      0   0.5   1.0   1.5   2.0   2.5   3.0        λ

  Stable operating zone:  λ ∈ [0, ∞)  →  g ∈ (0, 1]  (always stable)
  Instability onset only if g is set directly > 2 (λ has no such image)
```

*Note: setting `g = sech(λ)` guarantees `g ∈ (0, 1]` for any real λ, so
the battery always remains in the controlled-discharge regime when driven
through its natural Lyapunov parameterisation.*

---

## 7. Invariants and Guarantees

| Property                                | Condition               | Proof sketch                                                          |
|-----------------------------------------|-------------------------|-----------------------------------------------------------------------|
| `mean_phase()` conserved                | any `g`                 | `Σ δθ_j = 0` by definition ⇒ mean update cancels                     |
| `frustration()` non-increasing          | `g ∈ (0, 1]`            | `(1−g)² < 1` ⇒ each step shrinks `Σδθ_j²`                            |
| `circular_r()` non-decreasing           | `g ∈ (0, 1]`            | phases converge ⇒ vector sum magnitude grows                          |
| `circular_r() = 1` iff `E = 0`         | `N ≥ 1`                 | all phases equal ⟺ no frustration ⟺ R at maximum (1)                  |
| `step()` returns `ΔE ≥ 0`              | `g ∈ (0, 1]`            | follows from non-increasing frustration guarantee                     |
| `frustration() = 0` for flat init      | any `g`                 | `δθ_j = 0 ∀ j` ⇒ dead battery, nothing to release                    |
| `feedback_step()` stable               | `g ∈ (0,1]`, `α ∈ (0,1]` | `g_fb = g·α·R ≤ g ≤ 1` ⇒ both sub-steps are dissipative contractions |
| `feedback_step()` converges ≥ `step()` | `g ∈ (0,1]`, `α > 0`   | second sub-step always shrinks residual frustration further           |
| `R = 1` is a fixed point of `feedback_step()` | any `g`, `α` | zero frustration ⇒ all δθ_j = 0 ⇒ both sub-steps leave phases unchanged |
| `interaction_energy` monotone in R     | any `N`, `g`            | `R² · N · g` is strictly increasing in `R` for `N·g > 0`             |
| `metallic_oscillating_phases` contracts | `R·α ≤ 1`              | output spread = `(1 − R·α)` × input spread ≤ input spread            |
| Mirror symmetry preserved              | symmetric init, any `g`, `α` | EMA and feedback gains are uniform ⇒ symmetric deviation cancels symmetrically |

---

## 8. API Quick Reference

```cpp
#include "ohm_coherence_duality.hpp"
using namespace kernel::ohm;

// Construct: N=8 nodes, λ=0.5 → g=sech(0.5)≈0.887, uniform spread
double g = conductance(0.5);                       // sech(0.5) ≈ 0.887
std::vector<double> init = { /* 8 phases */ };
PhaseBattery bat(8, g, init);

// Observables (read-only):
double E       = bat.frustration();    // source energy
double R       = bat.circular_r();     // sink signal
double psi_bar = bat.mean_phase();     // conserved attractor

// Standard step — SOURCE → MEDIUM → SINK transfer:
double delta_E = bat.step();           // returns ΔE ≥ 0

// Feedback step — standard EMA + coherence-amplified second pass:
double delta_E_fb = bat.feedback_step(/*alpha=*/0.5);
// alpha=0 → identical to step(); alpha=1.0 → maximum amplification

// Phase lensing — project phases toward focal alignment angle:
bat.phases = metallic_oscillating_phases(
    bat.phases,           // input ensemble
    bat.mean_phase(),     // focal angle (e.g. current mean)
    /*alpha=*/0.5         // coherence amplification factor
);
// Output spread ≤ input spread (contraction); amp = R·alpha

// Interaction energy — focal intensity at current coherence:
double E_i = interaction_energy(bat.circular_r(), bat.N, bat.g);
// = R² · N · g; equals N·g at perfect coherence (R=1)

// Multi-phase stacked simulation:
for (int iter = 0; iter < 10; ++iter) {
    bat.phases = metallic_oscillating_phases(bat.phases, bat.mean_phase(), 0.5);
    bat.feedback_step(0.5);
}
// R converges to > 0.99 within ~4 iterations for moderate spread
```

---

## 9. Related Documents

| Document | Relation |
|---|---|
| `ohm_coherence_duality.hpp` | Source implementation of `PhaseBattery`, `metallic_oscillating_phases`, `interaction_energy` |
| `test_battery_analogy.cpp` | 37-assertion empirical proof suite (tests 1–6: base model; tests 7–9: feedback + lensing) |
| `experiments/kernelsync_demo/grover_analogy.md` | Grover-diffusion interpretation of the EMA update |
| `docs/scaling_laws.md` | Lyapunov exponent scaling and channel capacity |
| `docs/B11_palindrome_precession.md` | Conserved-quantity analysis pattern reference |
