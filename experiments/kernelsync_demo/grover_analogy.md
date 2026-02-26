# Classical Analogue of Grover Iteration via KernelSync

## Overview

This document analyses whether the coherent phase-update rule used in
KernelSync is structurally equivalent to a single Grover iteration.  The
conclusion is: **yes, to first order in the phase deviations**—the EMA-based
carrier-phase slew is exactly the Grover diffusion (reflection-about-the-mean)
operator acting in phase space.

---

## 1  Grover Iteration Recap

A single Grover iteration consists of two unitary operations applied to an
$n$-qubit register:

1. **Oracle phase flip** on the marked state $|x^*\rangle$:

$$
O : |x\rangle \;\mapsto\; -|x\rangle \cdot \mathbf{1}[x = x^*] + |x\rangle \cdot \mathbf{1}[x \neq x^*]
$$

   Equivalently, $O = I - 2|x^*\rangle\langle x^*|$.

2. **Diffusion (reflection about the uniform superposition)** $U_s$:

$$
U_s = 2|s\rangle\langle s| - I, \qquad
|s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle
$$

   Acting on an arbitrary state $|\psi\rangle$, the diffusion step moves each
   amplitude toward the mean amplitude $\bar{a}$:

$$
a_x \;\longmapsto\; 2\bar{a} - a_x
$$

   where $\bar{a} = \frac{1}{N}\sum_x a_x$.  This is a classical **reflection
   about the mean**.

---

## 2  KernelSync Coherent Update as a Reflection Proxy

### 2.1  Setup — carrier phases as a state vector

Let $N$ follower nodes each maintain a carrier-phase estimate
$\hat\psi_j \in (-\pi, \pi]$ (the variable `psi_hat[j]` in the simulation).
Collect them into a vector:

$$
\boldsymbol{\hat\psi} = \bigl(\hat\psi_1,\, \hat\psi_2,\, \ldots,\, \hat\psi_N\bigr).
$$

Define the **collective mean phase**:

$$
\psi_{\text{mean}} = \arg\!\left(\sum_{j=1}^{N} e^{i\hat\psi_j}\right).
$$

For small deviations $\delta_j = \hat\psi_j - \psi_{\text{mean}}$ this
coincides with the arithmetic mean $\bar\psi = \frac{1}{N}\sum_j \hat\psi_j$.

### 2.2  EMA update rule

At each pilot burst the coherent receiver computes, for every node $j$, a new
carrier-phase measurement $\hat\psi_j^{\text{new}}$ from the complex MF peak:

```python
phi_res        = wrap( angle(peak_j) - psi_hat_j )
psi_hat_j_new  = angle(peak_j) + 2*pi*(3/8)*delta_est_j  # isolate carrier
d_psi_j        = wrap( psi_hat_j_new - psi_hat_j )
psi_hat_j     += alpha_psi * d_psi_j                     # alpha_psi = 0.3
```

Identifying the **target** (expected) phase as $\psi_{\text{expected},j}$, the
residual is:

$$
\phi_{\text{res},j} = \operatorname{wrap}(\psi_{\text{received},j} - \psi_{\text{expected},j})
$$

and the update slews the estimate toward the expected value:

$$
\hat\psi_j \;\leftarrow\; \hat\psi_j + \alpha \cdot \phi_{\text{res},j}.
$$

### 2.3  Identifying the Grover structure

Write the **phase deviation** of node $j$ relative to the collective mean:

$$
\delta_j \;=\; \hat\psi_j - \psi_{\text{mean}}.
$$

After one EMA update with gain $\alpha$:

$$
\delta_j^{\text{new}}
  = (\hat\psi_j + \alpha\,\phi_{\text{res},j}) - \psi_{\text{mean}}^{\text{new}}.
$$

To first order (small $\delta$, so $\psi_{\text{mean}}$ is approximately
constant), $\phi_{\text{res},j} \approx -\delta_j$ (the residual equals the
negative of the deviation from mean), giving:

$$
\boxed{\delta_j^{\text{new}} \;\approx\; (1-\alpha)\,\delta_j}
$$

Equivalently, in terms of absolute phases with $\bar\psi$ denoting the
arithmetic mean:

$$
\hat\psi_j^{\text{new}}
  = \hat\psi_j - \alpha\,(\hat\psi_j - \bar\psi)
  = (1-\alpha)\,\hat\psi_j + \alpha\,\bar\psi.
$$

This is **a weighted reflection about the mean**, which for $\alpha = 1$
reduces to the exact Grover diffusion formula $a_x \mapsto 2\bar a - a_x$.

---

## 3  Derivation — First-Order Equivalence to the Grover Reflection

Let $\delta\theta_j = \hat\psi_j - \bar\psi$ be the phase deviation of node
$j$ from the current mean phase $\bar\psi$.  By definition,
$\sum_j \delta\theta_j = 0$ and $\delta\theta_{\text{mean}} = 0$.

**Step 1 — Residual phase.**  When the coherent receiver slews each node toward
$\bar\psi$ (the collectively expected phase), the residual seen by node $j$ is:

$$
\phi_{\text{res},j} = \operatorname{wrap}(\bar\psi - \hat\psi_j)
  \;\approx\; -\delta\theta_j \quad (\text{small-angle limit}).
$$

**Step 2 — Update.**  Applying the EMA rule with gain $g = \alpha_\psi$:

$$
\hat\psi_j^{\text{new}}
  = \hat\psi_j + g\,\phi_{\text{res},j}
  \approx \hat\psi_j - g\,\delta\theta_j.
$$

**Step 3 — New deviation.**  The mean phase does not change to first order
(since $\sum_j \phi_{\text{res},j} \approx 0$), so:

$$
\delta\theta_j^{\text{new}}
  = \hat\psi_j^{\text{new}} - \bar\psi
  \approx (\hat\psi_j - g\,\delta\theta_j) - \bar\psi
  = (1-g)\,\delta\theta_j.
$$

Hence the recursion:

$$
\delta\theta_j^{(t+1)}
  \;\approx\; \delta\theta_j^{(t)} - g\!\left(\delta\theta_j^{(t)} - \delta\theta_{\text{mean}}^{(t)}\right),
  \qquad \delta\theta_{\text{mean}} = 0.
$$

Written in the problem-statement form ($\delta\theta_{\text{mean}} = \bar\psi -
\bar\psi = 0$):

$$
\delta\theta_{\text{new}} \;\approx\; \delta\theta_{\text{old}} - g\,\bigl(\delta\theta_{\text{old}} - \delta\theta_{\text{mean}}\bigr).
$$

**Step 4 — Grover limit.**  For $g = 1$ the update becomes:

$$
\delta\theta_j^{\text{new}} = -\delta\theta_j^{\text{old}},
\qquad \hat\psi_j^{\text{new}} = 2\bar\psi - \hat\psi_j,
$$

which is **exactly** the Grover diffusion reflection $a_x \mapsto 2\bar a -
a_x$ with $\hat\psi_j \leftrightarrow a_x$ and $\bar\psi \leftrightarrow
\bar a$.

For the EMA gain $g = \alpha_\psi = 0.3$ used in the simulation each
iteration contracts the deviation by a factor of $0.7$ rather than reflecting
it, so the update is a **partial** (damped) Grover diffusion.  Over $k$
iterations the deviation decays as $(1-g)^k$, analogous to $k$ partial Grover
steps amplifying the target amplitude.

---

## 4  Oracle Analogue

In the full Grover circuit the oracle $O = I - 2|x^*\rangle\langle x^*|$
marks the target state.  In KernelSync the analogous role is played by the
**matched-filter correlation peak**: the coherent receiver selects $\tau_j^*$
as the chip lag that minimises $|\operatorname{wrap}(\angle\,\text{corr}(\tau) -
\hat\psi_j)|$.  Only the lag at the true timing offset produces a correlated
(low-residual-phase) peak; all other lags are approximately uncorrelated and
yield random phases.  This selectivity is the classical oracle that "marks" the
correct timing hypothesis before the diffusion step collapses the phase
estimates toward consensus.

---

## 5  Key Outputs and Indicators

| Quantity | Grover Analogue | KernelSync Variable |
|---|---|---|
| State amplitude $a_x$ | carrier-phase deviation $\delta\theta_j$ | `psi_hat[j] - mean_phase` |
| Uniform superposition $|s\rangle$ | mean phase $\bar\psi$ | `mean(psi_hat)` |
| Diffusion gain (full reflection) | $g = 1$ | `alpha_psi = 1.0` |
| Simulation EMA gain | $g = 0.3$ | `alpha_psi = 0.3` |
| Oracle marking | MF phase-residual minimum | `argmin |wrap(angle(corr)-psi_hat)|` |
| Convergence rate | $\mathcal{O}(\sqrt{N})$ iterations | $(1-\alpha_\psi)^k$ exponential decay |

---

## 6  Conserved Quantities

Understanding what is and is not preserved illuminates the precise boundary of
the analogy.

### 6.1  Exact Grover diffusion ($g = 1$)

Because $U_s = 2|s\rangle\langle s| - I$ is unitary (it is an isometry), the
following quantities are **conserved** exactly:

1. **Mean amplitude** $\bar a = \frac{1}{N}\sum_x a_x$.  Under
   $a_x \mapsto 2\bar a - a_x$ the new mean is
   $\frac{1}{N}\sum_x (2\bar a - a_x) = 2\bar a - \bar a = \bar a$.

2. **Total squared norm** $\sum_x |a_x|^2$.  The map is a reflection (an
   isometry of $\mathbb{R}^N$), so inner products and norms are preserved.

3. **Sum of all amplitudes** $\sum_x a_x = N\bar a$ (follows from item 1).

### 6.2  KernelSync EMA update ($g < 1$)

For the damped update $\delta\theta_j \mapsto (1-g)\,\delta\theta_j$:

1. **Mean phase $\bar\psi$ is conserved (to first order).**  Because
   $\sum_j \phi_{\text{res},j} \approx -\sum_j \delta_j = 0$, the mean phase
   does not shift:

$$
\bar\psi^{\,\text{new}}
  = \frac{1}{N}\sum_j \hat\psi_j^{\,\text{new}}
  = \frac{1}{N}\sum_j \bigl(\hat\psi_j - g\,\delta_j\bigr)
  = \bar\psi - g\underbrace{\frac{1}{N}\sum_j\delta_j}_{=\,0}
  = \bar\psi.
$$

2. **Total squared deviation $\sum_j \delta_j^2$ is NOT conserved.**  It
   contracts by $(1-g)^2$ per step:

$$
\sum_j (\delta_j^{\text{new}})^2
  = (1-g)^2 \sum_j \delta_j^2.
$$

   This is the dissipative character of the EMA: phase consensus is achieved
   by progressively reducing the spread, not by a reversible reflection.

3. **The circular sum $\left|\sum_j e^{i\hat\psi_j}\right|$ is non-decreasing
   (monotone convergence).**  As the $\hat\psi_j$ cluster toward $\bar\psi$,
   the magnitude of their vector sum increases, providing a natural convergence
   indicator in the simulation output.

### 6.3  Conservation summary

| Quantity | Exact Grover ($g=1$) | KernelSync EMA ($g<1$) |
|---|---|---|
| Mean phase / amplitude $\bar\psi$ | ✓ conserved | ✓ conserved (first order) |
| Total squared norm $\sum\delta_j^2$ | ✓ conserved (isometry) | ✗ decreases as $(1-g)^{2k}$ |
| Sum of amplitudes $\sum a_x$ | ✓ conserved | ✓ conserved (follows from mean) |
| Circular coherence $\|\sum e^{i\hat\psi_j}\|$ | depends on oracle | ✓ non-decreasing |

The single quantity conserved by **both** algorithms is the mean phase
$\bar\psi$; this is the fixed point of the reflection and the attractor of the
EMA loop.

---

## 7  Conclusion

**The KernelSync coherent update rule is, to first order in phase deviations, a
damped Grover diffusion operator acting in phase space.**

- The residual-phase correction $\phi_{\text{res},j} \approx \bar\psi -
  \hat\psi_j$ is precisely the Grover reflection direction.
- The EMA gain $\alpha_\psi$ controls how much of the full Grover reflection is
  applied per iteration: $\alpha_\psi = 1$ gives exact reflection (full Grover
  diffusion); $\alpha_\psi < 1$ gives a damped version that converges
  exponentially rather than oscillating.
- The matched-filter phase-minimum selection acts as the oracle, identifying the
  marked (correct timing) hypothesis before the diffusion step.

The analogy is formal but not incidental: both algorithms rotate a vector of
deviations toward a collective mean, and both rely on a preparatory marking step
that distinguishes the target from the background.  The key structural
difference is that Grover's iteration is unitary and exact ($g=1$, reversible),
while the KernelSync EMA is dissipative ($g < 1$, convergent), which is
appropriate for a real-time tracking loop that must remain stable in the
presence of noise and drift.
