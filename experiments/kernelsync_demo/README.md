# KernelSync Demo

A runnable simulation of **1000 nodes under heavy-tail packet-delay variation
(PDV)** that demonstrates an energy-proxy improvement using a structured pilot
burst code derived from the **μ 8-cycle + palindromic precession**.

---

## Background — What is KernelSync?

KernelSync is a network time-synchronisation scheme in which each leader node
transmits pilot bursts whose chip phases are drawn from a deterministic
schedule:

```
φ(n) = ( (3/8) · (n mod 8)  +  (n mod D) / D ) mod 1   [turns]
```

where:
- `n` is the global chip index,
- `3/8` turns = `3π/4` rad is the μ-step increment per chip step,
- `D = 13,717,421` is the palindromic precession period,
- the 8-step repetition is the **μ 8-cycle**.

This creates a pseudo-noise-like yet **fully deterministic** spread-spectrum
waveform.  Because both the leader and follower know the code exactly, the
matched-filter correlation peak at the receiver is much sharper under
heavy-tail PDV than an all-ones (BPSK DC) baseline code, allowing the follower
to estimate the integer chip-offset arrival time with fewer errors.

### Honest caveat

The deterministic code **does not eliminate clock drift** on its own.  What it
does is make the timing estimator more robust, so you can achieve the same RMS
synchronisation target with a *lower pilot overhead*:

> Lower pilot overhead = lower R (events/sec) or lower M (chips/event),
> both of which reduce the energy proxy E = R × M.

---

## PDV model

The simulation uses a **heavy-tail mixture**:

| Event     | Probability | Distribution             |
|-----------|-------------|--------------------------|
| Normal    | 99 %        | N(0, (2 ns)²)            |
| Outlier   | 1 %         | N(0, (50 ns)²)           |

---

## Node clock model

Each of the N follower nodes has:

- **Frequency skew** εᵢ ~ Uniform(−50, +50) ppm
- **Phase offset** θᵢ ~ Uniform(−1 µs, +1 µs)
- **Clock time** tᵢ(t) = (1 + εᵢ) t + θᵢ

The leader (node 0) is the reference: t₀(t) = t.

---

## Energy proxy E = R × M

| Symbol | Meaning                              |
|--------|--------------------------------------|
| R      | Pilot burst rate (events / second)   |
| M      | Chips per burst (chips / event)      |
| E      | Energy proxy = R × M                 |

A smaller E with the same RMS target means less bandwidth and power overhead.

---

## Simulation design

1. The **leader** transmits pilot bursts at rate R.
2. Each burst carries M chips separated by `Tc = 40 ns`.
3. The **follower** correlates the received chips against the known code over a
   ±250 ns (≈ ±6–7 chip) search window to estimate the integer chip offset τ̂.
4. The estimated arrival-time error is used to **slew** (not step-jump) the
   follower's offset and skew estimates with small gain constants.
5. The process repeats for `T_sim = 10 s`.

Two codes are compared:

| Scheme     | Code s_k                           |
|------------|------------------------------------|
| Baseline   | s_k = 1  (all-ones BPSK DC)        |
| KernelSync | s_k = exp(i 2π φ(n₀ + k))         |

---

## How to run

```bash
# Install dependencies
pip install -r experiments/kernelsync_demo/requirements.txt

# Quick smoke test (small grid, few nodes)
python experiments/kernelsync_demo/kernelsync_energy_proxy_sim.py \
    --nodes 10 --tsim 5 --grid-R "100,500" --grid-M "8,32" --seed 0

# Full 1000-node run (may take several minutes)
python experiments/kernelsync_demo/kernelsync_energy_proxy_sim.py

# Custom grid
python experiments/kernelsync_demo/kernelsync_energy_proxy_sim.py \
    --nodes 1000 \
    --tsim 10 \
    --tc 40e-9 \
    --window 250e-9 \
    --target-rms-ns 1.0 \
    --grid-R "50,100,200,500,1000" \
    --grid-M "8,16,32,64" \
    --seed 42
```

CLI options:

| Option            | Default                     | Description                          |
|-------------------|-----------------------------|--------------------------------------|
| `--nodes`         | 1000                        | Number of follower nodes             |
| `--tsim`          | 10                          | Simulation duration (s)              |
| `--tc`            | 40e-9                       | Chip period (s)                      |
| `--window`        | 250e-9                      | Timing search window ±(s)            |
| `--target-rms-ns` | 1.0                         | RMS synchronisation target (ns)      |
| `--grid-R`        | "50,100,200,500,1000"       | Comma-separated pilot rates          |
| `--grid-M`        | "8,16,32,64"                | Comma-separated burst lengths        |
| `--seed`          | 42                          | Random seed                          |
| `--output-dir`    | `experiments/kernelsync_demo/out/` | Output directory              |
| `--n-times`       | 100                         | Time samples per node trace          |

---

## Outputs

All outputs are written to `experiments/kernelsync_demo/out/` (configurable).

| File                       | Description                                        |
|----------------------------|----------------------------------------------------|
| `pareto_energy_vs_error.png` | E vs final RMS scatter for both schemes          |
| `heatmap_baseline.png`     | Heat-map of RMS over (R, M) grid — Baseline        |
| `heatmap_kernelsync.png`   | Heat-map of RMS over (R, M) grid — KernelSync      |
| `rms_vs_time_best.png`     | RMS time error vs time for best operating points   |
| `results.json`             | Min-E operating points, metrics, improvement factor|

### What the plots mean

- **Pareto plot**: each point is one (R, M) pair.  Points *below* the red
  dashed line meet the target.  KernelSync points tend to cluster lower-left
  (smaller E, same or better RMS).
- **Heat-maps**: green border = meets target.  Darker = worse.
- **RMS vs time**: shows convergence of both schemes at their best operating
  point.

---

## Running tests

```bash
pytest experiments/kernelsync_demo/tests/ -v
```
