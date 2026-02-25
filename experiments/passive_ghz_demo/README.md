# Passive GHz Demo — Drive-only Variant 1

## Experiment Overview

This package validates a **palindromic precession + μ 8-cycle phase program** for a
superconducting microwave resonator in reflection.  The waveform drives the resonator
via IQ modulation at baseband; no frequency conversion is needed beyond a standard IQ
mixer.

**Goal**: determine whether running the palindromic phase drive changes the measured
energy-decay rate κ compared to a matched control that omits the slow-precession
component.

---

## Hardware Assumptions

| Item | Specification |
|------|---------------|
| AWG  | Zurich Instruments HDAWG |
| Sample rate | 2.4 GS/s |
| Outputs | Two analogue channels → I and Q ports of an IQ mixer |
| Mixer topology | Reflection (circulator + resonator) |
| Drive power | Keep well below bifurcation / heating threshold |
| Readout | Separate VNA or homodyne IQ detector (not generated here) |

---

## Parameter Table

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Sample rate | f_s | 2.4 GS/s |
| Step duration | T_step | 40 ns |
| Samples per step | — | 96 |
| Number of steps | N_steps | 200 (configurable) |
| Slow-precession period | D | 13,717,421 steps |
| μ cycle length | — | 8 steps |
| Drive amplitude | A_drive | 0.5 (full scale = 1.0) |
| μ phase increment | φ_μ | 3π/4 per step |

---

## Phase Schedule

For each step n (0-indexed):

```
φ_μ(n)  = (3π/4) · (n mod 8)          # 8-cycle component
φ_p(n)  = 2π · n / D                  # slow palindromic precession
φ(n)    = φ_μ(n) + φ_p(n)             # palindromic variant
φ_ctrl(n) = φ_μ(n)                    # control variant (no precession)
```

The baseband envelope for each step is:

```
s[n] = A · exp(i φ(n))
I[n] = A · cos(φ(n))
Q[n] = A · sin(φ(n))
```

Each step produces 96 identical samples (flat-top pulse per step).

> **Note**: constant `|s| = A` does **not** imply constant `|β|` inside the resonator.
> The intra-cavity field `β(t)` depends on the drive history, detuning, and κ.

---

## Three-Run Measurement Protocol

```
Run 1 — Baseline ringdown
  1. Ensure drive is OFF.
  2. Excite resonator with a short calibration pulse.
  3. Record IQ ringdown for ≥ 5/κ seconds.
  4. Fit κ_baseline from log|β| vs time.

Run 2 — Palindromic drive, then ringdown
  1. Upload palindromic waveform (generate_waveforms.py --variant pal).
  2. Play waveform for full N_steps duration (≈ N_steps × 40 ns).
  3. Immediately after waveform ends (drive OFF), record IQ ringdown.
  4. Fit κ_after_pal from post-drive ringdown region.

Run 3 — Control drive, then ringdown
  1. Upload control waveform (generate_waveforms.py --variant ctrl).
  2. Play waveform for the same N_steps duration.
  3. Immediately after (drive OFF), record IQ ringdown.
  4. Fit κ_after_ctrl from post-drive ringdown region.
```

> **Critical**: κ must be fitted **only during the post-drive ringdown** (drive OFF).
> Fitting during active drive includes driven response and will give incorrect κ.

---

## Data Products and Plots

`analysis_ringdown.py` produces:

| Output | Description |
|--------|-------------|
| `kappa_baseline` | κ from Run 1 calibration ringdown |
| `kappa_after_pal` | κ from Run 2 post-drive ringdown |
| `kappa_after_ctrl` | κ from Run 3 post-drive ringdown |
| `delta_kappa` | (κ_after_pal − κ_after_ctrl) with bootstrap error |
| `phase_tracking_error.png` | Step-synchronous phase error during drive window |
| `amplitude_modulation.png` | |β(t)| amplitude modulation depth during drive |
| `ringdown_fits.png` | log|β| vs time with exponential fits overlaid |
| `kappa_summary.png` | Bar chart of κ values with error bars |

---

## File Format for Recorded Traces

Save IQ ringdown traces as NPZ files with the following arrays:

```python
np.savez("ringdown_baseline.npz",
    t      = t_array,      # time axis, seconds, shape (N,)
    I_rec  = I_rec_array,  # recorded I channel, shape (N,)
    Q_rec  = Q_rec_array,  # recorded Q channel, shape (N,)
    run    = "baseline",   # string tag
    fs_rec = 1e6,          # recording sample rate (Hz)
)
```

Alternatively, a CSV with columns `t,I_rec,Q_rec` is accepted (slower for large files).

---

## Safety Notes

- **Drive power**: start with A_drive ≤ 0.1 and increase slowly. Monitor device
  temperature. Excess power heats the mixing chamber and shifts κ.
- **Duration**: N_steps × 40 ns is typically < 10 µs. This is safe, but verify
  your specific device limits.
- **Mixer calibration**: LO leakage and IQ imbalance will add a DC offset and an
  image tone. Calibrate with the standard ZI IQ mixer calibration routine before
  running experiments.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Generate waveforms
python generate_waveforms.py --n-steps 200 --a-drive 0.5 --output waveforms.npz

# 3. Upload to HDAWG (fill in device serial in config_example.yaml first)
python upload_hdawg.py --config config_example.yaml --waveform waveforms.npz

# 4. (After recording ringdown traces) Analyse results
python analysis_ringdown.py \
    --baseline ringdown_baseline.npz \
    --pal      ringdown_pal.npz \
    --ctrl     ringdown_ctrl.npz \
    --config   config_example.yaml

# 5. Run unit tests
pytest tests/
```
