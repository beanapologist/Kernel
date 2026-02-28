# BFT Stress Tests — Kernel Coherence Model

This document describes the Ethereum BFT stress-test suite
(`test_bft_stress.cpp`) and how to build, run, and interpret the results.

## Overview

`test_bft_stress.cpp` evaluates the resilience of the Kernel repository's
coherence model (`KernelState`, `PhaseBattery`, `PalindromePrecession`)
against Byzantine Fault Tolerance (BFT) mechanisms modelled on Ethereum's
consensus layer (Gasper / Tendermint-style).

### BFT Simulation Layer

| Component | Description |
|-----------|-------------|
| `BftNode` | Single validator wrapping a `KernelState`. Supports crash, phase-fault, amplitude-corruption, and vote-delay injection. |
| `BftEnvironment` | N-node round-based consensus network with a ⌊2N/3⌋+1 quorum rule and per-round finality tracking. |
| `FaultKind` | Enumeration of injectable fault types: `CRASH`, `PHASE_FAULT`, `AMPLITUDE_CORRUPTION`, `MESSAGE_DELAY`. |

### Integration with the Kernel Coherence Model

`BftNode::tick()` calls `SpectralBridge::step(state, KernelMode::FULL)` each
round, exercising the full pipeline: µ-rotation, `PalindromePrecession`, and
`auto_renormalize`.  A node's "vote" is its `KernelState::all_invariants()`
result — the coherence analogue of a valid block proposal.

Recovery is performed via `BftNode::try_recover()`, which iterates
`KernelState::auto_renormalize()` until `has_drift()` returns false (or up to
100 iterations), guaranteeing convergence for any initial r.

## Test Scenarios

| # | Test | What it validates |
|---|------|-------------------|
| 1 | **Quorum Safety** | 2f+1 honest nodes maintain coherence; f crashed nodes do not break quorum. |
| 2 | **Phase Faults** | Single-node phase corruption is detected (`has_drift()`) and recovered (`auto_renormalize`). |
| 3 | **Node Failures** | Up to f=3 crashed nodes; network liveness preserved over 20 rounds. |
| 4 | **Delayed IPC** | Nodes with withheld votes for D rounds; quorum survives; coherence maintained. |
| 5 | **State Corruption** | `beta_unit_invariant` detects injected drift; iterative renorm restores r→1 for individual states and mass-corrupted networks. |
| 6 | **Finality** | `committed_rounds` never decreases after fault injection (safety). Post-fault liveness verified. |
| 7 | **Recovery Rate** | Mean coherence convergence measured over 30 recovery rounds after mass phase fault. |
| 8 | **Liveness & Safety** | Combined crash + delay + phase fault on N=13 network; safety and liveness properties verified simultaneously. |

## Build

```bash
g++ -std=c++17 -Wall -Wextra -O2 -o test_bft_stress test_bft_stress.cpp -lm
```

No additional dependencies beyond the C++17 standard library and the existing
Kernel headers.

## Run

```bash
./test_bft_stress
```

Exit code 0 indicates all assertions passed.  Non-zero indicates at least one
failure (the failing assertion name is printed to stdout).

### Example output

```
══════════════════════════════════════════════════════
  Kernel BFT Stress Tests
══════════════════════════════════════════════════════

╔═══ 1. Quorum Safety ═══╗
  ✓ N=7: quorum threshold = 5 (2f+1 with f=2)
  ✓ 10 rounds with 0 faults: all rounds committed
  ...

╔═══ 7. Recovery Rate ═══╗
  ✓ mean coherence improves over recovery rounds
  ✓ coherence converges monotonically toward final value
  ✓ mean coherence > 0.95 after full recovery period
    post-fault C=0.998811  final C=1.000000

══════════════════════════════════════════════════════
  Results: 39/39 passed  [0.6 ms]
══════════════════════════════════════════════════════
```

## Interpreting Results

| Metric | Meaning |
|--------|---------|
| `committed_rounds` | Rounds where ≥⌊2N/3⌋+1 nodes voted coherently — analogous to Ethereum finalized blocks. |
| `mean_coherence()` | Average `C = 2|α||β|` across live nodes; ideal value is 1.0. |
| `coherent_fraction()` | Fraction of live nodes satisfying all three invariants (unit-circle, palindrome residual, R_eff). |
| `post-fault C` | Mean coherence immediately after mass fault injection (< 1 expected). |
| `final C` | Mean coherence after recovery (should approach 1.0). |

A recovery is considered **successful** when `all_invariants()` returns true
for a node — i.e. `|α|²+|β|²=1`, `R(r)≈0`, and `R_eff≈1` simultaneously.

## CI Integration

The test is registered in:

- **`CMakeLists.txt`** — added to `_kernel_tests` so `ctest` picks it up.
- **`.github/workflows/ci.yml`** — added to:
  - `lint`: clang-format compliance check.
  - `build-and-test`: compiled and executed on every push / PR.
  - `coverage`: compiled with `--coverage` and executed for lcov data.

The test completes in under 5 ms and does not require OpenSSL or Eigen.
