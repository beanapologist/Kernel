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
| `BftEnvironment` | N-node round-based consensus network with a ⌊2N/3⌋+1 quorum rule, per-round finality tracking, and optional `EthValidatorSyncHook`. |
| `FaultKind` | Enumeration of injectable fault types: `CRASH`, `PHASE_FAULT`, `AMPLITUDE_CORRUPTION`, `MESSAGE_DELAY`. |
| `EthValidatorSyncHook` | Abstract interface for fetching per-validator metadata (index, balance, coherence weight) from an Ethereum beacon chain. |
| `StubEthValidatorSyncHook` | Deterministic synthetic implementation used in CI — hermetic, no network access required. |
| `LiveEthValidatorSyncHook` | Production template; reads `KERNEL_ETH_TESTNET_RPC` env var and falls back to stub when not set. |

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
| 9 | **Validator Sync Hook** | `StubEthValidatorSyncHook` interface contract; `LiveEthValidatorSyncHook` fallback behaviour; `BftEnvironment::set_sync_hook` / `apply_sync_hook` integration with quorum. |

## Ethereum Testnet Validator Sync Hook

To connect to a live Ethereum testnet beacon chain:

1. Export the RPC endpoint:
   ```bash
   export KERNEL_ETH_TESTNET_RPC=https://rpc.sepolia.org
   ```
2. Subclass `EthValidatorSyncHook` and implement `fetch_validator_info()` to
   call `GET <rpc_url>/eth/v1/beacon/states/head/validators/<index>` and map
   the JSON response fields to `EthValidatorInfo`.
3. Pass an instance to `BftEnvironment::set_sync_hook()` before calling
   `apply_sync_hook()` to initialise node coherence weights from the beacon
   state.

When `KERNEL_ETH_TESTNET_RPC` is not set (e.g. in CI), `LiveEthValidatorSyncHook`
falls back automatically to `StubEthValidatorSyncHook` so all tests remain
hermetic and network-free.

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

╔═══ 9. Validator Sync Hook ═══╗
  ✓ stub hook: is_live() = false (hermetic, no network)
  ...
    stub endpoint : stub://kernel-bft-testnet-simulator
    live endpoint : live://not-configured (fallback to stub)

══════════════════════════════════════════════════════
  Results: 67/67 passed  [0.5 ms]
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

---

## Extended BFT Tests (`test_bft_extended.cpp`)

`test_bft_extended.cpp` builds upon the framework above to cover a wider range
of adversarial scenarios and provides structured CSV metric output.

### Additional Test Scenarios

| # | Test | What it validates |
|---|------|-------------------|
| 10 | **Scale Testing** | Networks from 7 to 100 nodes; quorum thresholds and coherence fractions recorded per scale point. |
| 11 | **Adversarial Beyond-Quorum** | Crashing more than f nodes causes quorum loss; a reduced-fault environment restores quorum. |
| 12 | **Live Sepolia Integration** | `SepoliaHook` validates coherence weights using Sepolia beacon-chain data (or stub fallback in CI). |
| 13 | **Cascading Faults** | Progressive per-wave crash injection; quorum maintained within BFT tolerance. |
| 14 | **Oscillating Faults** | Periodic fault/recovery cycles; coherence restores fully every cycle. |
| 15 | **Correlated Faults** | Simultaneous burst of crashes + phase faults; quorum recovery after repair. |
| 16 | **Mixed Failure Modes** | Crash + phase + delay + amplitude faults combined; safety and liveness verified. |

### CSV Metric Output

Each run emits `METRIC,<key>,<value>` lines to stdout for downstream tooling:

```
METRIC,scale_N7_quorum_threshold,5
METRIC,scale_N100_committed_rounds,10
METRIC,cascade_wave1_coherent_frac,1.000000
METRIC,mixed_N19_recovery_round_to_90pct,1
METRIC,total_passed,50
METRIC,elapsed_ms,0.500000
```

### Build

```bash
# Without libcurl (CI default — Sepolia stub fallback):
g++ -std=c++17 -Wall -Wextra -O2 -o test_bft_extended test_bft_extended.cpp -lm

# With libcurl (live Sepolia HTTP integration):
g++ -std=c++17 -Wall -Wextra -O2 -DHAVE_CURL \
    -o test_bft_extended test_bft_extended.cpp -lcurl -lm
```

Set `KERNEL_ETH_TESTNET_RPC=https://rpc.sepolia.org` to activate live beacon
chain data when building with `-DHAVE_CURL`.

## CI Integration

Both test files are registered in:

- **`CMakeLists.txt`** — added to `_kernel_tests` so `ctest` picks them up.
- **`.github/workflows/ci.yml`** — added to:
  - `lint`: clang-format compliance check.
  - `build-and-test`: compiled and executed on every push / PR.
  - `coverage`: compiled with `--coverage` and executed for lcov data.

The extended test suite completes in under 5 ms and does not require OpenSSL,
Eigen, or libcurl for CI runs.
