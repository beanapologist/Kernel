# Coherent Bitcoin Mining - Python Implementation

**Discovery Date:** 2026-02-22/23  
**Framework:** Theory of Everything Kernel  
**Major Finding:** Universal scaling limit α_max = 1 + 1/e

## Overview

Python implementation of coherent Bitcoin mining using the Kernel framework. Includes full parameter sweep analysis (5,040 combinations) that discovered the 1 + 1/e universal bound.

## Components

### Mining Implementations
- `bitcoin_mainnet_miner.py` - Production mainnet miner with Bitcoin Core integration
- `coherent_regtest_miner.py` - Regtest validation environment (blocks found in seconds)
- `agent_sweep_runner.py` - Individual agent parameter sweep executor
- `agent_coordinator.py` - 8-agent orchestration and aggregation

### Analysis Scripts
- `verify_ceiling.py` - Confirms α_max = 1 + 1/e (0.057% error)
- `coherence_analysis.py` - Sweet spot characterization (C_opt = 0.817 ± 0.270)
- `precision_analysis.py` - Phase symmetry and noise-coherence trade-offs
- `global_analysis.py` - Aggregate 5,040-point sweep data

## Key Discoveries

### 1. Universal Scaling Limit
**α_max = 1 + 1/e ≈ 1.3679** (observed: 1.367099, error: 0.057%)

All 8 agents converged regardless of parameters. Physical origin: e-folding damping constant.

### 2. Stochastic Resonance
**C_opt = 0.817 ± 0.270** (sweet spot: [0.30, 0.83])

Moderate coherence outperforms pure order (C=1.0). Structure + noise > rigid perfection.

### 3. Phase Asymmetry
Weak 180° invariance (max asymmetry 0.292 bits at 135°↔315°). Suggests SHA-256 directional bias.

## Data

Full experimental data (636KB archive):
- 8 CSV files: 5,040 parameter combinations
- JSON exports: optima, curves, grid, plot data
- Documentation: discoveries, precision results, X thread

## Quick Start

```bash
# Regtest validation (instant blocks)
python3 coherent_regtest_miner.py 60 20

# Mainnet (requires Bitcoin Core)
python3 bitcoin_mainnet_miner.py 24.0
```

See `MINING_GUIDE.md` for full deployment instructions.
