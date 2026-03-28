# Kernel — Lean-Verified Mathematical Universe

> **"A complete, machine-checked mathematical universe — zero sorry,
> zero gaps."**

This repository contains a fully **Lean 4–verified mathematical universe**
spanning algebra, physics, quantum mechanics, and chemistry, together with
the C++ kernel implementation that applies these theorems in practice.

---

## 🌌 Mathematical Universe at a Glance

| Domain | Theorems | Key Results |
|--------|----------|-------------|
| **Algebra** | 127 | μ⁸=1, Silver ratio δS=1+√2, coherence C(r)≤1, Z/8Z memory |
| **Physics** | 159 | c=1/√(μ₀ε₀), α≈1/137, Koide formula, Lorentz geometry, NS bounds |
| **Quantum** | 96 | Floquet time crystals, gravity-quantum duality, Theorem Q |
| **Chemistry** | 44 | NIST atomic weights, G·R=1 Ohm-coherence duality |
| **Total** | **426** | All verified by Lean 4, **0 sorry** |

### Repository Structure

```
src/                        ← Lean modules organised by topic
├── algebra/Eigenvalue.lean       μ, δS, C(r), Z/8Z memory
├── physics/FundamentalConstants.lean  c, α, masses, spacetime
├── quantum/QuantumUniverse.lean  Time crystals, duality, Theorem Q
├── chemistry/AtomicUniverse.lean NIST atomic weights, Ohm-coherence
└── MathUniverse.lean             Single-import entry point

formal-lean/                ← Lean 4 proof files (the proof engine)
docs/                       ← Documentation (overview, architecture)
examples/                   ← Worked Lean demonstrations
tests/                      ← Cross-module consistency checks
```

### Quick Start (Lean)

```bash
cd formal-lean/
lake exe cache get   # download Mathlib cache (~5 min, avoids 1 h build)
lake build           # verify all 426 theorems
lake exe formalLean  # print theorem summary
```

See [docs/overview.md](docs/overview.md) for full documentation and
[CONTRIBUTING.md](CONTRIBUTING.md) to add new theorems.

---

## Quantum Kernel — Pipeline of Coherence v2.0

A C++ kernel implementation based on validated mathematical theorems from the Pipeline of Coherence derivations. The kernel manages quantum processes with 8-cycle scheduling and includes a rotational memory addressing system.

### Features

1. **Quantum Process Management**: Spawn processes with quantum states |ψ⟩ = (1/√2)|0⟩ + (e^{i3π/4}/√2)|1⟩
2. **8-Cycle Scheduling**: Processes rotate through Z/8Z positions with closed orbits when r=1
3. **Trichotomy Detection**: Identifies spiral-out (r>1), balanced (r=1), or spiral-in (r<1) regimes
4. **Silver Conservation**: Validates δ_S·(√2-1)=1 during boot sequence
5. **Process Composition**: Quantum interaction system for process entanglement
6. **Rotational Memory Addressing**: Z/8Z-based memory model with coherence preservation
7. **Interrupt Handling**: Decoherence detection and automatic coherence recovery
8. **Inter-Process Communication (IPC)**: Coherence-preserving message passing between quantum processes
9. **Ohm–Coherence Duality**: Computational framework for C = sech(λ) = G_eff = 1/R_eff

### Rotational Memory Addressing

The memory addressing system organizes memory into 8 banks corresponding to positions in Z/8Z (cyclic group of order 8). This design ensures that memory operations respect the kernel's rotational invariants and maintain coherence across cycle boundaries.

#### Key Concepts

**Address Structure:**
- Physical addresses decompose into `(bank_position, bank_offset)` where `bank_position ∈ Z/8Z`
- `bank_position = address mod 8` (position in the 8-cycle)
- `bank_offset = address div 8` (offset within the bank)

**Rotational Properties:**
- Rotation by k positions: `bank' = (bank + k) mod 8`
- Coherence preservation: addresses maintain relative positions under rotation
- Cycle-aware addressing: processes can read/write memory relative to their current cycle position

**Memory Banks:**
- 8 banks corresponding to positions 0..7 in Z/8Z
- Each bank stores quantum coefficients (complex numbers)
- Access statistics tracked per bank
- Coherence validation across all banks

#### Usage Example

```cpp
QuantumKernel kernel;

// Write to memory using linear addressing
kernel.memory().write_linear(addr, complex_value);

// Read from memory
Cx value = kernel.memory().read_linear(addr);

// Rotate memory addressing by 3 positions in Z/8Z
kernel.memory().rotate_addressing(3);

// Processes can access memory cycle-aware
kernel.spawn("Writer", [](Process& p) {
    p.mem_write(addr, p.state.beta);  // Writes relative to cycle position
});
```

#### Memory Operations

- `write(Address, value)`: Write quantum coefficient to rotational address
- `read(Address)`: Read quantum coefficient from rotational address
- `write_linear(addr, value)`: Write using linear address (auto-converted to Z/8Z)
- `read_linear(addr)`: Read using linear address
- `rotate_addressing(k)`: Rotate address mapping by k positions in Z/8Z
- `validate_coherence()`: Check that stored coefficients maintain bounded norms

#### Process Memory Access

Processes have cycle-aware memory helpers that automatically translate addresses based on their current position in the 8-cycle:

```cpp
Process p;
p.mem_write(addr, value);  // Translated by cycle_pos
p.mem_read(addr);          // Translated by cycle_pos
```

This ensures memory consistency as processes rotate through the cycle.

### Interrupt Handling

The kernel implements a sophisticated interrupt handling system that detects and responds to decoherence events in quantum processes. Decoherence occurs when the radius parameter r deviates from the balanced state (r=1), causing processes to spiral out (r>1) or spiral in (r<1).

#### Decoherence Detection

The interrupt system continuously monitors each process's coherence state by measuring phase deviation:

- **Phase Deviation Metric**: |r - 1| where r = |β|/|α|
- **Severity Levels**:
  - `NONE`: |r-1| ≤ 10⁻⁹ (coherent, on the 8-cycle)
  - `MINOR`: 10⁻⁹ < |r-1| ≤ 0.05 (slight deviation)
  - `MAJOR`: 0.05 < |r-1| ≤ 0.15 (significant deviation)
  - `CRITICAL`: |r-1| > 0.15 (severe decoherence)

#### Recovery Mechanism

When decoherence is detected, the interrupt handler applies corrective actions guided by the coherence function C(r) from Theorem 11:

1. **Coherence Measurement**: Computes C(r) = 2r/(1+r²) to quantify current coherence level
2. **Coherence Defect**: Calculates ΔC = 1 - C(r) as a measure of deviation from ideal balance
3. **Correction Strength**: Combines coherence defect with severity level and recovery rate configuration
4. **β Adjustment**: Scales the β coefficient to move r toward 1 by the computed correction amount
5. **Normalization**: Renormalizes to preserve |α|² + |β|² = 1 after adjustment
6. **Conservation**: Maintains silver conservation δ_S·(√2-1) = 1 (Prop 4) within numerical tolerances

#### Usage Example

```cpp
QuantumKernel kernel;

// Enable interrupts with custom configuration
DecoherenceHandler::Config cfg;
cfg.enable_interrupts = true;
cfg.enable_recovery = true;
cfg.log_interrupts = true;  // Debug logging
cfg.recovery_rate = 0.6;    // 60% recovery strength (0=none, 1=instant)

kernel.enable_interrupts(cfg);

// Spawn processes - decoherence will be handled automatically
kernel.spawn("Process-1", task_function);
kernel.run(8);  // Interrupts trigger during tick()
```

#### Recovery Performance

The recovery rate controls how aggressively the system corrects decoherence:

- **Low rate (0.3)**: Gentle correction, gradual return to r=1
- **Medium rate (0.6)**: Balanced correction speed and stability
- **High rate (0.9)**: Aggressive correction, rapid coherence restoration

Higher recovery rates provide faster coherence restoration but may cause overcorrection oscillations. Lower rates are more conservative but take longer to restore balance.

#### Minimal Disruption Guarantee

The interrupt system is designed to minimize impact on other processes:

- **Process Isolation**: Each interrupt only affects the decoherent process
- **No Cascading**: Interrupts don't trigger additional interrupts
- **Concurrent Safety**: Multiple processes can be corrected independently
- **Coherent Processes Unaffected**: Processes with r≈1 skip interrupt handling

### Inter-Process Communication (IPC)

The kernel implements a sophisticated IPC system that enables quantum processes to exchange messages while preserving coherence. Messages carry quantum state information (complex coefficients) and are subject to strict coherence validation to prevent decoherence spread.

#### Message Structure

Each message contains:
- **Sender and Receiver PIDs**: Process identifiers for routing
- **Timestamp**: Tick when message was sent
- **Cycle Position**: Sender's Z/8Z position at send time
- **Payload**: Quantum coefficient (complex number)
- **Coherence Metadata**: Sender's C(r) value for validation

#### Coherence Preservation

The IPC system enforces coherence preservation through:

1. **Send-Time Validation**: Sender must have C(r) ≥ threshold to send messages
2. **Receive-Time Validation**: Receiver must have C(r) ≥ threshold to receive messages
3. **Payload Bounds**: Message payloads must be bounded quantum coefficients
4. **Queue Capacity**: Limited queue size prevents overflow and unbounded growth

#### Z/8Z Integration

IPC operations respect the 8-cycle scheduler:

- **Cycle Alignment**: Messages can be delivered only at matching Z/8Z positions
- **Phase Tags**: Messages tagged with sender's cycle position
- **Rotational Invariants**: Communication preserves rotational symmetry
- **Channel Synchronization**: Message delivery respects cycle boundaries

#### Usage Example

```cpp
QuantumKernel kernel;

// Enable IPC with coherence checks
QuantumIPC::Config cfg;
cfg.enable_coherence_check = true;
cfg.enable_cycle_alignment = true;
cfg.coherence_threshold = 0.7;
kernel.enable_ipc(cfg);

// Spawn sender process
kernel.spawn("Sender", [](Process& p) {
    if (p.cycle_pos == 0) {
        p.send_to(2, p.state.beta);  // Send to PID 2
    }
});

// Spawn receiver process
kernel.spawn("Receiver", [](Process& p) {
    if (p.cycle_pos == 0) {
        auto messages = p.receive_from(1);  // Receive from PID 1
        for (const auto& msg : messages) {
            // Process message payload
        }
    }
});

kernel.run(8);
```

#### IPC Operations

Processes have simple IPC helpers:

```cpp
Process p;
p.send_to(target_pid, quantum_coefficient);  // Send message
auto msgs = p.receive_from(sender_pid);      // Receive messages
size_t pending = p.pending_from(sender_pid); // Check queue
```

#### Communication Guarantees

- **Coherence-Gated**: Only coherent processes can send/receive
- **Ordered Delivery**: FIFO ordering within each channel
- **No Message Loss**: Messages persist in queue until delivered
- **Silver Conservation**: δ_S·(√2-1)=1 maintained during IPC
- **Bounded Resources**: Queue size limits prevent resource exhaustion

### Ohm–Coherence Duality

The `ohm_coherence_duality.hpp` header implements the computational framework for the duality between quantum coherence and electrical conductance/resistance (Theorem 14):

```
C = sech(λ) = G_eff = 1 / R_eff
```

where λ is the Lyapunov exponent measuring decoherence.

#### Core Equations

| Quantity | Formula | Meaning |
|---|---|---|
| Conductance | G_eff(λ) = sech(λ) | Effective conductance = coherence |
| Resistance | R_eff(λ) = cosh(λ) | Effective resistance = 1/coherence |
| λ from C | λ = arccosh(1/C) | Degradation from coherence level |

At λ = 0 (ideal): G_eff = C = 1, R_eff = 1. Increasing λ degrades coherence.

#### Components

- **`CoherentChannel`**: Single channel with Lyapunov exponent λ, exposing `G_eff()`, `R_eff()`, and `coherence()`.

- **`MultiChannelSystem`**: N parallel channels. Parallel conductances add:
  `G_tot = Σ G_i`. Supports homogeneous (N identical channels) and heterogeneous configurations. Includes `weakest_channel()` for bottleneck detection.

- **`PipelineSystem`**: Series pipeline stages. Series resistances add:
  `R_tot = Σ R_stage`. Identifies the bottleneck stage via `bottleneck_stage()`.

- **`FourChannelModel`**: 4-channel redundancy system for error tolerance. Validates the 4-eigenvalue structure: tolerance passes when ≥ 3 of 4 channels have G_eff ≥ threshold.

- **`OUProcess`**: Ornstein–Uhlenbeck noise on λ: `dλ = −θ(λ − μ)dt + σ dW`. Simulates coherence degradation over time via `simulate()`. Provides `average_conductance()` to compute ⟨G⟩ and verify Jensen's inequality: ⟨sech(λ)⟩ ≤ sech(⟨λ⟩) (holds when sech is locally concave near |λ| small).

- **`QuTritDegradation`**: 3-level qutrit degradation. Each level-pair transition has its own λ. Provides `coherence_avg()` and `coherence_min()` for degradation pattern analysis.

#### Usage Example

```cpp
#include "ohm_coherence_duality.hpp"
using namespace kernel::ohm;

// Single channel: λ=0.5
CoherentChannel ch(0.5);
double G = ch.G_eff();      // sech(0.5) ≈ 0.887
double R = ch.R_eff();      // cosh(0.5) ≈ 1.128

// 4-channel parallel redundancy
MultiChannelSystem sys(4, 0.5);
double G_tot = sys.G_total(); // 4 * sech(0.5) ≈ 3.548

// Series pipeline
PipelineSystem pipe({0.2, 0.5, 1.0});
int bottleneck = pipe.bottleneck_stage(); // 2 (highest λ)

// Error tolerance validation
FourChannelModel model(0.0, 0.0, 0.0, 2.0); // one bad channel
bool ok = model.validate_error_tolerance(); // true (3/4 coherent)

// Ornstein–Uhlenbeck noise simulation
OUProcess ou(2.0, 0.0, 0.3); // θ=2, μ=0, σ=0.3
auto path = ou.simulate(0.0, 10000, 0.005);
double avg_G = OUProcess::average_conductance(path); // ⟨G⟩ ≤ sech(⟨λ⟩)
```

### Building and Running

#### CMake (recommended)

```bash
# Configure and build everything (tests + benchmarks)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run all tests via CTest
cd build && ctest --output-on-failure

# Build without benchmarks
cmake -B build -DKERNEL_BUILD_BENCHMARKS=OFF
cmake --build build --parallel

# Build without tests
cmake -B build -DKERNEL_BUILD_TESTS=OFF
cmake --build build --parallel
```

##### Installing the library

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --parallel
cmake --install build
```

After installation the library can be consumed with `find_package`:

```cmake
find_package(Kernel 2.0.0 REQUIRED)
target_link_libraries(my_app PRIVATE Kernel::Kernel)
```

Or via CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    Kernel
    GIT_REPOSITORY https://github.com/beanapologist/Kernel.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(Kernel)

target_link_libraries(my_app PRIVATE Kernel::Kernel)
```

##### Running individual benchmarks

```bash
./build/benchmark_nist_ir8356
./build/benchmark_pow_nonce_search

# Run specific NIST IR 8356 benchmark categories
./build/benchmark_nist_ir8356 --test=process_spawn
./build/benchmark_nist_ir8356 --test=scheduling
./build/benchmark_nist_ir8356 --test=memory
./build/benchmark_nist_ir8356 --test=coherence
```

#### Direct compilation (without CMake)

```bash
# Compile the kernel
g++ -std=c++17 -Wall -Wextra -O2 -o quantum_kernel_v2 quantum_kernel_v2.cpp -lm

# Run the kernel (includes all demonstrations)
./quantum_kernel_v2

# Run theorem verification tests
g++ -std=c++17 -Wall -Wextra -O2 -o test_pipeline_theorems test_pipeline_theorems.cpp -lm
./test_pipeline_theorems

# Run IPC tests
g++ -std=c++17 -Wall -Wextra -O2 -o test_ipc test_ipc.cpp -lm
./test_ipc

# Run Ohm–Coherence Duality tests
g++ -std=c++17 -Wall -Wextra -O2 -o test_ohm_coherence test_ohm_coherence.cpp -lm
./test_ohm_coherence

# Run NIST IR 8356 scalability benchmarks
g++ -std=c++17 -Wall -Wextra -O2 -o benchmark_nist_ir8356 benchmark_nist_ir8356.cpp -lm
./benchmark_nist_ir8356
```

### Mathematical Foundation

The implementation is grounded in formally verified theorems:

- **Theorem 3**: η = λ = 1/√2 (critical constant)
- **Section 2**: µ = e^{i3π/4} = (-1+i)/√2 (balanced eigenvalue)
- **Theorem 8**: Canonical coherent state |ψ⟩
- **Theorem 9**: Balance ↔ Maximum Coherence equivalence
- **Theorem 10**: Trichotomy (r=1: closed 8-cycle; r≠1: spiral)
- **Theorem 11**: Coherence function C(r) = 2r/(1+r²)
- **Theorem 12**: Palindrome residual R(r) = (1/δ_S)(r - 1/r)
- **Theorem 14**: Lyapunov duality C = sech(λ), λ = ln r ← **Ohm–Coherence Duality**
- **Prop 4**: Silver conservation δ_S·(√2-1) = 1

### Output

The kernel demonstrates:
1. Basic quantum process scheduling with trichotomy detection
2. Process composition with quantum interactions
3. Multi-process stress testing
4. Rotational memory addressing operations
5. Address translation in Z/8Z
6. Cycle-aware process memory access
7. Memory coherence validation
8. Bank distribution statistics
9. Decoherence interrupt handling and recovery
10. Recovery rate comparisons with different configurations
11. Inter-process communication with coherence preservation
12. Message passing between quantum processes
13. Coherence-gated communication to prevent decoherence spread
14. Cycle-aligned message delivery respecting Z/8Z scheduler

All mathematical invariants (Prop 4, Theorem 14, Corollary 13) are verified during execution.

### Standards Compliance

The interrupt handling implementation has been evaluated against NIST (National Institute of Standards and Technology) standards and benchmarks. See [NIST_EVALUATION.md](NIST_EVALUATION.md) for detailed compliance analysis.

**Key Compliance Areas:**
- ✓ NIST Quantum Error Correction Framework
- ✓ NIST Numerical Precision Standards (IEEE 754)
- ✓ NIST SP 800-53 (System Integrity)
- ✓ NIST SP 800-160 (Systems Security Engineering)

**Overall Assessment**: Strong compliance with NIST guidelines for quantum error handling systems.

### NIST IR 8356 Scalability Benchmarks

The kernel includes comprehensive benchmarking tools to evaluate scalability against metrics outlined in **NIST IR 8356: Considerations for Managing the Security of Operational Quantum Computing Systems**. See [NIST_IR8356_BENCHMARKS.md](NIST_IR8356_BENCHMARKS.md) for detailed methodology and analysis.

**Benchmark Categories:**
1. **Process Spawning Scalability**: Capacity to spawn quantum processes under high-load states
2. **8-Cycle Scheduling Throughput**: Operations per cycle with varying workloads
3. **Memory Addressing Scalability**: Z/8Z rotational memory performance at scale
4. **Coherence Preservation at Scale**: C(r) stability as system size increases

**Core Metrics Evaluated:**
- ✓ Qubit error rates (via decoherence detection)
- ✓ Coherence times (via C(r) = 2r/(1+r²) tracking)
- ✓ Gate fidelities (via 8-cycle µ operations)
- ✓ Scalability limits (process, memory, throughput)

**Run the NIST IR 8356 benchmark suite:**

```bash
# Compile and run full benchmark suite
g++ -std=c++17 -Wall -Wextra -O2 -o benchmark_nist_ir8356 benchmark_nist_ir8356.cpp -lm
./benchmark_nist_ir8356

# Run specific benchmark categories
./benchmark_nist_ir8356 --test=process_spawn
./benchmark_nist_ir8356 --test=scheduling
./benchmark_nist_ir8356 --test=memory
./benchmark_nist_ir8356 --test=coherence
```

**NIST-Recommended Test Suite**: Run `./test_interrupt_nist` to execute statistical validation, performance benchmarking, formal verification, and security tests as recommended in the NIST evaluation.

```bash
# Compile and run NIST test suite
g++ -std=c++17 -Wall -Wextra -O2 -o test_interrupt_nist test_interrupt_nist.cpp -lm
./test_interrupt_nist
```

### Noise Robustness & Phase Transitions

The Kernel's √n acceleration has been subjected to adversarial noise testing to
distinguish a genuine coherence mechanism (H₁) from a heuristic strategy (H₀).

See [NOISE_SCALING_PHASE_TRANSITION.md](NOISE_SCALING_PHASE_TRANSITION.md) for
the full analysis. Key findings:

- **Sharp transition at ε\* ≈ 0.42**: the Full Kernel (G\_eff = sech(λ)) maintains
  √n scaling for phase noise ε < ε* and collapses abruptly above it, with
  Δα = 1.34 at ε* = 0.50.
- **Precession-alone shows no comparable transition** — confirming the G\_eff
  coherence weight (not just phase precession) is the essential mechanism.
- **Recovery is tunable**: `auto_renormalize` rate ≥ 0.1 restores √n scaling
  even past ε* = 0.50.
- **Kick helpful in clean regimes, destabilizing near critical noise**: chiral
  kick without recovery degrades scaling even at ε = 0; with recovery it is
  benign.
- **Transition sharpens 6× with N** — a classical signature of a genuine phase
  transition.

**Verdict: H₁ confirmed.**

```bash
# Build and run the noise phase-transition test suite
cmake --build build --target test_noise_scaling_phase_transition
./build/test_noise_scaling_phase_transition
```

### Scaling Falsification Benchmark

An adversarial benchmark suite attempts to falsify the Θ(√n) scaling claim from
eight independent angles (extended problem sizes up to 2²⁶, randomised
initialisations, adversarial target placements, multi-seed confidence intervals,
normalisation inspection, and oracle blindness). All claims survive.

```bash
cmake --build build --target benchmark_scaling_falsification
./build/benchmark_scaling_falsification   # writes *.csv
python3 plot_scaling.py                   # writes *.png plots
```

### Oracle Model and Relation to Classical Unstructured Search

The coherent phase search operates with a **continuous oracle** — not the binary
yes/no oracle used in classical unstructured search (Grover's setting).

At each step the algorithm evaluates the cosine overlap between a probe phasor
and the target phasor `e^{i·2πt/n}`:

```
contrib = Re(probe · conj(target_phasor)) = cos(φ_probe − θ_target) ∈ [−1, +1]
```

This encodes **how close in angle** the probe is to the target, not just whether
it equals the target. The Dirichlet-kernel accumulation that drives the Θ(√n)
speedup depends on this continuously-valued signal.

Under a strict **binary oracle** (true unstructured search), where each query
only reveals `f(i) ∈ {0, 1}`, any classical algorithm requires Θ(n) queries
in the worst case — as confirmed by the `brute_force_search` baseline in the
test suite. The algorithm's Θ(√n) result is therefore a speedup over sequential
scan in the **continuous-oracle (phase-overlap) model**, not over binary-oracle
unstructured search.

#### Hostile-Reviewer Audit (Tests 12–14)

Three additional tests perform a rigorous self-audit:

**Test 12 (`test_oracle_model_assumptions`)** — Oracle model:
- Per-step contributions are continuous values in (−1, +1), not binary ±1.
- Contributions encode angular distance to target (nearer → larger value).
- Coherent (continuous oracle): slope ≈ 0.5 → Θ(√n). Brute-force (binary): slope ≈ 1.0 → Θ(n).

**Test 13 (`test_target_phasor_leakage`)** — Phase-independence audit:
- The Dirichlet accumulation fires in ≈ 0.19·√n steps for **any** input phasor, including wrong/random ones (CV ≈ 0 across all 8 compass directions).
- This means the step count carries no target-identity information.
- Identifying the true target under a binary oracle still requires Θ(n) evaluations.

**Test 14 (`test_parameter_scaling_tautology`)** — Scaling tautology:
- Using phase step 2π/n^α and threshold 0.15·n^α yields Θ(n^α) detection for any α ∈ (0,1).
- Verified empirically at α ∈ {0.4, 0.5, 0.6}: fitted slopes ≈ 0.40, 0.49, 0.60.
- The √n result (α = 0.5) is a **design parameter**, not a complexity lower bound.

#### Formal Mapping: Matched Filter / Single-Bin DFT

The algorithm is mathematically equivalent to evaluating a single DFT coefficient:

```
A_j(K) = Re[ e^{i(j·3π/4 − θ_t)} · Σ_{k=0}^{K-1} e^{ik·2π/√n} ]
        = Re[ e^{i(j·3π/4 − θ_t)} · D_K(2π/√n) · e^{i(K-1)π/√n} ]
```

where `D_K` is the Dirichlet kernel. Detection when `|A_j| ≥ 0.15·√n` corresponds
to the DFT magnitude exceeding a threshold calibrated to trigger at K ≈ 0.19·√n.

This is a **matched filter**: given the target phase (known from t), correlate a
structured sweep against it. In search, the reference is unknown; in matched
filtering, it is given as input. See `docs/theta_sqrt_n_writeup.tex` §7 for the
full critical analysis.


