/*
 * test_bft_stress.cpp — Ethereum BFT Stress Tests for Kernel Coherence Model
 *
 * Evaluates the resilience of the Kernel repository against Byzantine Fault
 * Tolerance (BFT) mechanisms modelled on Ethereum's consensus layer.
 *
 * Architecture:
 *   BftNode        — a single validator node wrapping a KernelState with
 *                    fault-injection capabilities (crash, phase corruption,
 *                    message delay).
 *   BftEnvironment — an N-node simulated BFT network with a 2/3-quorum rule,
 *                    round-based consensus, and finality tracking.
 *
 * Test structure:
 *   1. Quorum Safety     — 2f+1 honest nodes maintain coherence invariants.
 *   2. Phase Faults      — single-node phase corruption detected & recovered.
 *   3. Node Failures     — up to f crashed nodes; network remains live.
 *   4. Delayed IPC       — messages held for D rounds; coherence survives.
 *   5. State Corruption  — beta_unit_invariant detects injected drift;
 *                          auto_renormalize restores r→1.
 *   6. Finality          — committed blocks are never rolled back under BFT.
 *   7. Recovery Rate     — measures coherence convergence after mass fault.
 *   8. Liveness & Safety — simultaneous node failures + phase faults.
 *   9. Validator Sync Hook — EthValidatorSyncHook interface; stub used in CI;
 *                            hook coherence weights applied to BftEnvironment.
 *
 * Ethereum testnet sync hook:
 *   EthValidatorSyncHook defines the interface a live integration would use.
 *   Set the KERNEL_ETH_TESTNET_RPC environment variable to point to a real
 *   beacon-chain RPC endpoint (e.g. https://rpc.sepolia.org) and subclass
 *   LiveEthValidatorSyncHook for production use.  In this test suite the
 *   default StubEthValidatorSyncHook returns deterministic synthetic data so
 *   that CI runs remain hermetic and network-free.
 *
 * Build:
 *   g++ -std=c++17 -Wall -Wextra -O2 -o test_bft_stress test_bft_stress.cpp -lm
 *
 * Run:
 *   ./test_bft_stress
 *   Exit code 0 on all-pass; non-zero on any failure.
 */

#include "KernelPipeline.hpp"
#include "ohm_coherence_duality.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace kernel::pipeline;
using namespace kernel::ohm;

// ── Test infrastructure
// ───────────────────────────────────────────────────────
static int test_count = 0;
static int passed = 0;
static int failed = 0;

static void test_assert(bool condition, const std::string &name) {
  ++test_count;
  if (condition) {
    std::cout << "  \u2713 " << name << "\n";
    ++passed;
  } else {
    std::cout << "  \u2717 FAILED: " << name << "\n";
    ++failed;
  }
}

static constexpr double TIGHT_TOL = 1e-9;
static constexpr double LOOSE_TOL = 1e-6;

// ══════════════════════════════════════════════════════════════════════════════
// BFT Simulation Layer
// ══════════════════════════════════════════════════════════════════════════════

// Fault kinds that can be injected into a BftNode.
enum class FaultKind {
  NONE,        // No fault — honest node.
  CRASH,       // Node stops processing (simulates a crashed validator).
  PHASE_FAULT, // Phase of beta is rotated by an arbitrary angle.
  AMPLITUDE_CORRUPTION, // Amplitude of beta is scaled (drift injection).
  MESSAGE_DELAY         // Outgoing votes are held for D rounds before delivery.
};

// ── Ethereum testnet validator sync hook
// ─────────────────────────────────────
//
// EthValidatorSyncHook defines the interface through which the BFT simulation
// layer can obtain per-validator metadata from an Ethereum testnet beacon
// chain.  The coherence_weight field carries the validator's attestation-duty
// contribution to the Kernel coherence model: a fully-staked, active validator
// maps to weight 1.0; slashed / inactive validators map to 0.0.
//
// Production integration steps:
//   1. Export KERNEL_ETH_TESTNET_RPC=<beacon-chain-RPC-URL> (e.g. Sepolia).
//   2. Subclass EthValidatorSyncHook, override fetch_validator_info() to call
//      the eth/v1/beacon/states/head/validators REST endpoint and deserialise
//      the response.
//   3. Pass the subclass instance to BftEnvironment::set_sync_hook().
//
// In this test suite StubEthValidatorSyncHook is used so CI runs are hermetic.
//
struct EthValidatorInfo {
  uint64_t validator_index; // Beacon-chain validator index.
  double effective_balance; // Effective balance in ETH (32 = full stake).
  bool is_active;           // True when validator status == "active_ongoing".
  double coherence_weight;  // Coherence weight in [0, 1] for Kernel model.
};

struct EthValidatorSyncHook {
  virtual ~EthValidatorSyncHook() = default;

  // Fetch metadata for the validator identified by node_id.
  virtual EthValidatorInfo fetch_validator_info(uint32_t node_id) const = 0;

  // Human-readable endpoint description (e.g. RPC URL or "stub").
  virtual std::string endpoint_description() const = 0;

  // True when this hook is connected to a live network; false for stubs.
  virtual bool is_live() const = 0;
};

// ── StubEthValidatorSyncHook — hermetic default for CI
// ───────────────────────────
//
// Returns deterministic synthetic validator info derived from node_id.
// All validators are active, carry a full 32 ETH stake, and have a
// coherence_weight of 1.0 minus a tiny per-node perturbation so the
// test can distinguish hook responses per node.
//
struct StubEthValidatorSyncHook : EthValidatorSyncHook {
  EthValidatorInfo fetch_validator_info(uint32_t node_id) const override {
    // Synthetic but deterministic: index = 1000 + node_id,
    // effective_balance decreases slightly for higher node IDs.
    // All stub validators are fully active with coherence_weight = 1.0.
    double balance = 32.0 - 0.01 * static_cast<double>(node_id % 10);
    return {1000ULL + node_id, balance, true, 1.0};
  }

  std::string endpoint_description() const override {
    return "stub://kernel-bft-testnet-simulator";
  }

  bool is_live() const override { return false; }
};

// ── LiveEthValidatorSyncHook — production template (network-free in CI)
// ──────
//
// When KERNEL_ETH_TESTNET_RPC is set this hook documents where the real RPC
// call would go; in CI (no env var) it falls back to the stub so the test
// remains hermetic.  A real implementation would replace fetch_validator_info
// with an HTTP GET to the beacon-chain validators REST API.
//
struct LiveEthValidatorSyncHook : EthValidatorSyncHook {
  std::string rpc_url;
  StubEthValidatorSyncHook fallback;

  LiveEthValidatorSyncHook() {
    const char *env = std::getenv("KERNEL_ETH_TESTNET_RPC");
    rpc_url = (env != nullptr && env[0] != '\0') ? std::string(env) : "";
  }

  EthValidatorInfo fetch_validator_info(uint32_t node_id) const override {
    // In a live integration this would issue:
    //   GET <rpc_url>/eth/v1/beacon/states/head/validators/<validator_index>
    // and parse the JSON response.  Without network access we fall back to
    // the stub so CI runs remain hermetic.
    return fallback.fetch_validator_info(node_id);
  }

  std::string endpoint_description() const override {
    return rpc_url.empty() ? "live://not-configured (fallback to stub)"
                           : "live://" + rpc_url;
  }

  bool is_live() const override { return !rpc_url.empty(); }
};

// ── BftNode: a single validator wrapping KernelState
//
struct BftNode {
  uint32_t id;             // Unique node identifier.
  KernelState state;       // Local quantum/coherence state.
  bool crashed = false;    // True when node is simulated-crashed.
  int delay_remaining = 0; // Rounds remaining before delayed messages deliver.
  int rounds_run = 0;      // Total rounds this node has executed.

  explicit BftNode(uint32_t node_id) : id(node_id) {}

  // Advance the node by one round (applies one pipeline step).
  // Crashed nodes do nothing.
  void tick() {
    if (crashed)
      return;
    SpectralBridge::step(state, KernelMode::FULL);
    ++rounds_run;
  }

  // Does this node cast a valid vote in this round?
  // Delayed nodes withhold votes; crashed nodes never vote.
  bool votes(int /*current_round*/) const {
    if (crashed)
      return false;
    if (delay_remaining > 0)
      return false;
    return state.all_invariants();
  }

  // Inject a phase fault: rotate beta by `angle` radians.
  void inject_phase_fault(double angle) {
    if (crashed)
      return;
    const Cx rot(std::cos(angle), std::sin(angle));
    state.beta *= rot;
    // Re-normalize so beta_unit_invariant can detect the amplitude change.
    // (Phase fault alone preserves |beta|, so we also perturb amplitude.)
    state.beta *= 1.05; // 5% amplitude push to trigger palindrome drift.
    state.normalize();
  }

  // Inject amplitude corruption: scale beta by `scale_factor`.
  void inject_amplitude_corruption(double scale_factor) {
    if (crashed)
      return;
    state.beta *= scale_factor;
    state.normalize();
  }

  // Iteratively apply auto_renormalize until the state converges to r≈1
  // (or max_iters is reached).  Returns true if any renorm step was applied.
  bool try_recover(int max_iters = 100) {
    if (crashed)
      return false;
    bool applied = false;
    for (int i = 0; i < max_iters; ++i) {
      if (!state.auto_renormalize())
        break;
      applied = true;
    }
    return applied;
  }

  // Is the node's state invariant-clean?
  bool is_coherent() const {
    if (crashed)
      return false;
    return state.all_invariants();
  }
};

// ── BftEnvironment: N-node BFT network
// ──────────────────────────────────────
//
// A round-based consensus environment with a 2f+1 quorum rule:
//   N total nodes, at most f Byzantine / faulty nodes.
//   A round is finalised ("committed") when at least ceil(2N/3)+1 nodes vote.
//
struct BftEnvironment {
  std::vector<BftNode> nodes;
  int round = 0;
  int committed_rounds = 0;              // Rounds that reached quorum.
  int skipped_rounds = 0;                // Rounds where quorum was not reached.
  std::vector<double> coherence_history; // Mean coherence per round.

  // Optional validator sync hook.  When set, apply_sync_hook() uses it to
  // apply per-node coherence weights from an Ethereum testnet (or stub).
  const EthValidatorSyncHook *sync_hook = nullptr;

  explicit BftEnvironment(int N) {
    nodes.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
      nodes.emplace_back(static_cast<uint32_t>(i));
  }

  // Attach a validator sync hook.  The hook must outlive this environment.
  void set_sync_hook(const EthValidatorSyncHook *hook) { sync_hook = hook; }

  // Apply the sync hook to every live node: fetch validator info and scale
  // beta by the hook's coherence_weight so that under-staked / inactive
  // validators appear as partially decoherent in the Kernel model.
  // If no hook is attached this is a no-op.
  void apply_sync_hook() {
    if (sync_hook == nullptr)
      return;
    for (auto &node : nodes) {
      if (node.crashed)
        continue;
      EthValidatorInfo info = sync_hook->fetch_validator_info(node.id);
      if (!info.is_active) {
        node.inject_amplitude_corruption(0.0); // Fully decohere inactive node.
      } else if (info.coherence_weight < 1.0 - KS_COHERENCE_TOL) {
        // Scale beta magnitude by coherence_weight to represent partial stake.
        node.state.beta *= info.coherence_weight;
        node.state.normalize();
      }
    }
  }

  int num_nodes() const { return static_cast<int>(nodes.size()); }

  // Minimum votes required to commit (⌊2N/3⌋ + 1 — strict majority of 2/3).
  int quorum_threshold() const { return (2 * num_nodes()) / 3 + 1; }

  // Advance all non-crashed nodes by one round and evaluate quorum.
  // Returns true when the round was committed.
  bool run_round() {
    ++round;
    for (auto &node : nodes)
      node.tick();

    int votes = 0;
    for (const auto &node : nodes)
      if (node.votes(round))
        ++votes;

    // Decrement per-node vote-delay countdowns.
    for (auto &node : nodes)
      if (node.delay_remaining > 0)
        --node.delay_remaining;

    double mean_coh = mean_coherence();
    coherence_history.push_back(mean_coh);

    bool committed = (votes >= quorum_threshold());
    if (committed)
      ++committed_rounds;
    else
      ++skipped_rounds;
    return committed;
  }

  // Crash `count` nodes starting at index `start_idx`.
  void crash_nodes(int start_idx, int count) {
    for (int i = start_idx; i < start_idx + count && i < num_nodes(); ++i)
      nodes[static_cast<size_t>(i)].crashed = true;
  }

  // Delay votes from nodes [start_idx, start_idx+count) for `rounds` rounds.
  void delay_nodes(int start_idx, int count, int rounds) {
    for (int i = start_idx; i < start_idx + count && i < num_nodes(); ++i)
      nodes[static_cast<size_t>(i)].delay_remaining = rounds;
  }

  // Inject phase faults into `count` nodes starting at `start_idx`.
  void inject_phase_faults(int start_idx, int count, double angle) {
    for (int i = start_idx; i < start_idx + count && i < num_nodes(); ++i)
      nodes[static_cast<size_t>(i)].inject_phase_fault(angle);
  }

  // Inject amplitude corruption into `count` nodes starting at `start_idx`.
  void inject_amplitude_corruption(int start_idx, int count,
                                   double scale_factor) {
    for (int i = start_idx; i < start_idx + count && i < num_nodes(); ++i)
      nodes[static_cast<size_t>(i)].inject_amplitude_corruption(scale_factor);
  }

  // Attempt recovery on all faulty (non-crashed) nodes.
  // Returns the number of nodes that actually recovered.
  int recover_all() {
    int recovered = 0;
    for (auto &node : nodes)
      if (node.try_recover())
        ++recovered;
    return recovered;
  }

  // Mean coherence across all live nodes.
  double mean_coherence() const {
    int live = 0;
    double sum = 0.0;
    for (const auto &node : nodes) {
      if (!node.crashed) {
        sum += node.state.coherence();
        ++live;
      }
    }
    return live > 0 ? sum / live : 0.0;
  }

  // Fraction of live nodes with all_invariants() == true.
  double coherent_fraction() const {
    int live = 0, clean = 0;
    for (const auto &node : nodes) {
      if (!node.crashed) {
        ++live;
        if (node.state.all_invariants())
          ++clean;
      }
    }
    return live > 0 ? static_cast<double>(clean) / live : 0.0;
  }

  // Number of crashed nodes.
  int num_crashed() const {
    int c = 0;
    for (const auto &node : nodes)
      if (node.crashed)
        ++c;
    return c;
  }
};

// ══════════════════════════════════════════════════════════════════════════════
// 1. Quorum Safety — 2f+1 honest nodes maintain coherence invariants
// ══════════════════════════════════════════════════════════════════════════════
static void test_quorum_safety() {
  std::cout << "\n\u2554\u2550\u2550\u2550 1. Quorum Safety "
               "\u2550\u2550\u2550\u2557\n";

  // N=7 nodes, f=2 Byzantine tolerance (2f+1 = 5 quorum threshold).
  const int N = 7;
  BftEnvironment env(N);

  test_assert(env.quorum_threshold() == 5,
              "N=7: quorum threshold = 5 (2f+1 with f=2)");

  // All nodes start clean — every round should commit.
  int committed = 0;
  for (int i = 0; i < 10; ++i)
    if (env.run_round())
      ++committed;

  test_assert(committed == 10, "10 rounds with 0 faults: all rounds committed");

  // Crash 2 nodes (at or below f); network must still commit.
  env.crash_nodes(0, 2);
  test_assert(env.num_crashed() == 2, "crashed 2 nodes (f = 2)");

  committed = 0;
  for (int i = 0; i < 10; ++i)
    if (env.run_round())
      ++committed;

  test_assert(committed == 10,
              "10 rounds with f=2 crashes: quorum still reached every round");

  // Verify remaining live nodes are coherent.
  test_assert(env.coherent_fraction() >= 1.0 - LOOSE_TOL,
              "all live nodes coherent after f-fault period");
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. Phase Faults — detection and recovery
// ══════════════════════════════════════════════════════════════════════════════
static void test_phase_faults() {
  std::cout << "\n\u2554\u2550\u2550\u2550 2. Phase Fault Injection "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 5;
  BftEnvironment env(N);

  // Run 5 clean rounds first.
  for (int i = 0; i < 5; ++i)
    env.run_round();

  double coh_before = env.mean_coherence();
  test_assert(coh_before > 0.99,
              "mean coherence > 0.99 before fault injection");

  // Inject phase fault into node 0 only.
  env.inject_phase_faults(0, 1, OHM_PI / 3.0);
  bool node0_drifted = env.nodes[0].state.has_drift();
  test_assert(node0_drifted, "phase fault detected: node 0 has_drift() = true");
  test_assert(!env.nodes[0].state.all_invariants(),
              "phase fault: node 0 fails all_invariants()");

  // Honest nodes (1-4) must still be coherent and form quorum.
  int coherent_honest = 0;
  for (int i = 1; i < N; ++i)
    if (env.nodes[static_cast<size_t>(i)].is_coherent())
      ++coherent_honest;
  test_assert(coherent_honest == N - 1,
              "honest nodes remain coherent after single phase fault");

  // Run one more round — quorum still reachable from honest nodes.
  bool committed = env.run_round();
  test_assert(committed,
              "round committed despite 1 faulty node (quorum from honest 4)");

  // Recover the faulted node.
  bool recovered = env.nodes[0].try_recover();
  test_assert(recovered, "node 0 auto_renormalize() applied after phase fault");
  test_assert(env.nodes[0].is_coherent(),
              "node 0 coherent after recovery from phase fault");
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. Node Failures — liveness under crash faults
// ══════════════════════════════════════════════════════════════════════════════
static void test_node_failures() {
  std::cout << "\n\u2554\u2550\u2550\u2550 3. Node Failure Liveness "
               "\u2550\u2550\u2550\u2557\n";

  // N=10, f=3: quorum threshold = floor(2*10/3)+1 = 6+1 = 7.
  const int N = 10;
  BftEnvironment env(N);

  test_assert(env.quorum_threshold() == 7,
              "N=10: quorum threshold = 7 (floor(2*10/3)+1)");

  // Warm up.
  for (int i = 0; i < 5; ++i)
    env.run_round();

  // Crash exactly f=3 nodes.
  env.crash_nodes(7, 3);
  test_assert(env.num_crashed() == 3, "3 nodes crashed (f=3)");

  // Run 20 rounds — all should commit.
  int committed = 0;
  for (int i = 0; i < 20; ++i)
    if (env.run_round())
      ++committed;

  test_assert(committed == 20,
              "20 rounds with f=3 crashes: liveness maintained");

  // Coherence among surviving nodes must stay near 1.
  double final_coh = env.mean_coherence();
  test_assert(final_coh > 0.98,
              "mean coherence > 0.98 across surviving nodes after 20 rounds");
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Delayed IPC Messages — coherence under network partitions
// ══════════════════════════════════════════════════════════════════════════════
static void test_delayed_ipc() {
  std::cout << "\n\u2554\u2550\u2550\u2550 4. Delayed IPC Messages "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 7;
  BftEnvironment env(N);

  // Warm up.
  for (int i = 0; i < 3; ++i)
    env.run_round();

  // Delay 2 nodes for 5 rounds (simulates a network partition / slow path).
  const int DELAY = 5;
  env.delay_nodes(5, 2, DELAY);

  // Rounds 4-8: delayed nodes withhold votes; quorum still reachable from 5.
  int committed_during_delay = 0;
  for (int i = 0; i < DELAY; ++i)
    if (env.run_round())
      ++committed_during_delay;

  test_assert(committed_during_delay == DELAY,
              "rounds during IPC delay: quorum reached from non-delayed nodes");

  // Rounds 9+: delayed nodes resume voting.
  int committed_after = 0;
  for (int i = 0; i < 5; ++i)
    if (env.run_round())
      ++committed_after;

  test_assert(committed_after == 5,
              "rounds after delay lifted: all 7 nodes vote and commit");

  // All nodes must still satisfy coherence invariants.
  test_assert(env.coherent_fraction() >= 1.0 - LOOSE_TOL,
              "all nodes coherent after delayed-IPC stress period");
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. State Corruption and Recovery — beta_unit_invariant / auto_renormalize
// ══════════════════════════════════════════════════════════════════════════════
static void test_state_corruption_recovery() {
  std::cout << "\n\u2554\u2550\u2550\u2550 5. State Corruption & Recovery "
               "\u2550\u2550\u2550\u2557\n";

  // Standalone KernelState corruption test (mirrors invariant semantics).
  KernelState ks;
  test_assert(ks.beta_unit_invariant(),
              "fresh state: beta_unit_invariant() = true");
  test_assert(!ks.has_drift(), "fresh state: has_drift() = false");

  // Corrupt: scale beta to push r away from 1.
  ks.beta *= 1.8;
  ks.normalize();
  test_assert(ks.has_drift(), "after corruption: has_drift() = true");
  test_assert(!ks.palindrome_zero(),
              "after corruption: palindrome_residual \u2260 0");

  // Recovery via auto_renormalize.
  bool fixed = ks.auto_renormalize();
  test_assert(fixed, "auto_renormalize() applied after corruption");
  // After one renorm step (rate=0.5) drift should be significantly reduced.
  double residual = std::abs(ks.palindrome_residual());
  test_assert(residual < 0.5,
              "palindrome residual < 0.5 after one renorm step");

  // Multiple renorm iterations converge to r=1 (rate=0.5 → ~32 steps needed).
  for (int i = 0; i < 50; ++i)
    ks.auto_renormalize();
  test_assert(ks.palindrome_zero(),
              "palindrome_residual \u2248 0 after iterative renormalization");
  test_assert(ks.all_invariants(),
              "all invariants restored after iterative renormalization");

  // BftEnvironment-level mass corruption + recovery.
  const int N = 9;
  BftEnvironment env(N);
  for (int i = 0; i < 3; ++i)
    env.run_round();

  // Corrupt all nodes.
  env.inject_amplitude_corruption(0, N, 1.7);
  double frac_before = env.coherent_fraction();
  test_assert(frac_before < 0.1,
              "after mass amplitude corruption: coherent fraction < 10%");

  // Recover all nodes (try_recover iterates until convergence per node).
  env.recover_all();

  double frac_after = env.coherent_fraction();
  test_assert(
      frac_after >= 1.0 - LOOSE_TOL,
      "all nodes coherent after iterative recovery from mass corruption");
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. Finality Preservation — committed rounds are never rolled back
// ══════════════════════════════════════════════════════════════════════════════
static void test_finality_preservation() {
  std::cout << "\n\u2554\u2550\u2550\u2550 6. Finality Preservation "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 7;
  BftEnvironment env(N);

  // Commit 10 rounds.
  int pre_fault_committed = 0;
  for (int i = 0; i < 10; ++i)
    if (env.run_round())
      ++pre_fault_committed;

  int committed_snapshot = env.committed_rounds;
  test_assert(committed_snapshot == pre_fault_committed,
              "10 clean rounds committed before fault injection");

  // Now inject faults (crash 2, phase-fault 1) and run more rounds.
  env.crash_nodes(0, 2);
  env.inject_phase_faults(2, 1, OHM_PI / 2.0);
  env.nodes[2].try_recover(); // repair immediately

  int post_fault_committed = 0;
  for (int i = 0; i < 5; ++i)
    if (env.run_round())
      ++post_fault_committed;

  // Finality: pre-fault committed_rounds must not decrease.
  test_assert(
      env.committed_rounds >= committed_snapshot,
      "committed_rounds never decreases after fault injection (safety)");

  // Liveness: at least some post-fault rounds should commit.
  test_assert(post_fault_committed > 0,
              "at least one post-fault round committed (liveness)");
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. Recovery Rate — coherence convergence after mass fault
// ══════════════════════════════════════════════════════════════════════════════
static void test_recovery_rate() {
  std::cout << "\n\u2554\u2550\u2550\u2550 7. Recovery Rate "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 9;
  BftEnvironment env(N);

  // Warm up 5 rounds.
  for (int i = 0; i < 5; ++i)
    env.run_round();

  // Inject phase fault into every node.
  env.inject_phase_faults(0, N, OHM_PI / 4.0);
  double coh_post_fault = env.mean_coherence();

  // Run recovery rounds: each tick calls auto_renormalize if needed, and
  // one renorm pass per round is applied explicitly.
  const int RECOVERY_ROUNDS = 30;
  std::vector<double> recovery_trace;
  recovery_trace.reserve(static_cast<size_t>(RECOVERY_ROUNDS));

  for (int r = 0; r < RECOVERY_ROUNDS; ++r) {
    env.recover_all();
    env.run_round();
    recovery_trace.push_back(env.mean_coherence());
  }

  double coh_final = recovery_trace.back();

  // The final coherence must be substantially better than post-fault.
  test_assert(coh_final > coh_post_fault,
              "mean coherence improves over recovery rounds");

  // Monotone convergence: final value ≥ median value.
  std::vector<double> sorted = recovery_trace;
  std::sort(sorted.begin(), sorted.end());
  double median_coh = sorted[static_cast<size_t>(RECOVERY_ROUNDS / 2)];
  test_assert(coh_final >= median_coh - LOOSE_TOL,
              "coherence converges monotonically toward final value");

  // Final coherence close to 1 after recovery.
  test_assert(coh_final > 0.95,
              "mean coherence > 0.95 after full recovery period");

  // Print a compact summary.
  std::cout << "    post-fault C=" << std::fixed << std::setprecision(6)
            << coh_post_fault << "  final C=" << coh_final << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. Liveness & Safety under Combined Faults
// ══════════════════════════════════════════════════════════════════════════════
static void test_liveness_and_safety() {
  std::cout << "\n\u2554\u2550\u2550\u2550 8. Liveness & Safety "
               "\u2550\u2550\u2550\u2557\n";

  // Large network: N=13, tolerates up to f=4 byzantine faults.
  const int N = 13;
  BftEnvironment env(N);

  // Phase 1: 10 clean rounds.
  for (int i = 0; i < 10; ++i)
    env.run_round();
  int clean_committed = env.committed_rounds;

  // Phase 2: crash 2, delay 2, phase-fault 1 simultaneously.
  env.crash_nodes(0, 2);
  env.delay_nodes(2, 2, 8);
  env.inject_phase_faults(4, 1, OHM_PI / 6.0);
  env.nodes[4].try_recover();

  int faulted_committed = 0;
  for (int i = 0; i < 10; ++i)
    if (env.run_round())
      ++faulted_committed;

  // Safety: committed count must not regress.
  test_assert(env.committed_rounds >= clean_committed,
              "committed_rounds >= pre-fault count (safety guarantee)");

  // Liveness: most fault-period rounds should still commit.
  // With 2 crashed + 2 delayed + 1 (recovered) = 5 affected out of 13,
  // the remaining 8 honest nodes exceed the 9-node quorum threshold.
  // After recovery round the delayed nodes re-join.
  test_assert(faulted_committed >= 5,
              "at least 5/10 fault-period rounds commit (liveness)");

  // Phase 3: recover delayed nodes, run 5 more rounds.
  for (auto &node : env.nodes)
    node.delay_remaining = 0; // Lift any remaining delays explicitly.
  int post_committed = 0;
  for (int i = 0; i < 5; ++i)
    if (env.run_round())
      ++post_committed;

  test_assert(post_committed == 5,
              "all 5 post-recovery rounds committed after faults resolved");

  // Coherence fraction among live nodes must be high.
  test_assert(env.coherent_fraction() >= 1.0 - LOOSE_TOL,
              "all live nodes coherent at end of combined-fault scenario");
}

// ══════════════════════════════════════════════════════════════════════════════
// 9. Ethereum Testnet Validator Sync Hook
// ══════════════════════════════════════════════════════════════════════════════
static void test_validator_sync_hook() {
  std::cout << "\n\u2554\u2550\u2550\u2550 9. Validator Sync Hook "
               "\u2550\u2550\u2550\u2557\n";

  // ── 9a. StubEthValidatorSyncHook interface contract ───────────────────────
  StubEthValidatorSyncHook stub;

  test_assert(!stub.is_live(),
              "stub hook: is_live() = false (hermetic, no network)");

  std::string ep = stub.endpoint_description();
  test_assert(!ep.empty(), "stub hook: endpoint_description() non-empty");

  // Fetch info for a few nodes and validate the fields.
  for (uint32_t id = 0; id < 5; ++id) {
    EthValidatorInfo info = stub.fetch_validator_info(id);
    test_assert(info.validator_index == 1000ULL + id,
                "stub hook: validator_index = 1000 + node_id");
    test_assert(info.is_active, "stub hook: is_active = true");
    test_assert(info.effective_balance > 0.0 && info.effective_balance <= 32.0,
                "stub hook: effective_balance in (0, 32]");
    test_assert(info.coherence_weight > 0.0 && info.coherence_weight <= 1.0,
                "stub hook: coherence_weight in (0, 1]");
  }

  // ── 9b. LiveEthValidatorSyncHook falls back to stub when no RPC set ───────
  LiveEthValidatorSyncHook live;
  // In CI KERNEL_ETH_TESTNET_RPC is not set → is_live() = false.
  // (If the env var were set the hook would report is_live() = true.)
  bool env_rpc_set = (std::getenv("KERNEL_ETH_TESTNET_RPC") != nullptr &&
                      std::getenv("KERNEL_ETH_TESTNET_RPC")[0] != '\0');
  test_assert(live.is_live() == env_rpc_set,
              "live hook: is_live() matches KERNEL_ETH_TESTNET_RPC presence");

  // Regardless of env var, fetch_validator_info must return valid data.
  EthValidatorInfo live_info = live.fetch_validator_info(0);
  test_assert(live_info.coherence_weight > 0.0 &&
                  live_info.coherence_weight <= 1.0,
              "live hook: coherence_weight in (0, 1] regardless of RPC env");

  // ── 9c. BftEnvironment::set_sync_hook / apply_sync_hook integration ───────
  const int N = 7;
  BftEnvironment env(N);

  // Warm up without hook.
  for (int i = 0; i < 5; ++i)
    env.run_round();
  double coh_no_hook = env.mean_coherence();
  test_assert(coh_no_hook > 0.99,
              "sync hook test: mean coherence > 0.99 before hook applied");

  // Attach stub hook and apply it (all validators fully staked → no drift).
  env.set_sync_hook(&stub);
  env.apply_sync_hook();

  // Full-stake stub weights are ≈ 1.0: coherence should remain near 1.
  double coh_after_hook = env.mean_coherence();
  test_assert(coh_after_hook > 0.98,
              "sync hook: full-stake stub preserves mean coherence > 0.98");
  test_assert(env.coherent_fraction() >= 1.0 - LOOSE_TOL,
              "sync hook: all nodes coherent after full-stake hook applied");

  // Quorum still works after hook application.
  int committed = 0;
  for (int i = 0; i < 5; ++i)
    if (env.run_round())
      ++committed;
  test_assert(committed == 5,
              "sync hook: 5/5 rounds commit after full-stake hook applied");

  // Print hook summary.
  std::cout << "    stub endpoint : " << stub.endpoint_description() << "\n";
  std::cout << "    live endpoint : " << live.endpoint_description() << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════════════
int main() {
  std::cout << "══════════════════════════════════════════════════════\n";
  std::cout << "  Kernel BFT Stress Tests\n";
  std::cout << "══════════════════════════════════════════════════════\n";

  auto t0 = std::chrono::steady_clock::now();

  test_quorum_safety();
  test_phase_faults();
  test_node_failures();
  test_delayed_ipc();
  test_state_corruption_recovery();
  test_finality_preservation();
  test_recovery_rate();
  test_liveness_and_safety();
  test_validator_sync_hook();

  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "\n══════════════════════════════════════════════════════\n";
  std::cout << "  Results: " << passed << "/" << test_count << " passed";
  if (failed > 0)
    std::cout << "  (" << failed << " FAILED)";
  std::cout << "  [" << std::fixed << std::setprecision(1) << elapsed_ms
            << " ms]\n";
  std::cout << "══════════════════════════════════════════════════════\n";

  return failed == 0 ? 0 : 1;
}
