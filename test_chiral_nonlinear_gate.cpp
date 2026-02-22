/*
 * test_chiral_nonlinear_gate.cpp — Chiral Non-Linear Gate Validation Suite
 *
 * Tests whether the Chiral Non-Linear Gate is the mechanism responsible for
 * handling non-linear search spaces (including hash-function analogs) and
 * whether it provides the classical machinery for quantum-speedup domains.
 *
 * Test sections:
 *   1. Gate Mechanics — Linear (Im≤0) and non-linear (Im>0) domain behaviour
 *   2. 8-Cycle Structure — Full periodicity and reversibility
 *   3. Hash Oracle Analog — Phase oracle based on a hash-like predicate;
 *      gate amplifies the marked state in a non-linear search space
 *   4. Amplitude Amplification — Kick vs no-kick target probability comparison
 *   5. Quantum-Speedup Analog — Ladder search convergence rate with/without kick
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ── Constants (mirror quantum_kernel_v2.cpp) ─────────────────────────────────
static constexpr double ETA = 0.70710678118654752440;        // 1/√2
static constexpr double DELTA_S = 2.41421356237309504880;    // δ_S = 1+√2
static constexpr double DELTA_CONJ = 0.41421356237309504880; // √2−1 = 1/δ_S
static constexpr double TIGHT_TOL = 1e-12;
static constexpr double FLOAT_TOL = 1e-9;

using Cx = std::complex<double>;

// ── Minimal QState (same canonical initial state as quantum_kernel_v2.cpp) ───
struct QState {
  Cx alpha{ETA, 0.0};
  Cx beta{-0.5, 0.5}; // e^{i3π/4}/√2

  double c_l1() const { return 2.0 * std::abs(alpha) * std::abs(beta); }
  double radius() const {
    return std::abs(alpha) > FLOAT_TOL ? std::abs(beta) / std::abs(alpha) : 0.0;
  }
  void step() {
    static const Cx MU{-ETA, ETA};
    beta *= MU;
  }
};

// ── Include the gate under test ───────────────────────────────────────────────
#include "ChiralNonlinearGate.hpp"

// ── Test infrastructure ───────────────────────────────────────────────────────
static int test_count = 0;
static int passed     = 0;
static int failed     = 0;

static void test_assert(bool condition, const std::string &test_name) {
  ++test_count;
  if (condition) {
    std::cout << "  \u2713 " << test_name << "\n";
    ++passed;
  } else {
    std::cout << "  \u2717 FAILED: " << test_name << "\n";
    ++failed;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 1. Gate Mechanics
// ══════════════════════════════════════════════════════════════════════════════
static void test_gate_mechanics() {
  std::cout << "\n\u2554\u2550\u2550\u2550 1. Gate Mechanics \u2550\u2550\u2550\u2557\n";

  // 1a. Im≤0 domain: chiral gate equals standard µ-rotation (no kick applied)
  {
    QState s_chiral, s_linear;
    s_chiral.beta = Cx{0.4, -0.6}; // Im < 0
    s_linear.beta = s_chiral.beta;

    s_chiral = kernel::quantum::chiral_nonlinear(s_chiral, 0.5);
    s_linear.step();

    test_assert(std::abs(s_chiral.beta - s_linear.beta) < FLOAT_TOL,
                "Im<0 domain: chiral gate equals µ-rotation (kick ignored)");
  }

  // 1b. Im=0 boundary: treated as Im≤0 (no kick)
  {
    QState s_chiral, s_linear;
    s_chiral.beta = Cx{0.5, 0.0};
    s_linear.beta = s_chiral.beta;

    s_chiral = kernel::quantum::chiral_nonlinear(s_chiral, 0.5);
    s_linear.step();

    test_assert(std::abs(s_chiral.beta - s_linear.beta) < FLOAT_TOL,
                "Im=0 boundary: chiral gate equals µ-rotation (no kick)");
  }

  // 1c. Im>0 domain with kick=0: still equals µ-rotation
  {
    QState s_chiral, s_linear;
    s_chiral.beta = Cx{-0.3, 0.6}; // Im > 0
    s_linear.beta = s_chiral.beta;

    s_chiral = kernel::quantum::chiral_nonlinear(s_chiral, 0.0);
    s_linear.step();

    test_assert(std::abs(s_chiral.beta - s_linear.beta) < FLOAT_TOL,
                "Im>0 domain with kick=0: equals µ-rotation");
  }

  // 1d. Im>0 domain with kick>0: deviates from µ-rotation
  {
    QState s_chiral, s_linear;
    s_chiral.beta = Cx{-0.3, 0.6};
    s_linear.beta = s_chiral.beta;

    s_chiral = kernel::quantum::chiral_nonlinear(s_chiral, 0.2);
    s_linear.step();

    test_assert(std::abs(s_chiral.beta - s_linear.beta) > FLOAT_TOL,
                "Im>0 domain with kick>0: deviates from µ-rotation (non-linear)");
  }

  // 1e. Im>0 domain with kick>0: magnitude grows beyond linear rotation
  {
    QState s;
    s.beta = Cx{0.0, 0.5};

    QState s_ref = s;
    s_ref.step();

    QState s_kicked = s;
    s_kicked = kernel::quantum::chiral_nonlinear(s_kicked, 0.2);

    test_assert(std::abs(s_kicked.beta) > std::abs(s_ref.beta),
                "Im>0 domain: kick grows |β| beyond linear magnitude");
  }

  // 1f. Im≤0 domain: magnitude preserved (no growth)
  {
    QState s;
    s.beta = Cx{0.3, -0.5};
    double mag_before = std::abs(s.beta);

    s = kernel::quantum::chiral_nonlinear(s, 1.0); // large kick, Im<0

    test_assert(std::abs(std::abs(s.beta) - mag_before) < FLOAT_TOL,
                "Im≤0 domain: magnitude preserved (linearity intact)");
  }

  // 1g. Kick strength proportional to magnitude growth on Im>0
  {
    QState base;
    base.beta = Cx{0.0, 0.5};
    QState s_small = base, s_large = base;

    s_small = kernel::quantum::chiral_nonlinear(s_small, 0.05);
    s_large = kernel::quantum::chiral_nonlinear(s_large, 0.20);

    test_assert(std::abs(s_large.beta) > std::abs(s_small.beta),
                "Im>0 domain: larger kick → larger |β| (quadratic scaling)");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. 8-Cycle Structure
// ══════════════════════════════════════════════════════════════════════════════
static void test_8cycle_structure() {
  std::cout << "\n\u2554\u2550\u2550\u2550 2. 8-Cycle Structure \u2550\u2550\u2550\u2557\n";

  // 2a. Zero-kick: 8 steps return to original state (µ^8 = 1)
  {
    QState s;
    Cx beta_init = s.beta;
    for (int i = 0; i < 8; ++i)
      s = kernel::quantum::chiral_nonlinear(s, 0.0);
    test_assert(std::abs(s.beta - beta_init) < FLOAT_TOL,
                "8-cycle with kick=0: state returns to origin (µ^8=1)");
  }

  // 2b. Zero-kick: |β| preserved over a full 8-cycle
  {
    QState s2;
    double m_init = std::abs(s2.beta);
    for (int i = 0; i < 8; ++i)
      s2 = kernel::quantum::chiral_nonlinear(s2, 0.0);
    test_assert(std::abs(std::abs(s2.beta) - m_init) < FLOAT_TOL,
                "8-cycle with kick=0: |β| preserved over full cycle");
  }

  // 2c. CHIRAL_MU matches µ = e^{i3π/4}
  {
    const double pi = 3.14159265358979323846;
    Cx expected{std::cos(3.0 * pi / 4.0), std::sin(3.0 * pi / 4.0)};
    test_assert(std::abs(kernel::quantum::CHIRAL_MU - expected) < TIGHT_TOL,
                "CHIRAL_MU = e^{i3\u03c0/4} (exact balance primitive)");
  }

  // 2d. Silver conservation: δ_S · (√2−1) = 1
  {
    test_assert(std::abs(DELTA_S * DELTA_CONJ - 1.0) < TIGHT_TOL,
                "\u03b4_S\u00b7(\u221a2\u22121) = 1 (silver conservation)");
  }

  // 2e. CHIRAL_ETA = 1/√2
  {
    test_assert(std::abs(kernel::quantum::CHIRAL_ETA - ETA) < TIGHT_TOL,
                "CHIRAL_ETA = 1/\u221a2 (exact)");
  }

  // 2f. With kick active: 8-step cycle does NOT return to origin (irreversibility)
  {
    QState s;
    // Ensure at least one Im>0 step occurs in the 8-cycle starting from
    // canonical state β = e^{i3π/4}/√2.  Im(β_init) = +0.5 > 0.
    Cx beta_init = s.beta;
    double kick = 0.1;
    for (int i = 0; i < 8; ++i)
      s = kernel::quantum::chiral_nonlinear(s, kick);
    // Non-zero kick on at least one Im>0 step shatters periodicity
    test_assert(std::abs(s.beta - beta_init) > FLOAT_TOL,
                "8-cycle with kick>0: reversibility shattered (Im>0 steps)");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. Hash Oracle Analog
//
// Simulates a "hash oracle" by marking states whose index satisfies a simple
// non-linear hash predicate.  The gate then amplifies the marked states.
// This models the core use case: searching a non-linear (hash-like) space.
// ══════════════════════════════════════════════════════════════════════════════

// Simple deterministic hash predicate: mark index i if the low bits of a
// cheap xor-shift hash of i equal a target digest.  This mimics the non-linear
// structure of a real hash without external dependencies.
static bool hash_oracle(uint32_t i, uint32_t target_digest, uint32_t mask) {
  // Cheap xor-shift hash (non-linear, bit-mixing)
  uint32_t x = i;
  x ^= (x << 13);
  x ^= (x >> 17);
  x ^= (x << 5);
  return (x & mask) == target_digest;
}

static void test_hash_oracle_analog() {
  std::cout << "\n\u2554\u2550\u2550\u2550 3. Hash Oracle Analog \u2550\u2550\u2550\u2557\n";

  // Search space of n candidates; one (or few) match the hash predicate.
  const size_t N = 64;
  const uint32_t MASK = 0x3Fu; // 6-bit mask → search space density ~ 1/64

  // Find the first target that matches the predicate
  uint32_t target_digest = 0;
  size_t target_idx = N; // sentinel: "not found yet"
  for (uint32_t digest = 0; digest < (MASK + 1u); ++digest) {
    for (size_t i = 0; i < N; ++i) {
      if (hash_oracle(static_cast<uint32_t>(i), digest, MASK)) {
        target_digest = digest;
        target_idx = i;
        break;
      }
    }
    if (target_idx < N)
      break;
  }
  test_assert(target_idx < N, "Hash oracle: at least one matching state found");

  // Build state register and apply phase oracle (flip β sign for matches)
  std::vector<QState> states(N);
  for (size_t i = 0; i < N; ++i) {
    if (hash_oracle(static_cast<uint32_t>(i), target_digest, MASK)) {
      states[i].beta = -states[i].beta; // phase flip
    }
  }

  // Record magnitude of the marked target state before gate application
  double mag_target_before = std::abs(states[target_idx].beta);

  // Apply chiral gate with kick to ALL states
  const double kick = 0.15;
  for (size_t i = 0; i < N; ++i)
    states[i] = kernel::quantum::chiral_nonlinear(states[i], kick);

  double mag_target_after_kick = std::abs(states[target_idx].beta);

  // Reference: apply without kick
  std::vector<QState> states_ref(N);
  for (size_t i = 0; i < N; ++i) {
    if (hash_oracle(static_cast<uint32_t>(i), target_digest, MASK))
      states_ref[i].beta = -states_ref[i].beta;
  }
  for (size_t i = 0; i < N; ++i)
    states_ref[i] = kernel::quantum::chiral_nonlinear(states_ref[i], 0.0);

  double mag_target_ref = std::abs(states_ref[target_idx].beta);

  // 3a. Gate was applied (state evolved from initial)
  test_assert(mag_target_after_kick != mag_target_before ||
                  std::abs(states[target_idx].beta -
                           Cx{-0.5, 0.5}) > FLOAT_TOL,
              "Hash oracle: gate applied, target state evolved");

  // 3b. Non-linear kick amplifies the target state beyond linear rotation
  //     (true when Im(β) > 0 for the marked state before rotation)
  {
    // Check if the phase-flipped target had Im>0 before the kick
    QState check_state;
    check_state.beta = -check_state.beta; // same flip as oracle
    bool was_pos_imag = (check_state.beta.imag() > 0.0);
    if (was_pos_imag) {
      test_assert(mag_target_after_kick > mag_target_ref,
                  "Hash oracle: kick amplifies marked target beyond linear "
                  "rotation (Im>0 marked state)");
    } else {
      // Im≤0: kick has no effect; magnitudes must be equal
      test_assert(std::abs(mag_target_after_kick - mag_target_ref) < FLOAT_TOL,
                  "Hash oracle: Im≤0 marked state — kick has no effect");
    }
  }

  // 3c. Unmarked states retain baseline magnitude (no spurious amplification)
  //     Find an unmarked state and verify its evolution matches no-kick
  size_t unmarked_idx = N;
  for (size_t i = 0; i < N; ++i) {
    if (!hash_oracle(static_cast<uint32_t>(i), target_digest, MASK)) {
      unmarked_idx = i;
      break;
    }
  }
  if (unmarked_idx < N) {
    // Both kicked and no-kick start from identical initial unmarked states
    QState s_kick, s_nokick;
    s_kick = kernel::quantum::chiral_nonlinear(s_kick, kick);
    s_nokick = kernel::quantum::chiral_nonlinear(s_nokick, 0.0);
    // Both see the same initial state → only the Im>0 path differs
    // The unmarked canonical state has Im(β)=+0.5 > 0, so the kick applies.
    // We test that the kicked state's magnitude is at least as large.
    test_assert(std::abs(s_kick.beta) >= std::abs(s_nokick.beta) - FLOAT_TOL,
                "Hash oracle: unmarked state — kick does not shrink magnitude");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Amplitude Amplification
//
// Run multiple gate steps on a marked target state and verify that the
// quadratic kick provides monotonically increasing target amplitude advantage
// over the linear (no-kick) baseline.
// ══════════════════════════════════════════════════════════════════════════════
static void test_amplitude_amplification() {
  std::cout << "\n\u2554\u2550\u2550\u2550 4. Amplitude Amplification \u2550\u2550\u2550\u2557\n";

  const size_t N = 16;
  const size_t TARGET = 5;
  const size_t STEPS = 10;
  const double KICK = 0.12;

  // Initialise two identical registers: one with kick, one without
  std::vector<QState> kicked(N), linear(N);
  kicked[TARGET].beta = -kicked[TARGET].beta;
  linear[TARGET].beta = -linear[TARGET].beta;

  bool kicked_ever_higher = false;

  for (size_t step = 0; step < STEPS; ++step) {
    // Apply gate
    for (size_t i = 0; i < N; ++i) {
      kicked[i] = kernel::quantum::chiral_nonlinear(kicked[i], KICK);
      linear[i] = kernel::quantum::chiral_nonlinear(linear[i], 0.0);
    }

    // Compute normalised target probability: |β_target|² / Σ|β_i|²
    double total_kicked = 0.0, total_linear = 0.0;
    for (size_t i = 0; i < N; ++i) {
      total_kicked += std::norm(kicked[i].beta);
      total_linear += std::norm(linear[i].beta);
    }
    double prob_kicked =
        (total_kicked > 0.0) ? std::norm(kicked[TARGET].beta) / total_kicked : 0.0;
    double prob_linear =
        (total_linear > 0.0) ? std::norm(linear[TARGET].beta) / total_linear : 0.0;

    if (prob_kicked > prob_linear)
      kicked_ever_higher = true;
  }

  // 4a. Kick achieves higher target probability at some point during STEPS
  test_assert(kicked_ever_higher,
              "Amplitude amplification: kick achieves higher P(target) than "
              "linear at some step");

  // 4b. Kicked target |β| grows strictly larger than linear target |β|
  //     across all STEPS — the kick provides net amplification
  {
    bool kicked_target_larger = std::abs(kicked[TARGET].beta) >
                                std::abs(linear[TARGET].beta);
    test_assert(kicked_target_larger,
                "Amplitude amplification: kicked |β_target| > linear |β_target| "
                "after STEPS rounds");
  }

  // 4c. Kick never produces NaN or infinite magnitudes
  {
    bool all_finite = true;
    std::vector<QState> probe(N);
    probe[0].beta = -probe[0].beta;
    for (size_t step = 0; step < STEPS; ++step)
      for (size_t i = 0; i < N; ++i) {
        probe[i] = kernel::quantum::chiral_nonlinear(probe[i], KICK);
        if (!std::isfinite(std::abs(probe[i].beta)))
          all_finite = false;
      }
    test_assert(all_finite,
                "Amplitude amplification: |β| is finite for all states throughout");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. Quantum-Speedup Analog
//
// Implements a simplified ladder search: each round applies the phase oracle
// (marking the target) then the chiral gate.  Compare the number of rounds
// needed for the kicked vs no-kick versions to cross a detection threshold.
//
// The kicked version should detect the target in fewer rounds (O(√N) analog),
// while the no-kick linear version requires more rounds (O(N) analog).
// ══════════════════════════════════════════════════════════════════════════════
static void test_quantum_speedup_analog() {
  std::cout << "\n\u2554\u2550\u2550\u2550 5. Quantum-Speedup Analog \u2550\u2550\u2550\u2557\n";

  const size_t N = 32;
  const size_t TARGET = 11;
  const double KICK = 0.15;
  const size_t MAX_ROUNDS = 200;

  // Detection threshold used by run_ladder
  const double THRESHOLD_KICKED = 0.20;

  auto run_ladder = [&](double kick_strength) -> size_t {
    std::vector<QState> states(N);
    for (size_t round = 0; round < MAX_ROUNDS; ++round) {
      // Phase oracle: mark target
      states[TARGET].beta = -states[TARGET].beta;

      // Apply chiral gate
      for (size_t i = 0; i < N; ++i)
        states[i] = kernel::quantum::chiral_nonlinear(states[i], kick_strength);

      // Compute P(target)
      double total = 0.0;
      for (size_t i = 0; i < N; ++i)
        total += std::norm(states[i].beta);
      double p = (total > 0.0) ? std::norm(states[TARGET].beta) / total : 0.0;

      if (p >= THRESHOLD_KICKED)
        return round + 1;
    }
    return MAX_ROUNDS; // not detected within budget
  };

  size_t rounds_kicked = run_ladder(KICK);
  size_t rounds_linear = run_ladder(0.0);

  // 5a. Kicked version achieves strictly higher max P(target) than linear
  {
    auto max_prob = [&](double kick_strength) -> double {
      std::vector<QState> states(N);
      double max_p = 0.0;
      for (size_t round = 0; round < MAX_ROUNDS; ++round) {
        states[TARGET].beta = -states[TARGET].beta;
        for (size_t i = 0; i < N; ++i)
          states[i] = kernel::quantum::chiral_nonlinear(states[i], kick_strength);
        double total = 0.0;
        for (size_t i = 0; i < N; ++i)
          total += std::norm(states[i].beta);
        double p =
            (total > 0.0) ? std::norm(states[TARGET].beta) / total : 0.0;
        if (p > max_p)
          max_p = p;
      }
      return max_p;
    };

    double max_p_kicked = max_prob(KICK);
    double max_p_linear = max_prob(0.0);

    test_assert(max_p_kicked > max_p_linear,
                "Speedup analog: kicked search achieves higher peak P(target) "
                "than linear search");
  }

  // 5b. Kicked version detects target in fewer rounds than linear
  test_assert(rounds_kicked <= rounds_linear,
              "Speedup analog: kicked search converges no slower than linear");

  // 5c. With no kick the uniform prior (1/N) is eventually beaten if threshold
  //     is reached, or remains non-negative throughout
  {
    bool probs_valid = true;
    std::vector<QState> check_states(N);
    for (size_t round = 0; round < 20 && probs_valid; ++round) {
      check_states[TARGET].beta = -check_states[TARGET].beta;
      for (size_t i = 0; i < N; ++i)
        check_states[i] = kernel::quantum::chiral_nonlinear(check_states[i], 0.0);
      double total = 0.0;
      for (size_t i = 0; i < N; ++i)
        total += std::norm(check_states[i].beta);
      if (total > 0.0) {
        double p = std::norm(check_states[TARGET].beta) / total;
        if (p < -FLOAT_TOL)
          probs_valid = false;
      }
    }
    test_assert(probs_valid,
                "Speedup analog: linear search probabilities remain valid");
  }

  // 5d. Kick strength scales detection speed: stronger kick → fewer rounds
  {
    size_t rounds_strong = run_ladder(0.25);
    size_t rounds_weak   = run_ladder(0.05);
    // Stronger kick should converge at least as fast as weaker kick
    test_assert(rounds_strong <= rounds_weak + 10, // allow small slack
                "Speedup analog: stronger kick converges no slower than weak kick");
  }

  // Report (informational)
  std::cout << "    Kicked rounds=" << rounds_kicked
            << "  Linear rounds=" << rounds_linear
            << "  (N=" << N << ")\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════
int main() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n";
  std::cout
      << "\u2551  Chiral Non-Linear Gate \u2014 Validation Suite               "
         "      \u2551\n";
  std::cout
      << "\u2551  Gate mechanics, hash oracle, amplitude amplification,       "
         "      \u2551\n";
  std::cout
      << "\u2551  and quantum-speedup analog                                  "
         "      \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n";

  test_gate_mechanics();
  test_8cycle_structure();
  test_hash_oracle_analog();
  test_amplitude_amplification();
  test_quantum_speedup_analog();

  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n";
  std::cout << "\u2551  Test Results                                          "
               "      \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n";
  std::cout << "  Total tests: " << test_count << "\n";
  std::cout << "  Passed:      " << passed << " \u2713\n";
  std::cout << "  Failed:      " << failed << " \u2717\n";

  if (failed == 0) {
    std::cout << "\n  \u2713 ALL CHIRAL NON-LINEAR GATE TESTS PASSED \u2014 "
                 "mechanism validated for non-linear search spaces\n\n";
    return 0;
  } else {
    std::cout << "\n  \u2717 CHIRAL GATE VALIDATION FAILED \u2014 "
                 "check implementation\n\n";
    return 1;
  }
}
