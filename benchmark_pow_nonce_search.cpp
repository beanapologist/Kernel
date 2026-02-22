/*
 * Bitcoin Proof-of-Work Nonce Search Benchmarking Suite
 *
 * Benchmarks a hybrid kernel-enhanced nonce search against classical
 * brute-force, leveraging the chiral non-linear gate for directed amplification
 * (Euler kicks).
 *
 * The hybrid approach uses a "ladder" of QState oscillators whose imaginary
 * components bias the nonce candidate selection.  Each ladder step applies the
 * chiral non-linear map (Section 2 / ChiralNonlinearGate.hpp), causing
 * selective quadratic amplification on the Im > 0 half-domain, which seeds the
 * nonce candidates before the SHA-256 PoW check.
 *
 * Metrics recorded per run:
 *   - Wall-clock time (milliseconds)
 *   - Total hash attempts
 *   - First valid nonce found (if any)
 *   - Success rate across multiple trials
 *
 * Proof-of-Concept (PoC) section:
 *   - Continuous multi-window nonce search collecting every valid nonce found
 *   - Per-discovery output: nonce, valid hex digest, discovering oscillator
 * index
 *   - Euler-kick coherence trace: oscillator |β| evolution across ladder steps
 *   - Side-by-side comparison of brute-force and hybrid findings
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <openssl/sha.h>
#include <sstream>
#include <string>
#include <vector>

// ── Constants (mirrors quantum_kernel_v2.cpp) ────────────────────────────────
constexpr double ETA = 0.70710678118654752440; // 1/√2
constexpr double COHERENCE_TOLERANCE = 1e-9;

using Cx = std::complex<double>;
const Cx MU{-ETA, ETA}; // µ = e^{i3π/4}

// ── Minimal QState (self-contained; does not include quantum_kernel_v2.cpp) ──
struct QState {
  Cx alpha{ETA, 0.0};
  Cx beta{-0.5, 0.5}; // e^{i3π/4}/√2

  double radius() const {
    return std::abs(alpha) > COHERENCE_TOLERANCE
               ? std::abs(beta) / std::abs(alpha)
               : 0.0;
  }

  void step() { beta *= MU; }
};

// ── Chiral non-linear gate (inline; avoids ODR conflict with kernel header) ──
static const Cx CHIRAL_MU{-ETA, ETA};

static inline QState chiral_nonlinear_local(QState state,
                                            double kick_strength) {
  const bool positive_imag = (state.beta.imag() > 0.0);
  state.beta *= CHIRAL_MU;
  if (positive_imag && kick_strength != 0.0) {
    state.beta += kick_strength * state.beta * std::abs(state.beta);
  }
  return state;
}

// ── SHA-256 helpers
// ───────────────────────────────────────────────────────────

// Returns lowercase hex SHA-256 of the input string.
static std::string sha256_hex(const std::string &input) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(input.c_str()), input.size(),
         digest);

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    oss << std::setw(2) << static_cast<unsigned>(digest[i]);
  }
  return oss.str();
}

// Returns true when the hex digest has at least `difficulty` leading zero
// nibbles (half-bytes), giving a simple adjustable PoW difficulty knob.
static bool check_hash(const std::string &hex_digest, size_t difficulty) {
  if (difficulty > hex_digest.size())
    return false;
  for (size_t i = 0; i < difficulty; ++i) {
    if (hex_digest[i] != '0')
      return false;
  }
  return true;
}

// ── Benchmark result types
// ────────────────────────────────────────────────────

struct SearchResult {
  bool found;
  uint64_t nonce;
  uint64_t attempts;
  double elapsed_ms;
};

// ── Brute-force search
// ────────────────────────────────────────────────────────

SearchResult brute_force_search(const std::string &block_header,
                                uint64_t max_nonce, size_t difficulty) {
  auto start = std::chrono::high_resolution_clock::now();
  uint64_t attempts = 0;

  for (uint64_t nonce = 0; nonce <= max_nonce; ++nonce) {
    ++attempts;
    std::string candidate = block_header + std::to_string(nonce);
    if (check_hash(sha256_hex(candidate), difficulty)) {
      auto end = std::chrono::high_resolution_clock::now();
      double elapsed =
          std::chrono::duration<double, std::milli>(end - start).count();
      return {true, nonce, attempts, elapsed};
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  return {false, 0, attempts, elapsed};
}

// ── Ladder / hybrid kernel search ────────────────────────────────────────────
// LADDER_DIM controls how many QState oscillators are maintained per step.
// Each oscillator's imaginary component seeds a candidate nonce offset.

constexpr size_t LADDER_DIM = 16;

// One ladder step: evolve every oscillator through the chiral non-linear gate
// and test the resulting candidate nonces against the PoW target.
// Returns the first valid nonce found, or 0 with found=false if none matched.
static SearchResult ladder_search(const std::string &block_header,
                                  uint64_t max_nonce, size_t difficulty,
                                  double kick_strength = 0.05) {
  auto start = std::chrono::high_resolution_clock::now();

  // Initialise ladder oscillators with phase-staggered states
  std::vector<QState> psi(LADDER_DIM);
  for (size_t i = 0; i < LADDER_DIM; ++i) {
    // Rotate each oscillator by i steps to spread initial phases
    for (size_t s = 0; s < i; ++s) {
      psi[i].step();
    }
  }

  uint64_t attempts = 0;
  uint64_t base = 0; // Current nonce window base

  while (base <= max_nonce) {
    // One pass: each oscillator proposes a candidate nonce offset
    for (size_t i = 0; i < LADDER_DIM; ++i) {
      // Apply chiral non-linear gate (Euler kick on Im > 0 domain)
      psi[i] = chiral_nonlinear_local(psi[i], kick_strength);

      // Derive candidate nonce: |Im(β)| maps to an offset within the window.
      // Use modulo to keep offset in [0, LADDER_DIM) regardless of |Im(β)|
      // magnitude.
      uint64_t offset = static_cast<uint64_t>(std::abs(psi[i].beta.imag()) *
                                              static_cast<double>(LADDER_DIM)) %
                        LADDER_DIM;
      uint64_t candidate_nonce = base + offset;

      ++attempts;
      std::string input = block_header + std::to_string(candidate_nonce);
      if (check_hash(sha256_hex(input), difficulty)) {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration<double, std::milli>(end - start).count();
        return {true, candidate_nonce, attempts, elapsed};
      }
    }

    // Advance window by LADDER_DIM nonces
    base += LADDER_DIM;
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  return {false, 0, attempts, elapsed};
}

// ── Adaptive-kick ladder search ──────────────────────────────────────────────
// Starts with kick_start and decays k toward zero whenever phase dispersion
// (std-dev of |β| / mean |β| across oscillators) exceeds dispersion_threshold.
// This prevents |β| magnitudes from growing unboundedly while preserving the
// directed-amplification benefit early in the search.
static SearchResult ladder_search_adaptive(const std::string &block_header,
                                           uint64_t max_nonce,
                                           size_t difficulty,
                                           double kick_start = 0.05,
                                           double decay_rate = 0.95,
                                           double dispersion_threshold = 0.1) {
  auto start_t = std::chrono::high_resolution_clock::now();

  std::vector<QState> psi(LADDER_DIM);
  for (size_t i = 0; i < LADDER_DIM; ++i)
    for (size_t s = 0; s < i; ++s)
      psi[i].step();

  double k = kick_start;
  uint64_t attempts = 0;
  uint64_t base = 0;

  while (base <= max_nonce) {
    for (size_t i = 0; i < LADDER_DIM; ++i) {
      psi[i] = chiral_nonlinear_local(psi[i], k);

      uint64_t offset = static_cast<uint64_t>(std::abs(psi[i].beta.imag()) *
                                              static_cast<double>(LADDER_DIM)) %
                        LADDER_DIM;
      uint64_t candidate_nonce = base + offset;

      ++attempts;
      std::string input = block_header + std::to_string(candidate_nonce);
      if (check_hash(sha256_hex(input), difficulty)) {
        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration<double, std::milli>(end_t - start_t).count();
        return {true, candidate_nonce, attempts, elapsed};
      }
    }

    // Compute phase dispersion (coefficient of variation of |β|)
    double sum_mag = 0.0, sum_sq = 0.0;
    for (const auto &p : psi) {
      double m = std::abs(p.beta);
      sum_mag += m;
      sum_sq += m * m;
    }
    double mean_mag = sum_mag / LADDER_DIM;
    double variance = sum_sq / LADDER_DIM - mean_mag * mean_mag;
    double dispersion =
        (mean_mag > 1e-12) ? std::sqrt(variance) / mean_mag : 0.0;

    // Decay kick when phase spread exceeds threshold
    if (dispersion > dispersion_threshold)
      k *= decay_rate;

    base += LADDER_DIM;
  }

  auto end_t = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end_t - start_t).count();
  return {false, 0, attempts, elapsed};
}

// ── Proof-of-Concept types
// oscillator index discovered the nonce (LADDER_DIM == brute-force sentinel).
// `cumulative_attempts` is the total hash attempts made up to this discovery
// (cumulative across all prior discoveries in the same run).
struct PoCResult {
  bool found;
  uint64_t nonce;
  std::string hash;             // full 64-char hex digest
  size_t oscillator_idx;        // LADDER_DIM → found by sequential scan
  uint64_t cumulative_attempts; // total hashes tried up to this discovery
  double elapsed_ms;
};

// ── Brute-force PoC search (collects ALL valid nonces up to max_nonce)
// ────────
static std::vector<PoCResult> brute_force_poc(const std::string &block_header,
                                              uint64_t max_nonce,
                                              size_t difficulty,
                                              size_t collect_limit = 5) {
  std::vector<PoCResult> found;
  uint64_t cumulative_attempts = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (uint64_t nonce = 0; nonce <= max_nonce && found.size() < collect_limit;
       ++nonce) {
    ++cumulative_attempts;
    std::string input = block_header + std::to_string(nonce);
    std::string digest = sha256_hex(input);
    if (check_hash(digest, difficulty)) {
      auto now = std::chrono::high_resolution_clock::now();
      double ms =
          std::chrono::duration<double, std::milli>(now - start).count();
      found.push_back(
          {true, nonce, digest, LADDER_DIM, cumulative_attempts, ms});
    }
  }
  return found;
}

// ── Hybrid ladder PoC search (collects ALL valid nonces up to max_nonce) ─────
static std::vector<PoCResult> ladder_poc(const std::string &block_header,
                                         uint64_t max_nonce, size_t difficulty,
                                         double kick_strength,
                                         size_t collect_limit = 5) {
  std::vector<PoCResult> found;
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<QState> psi(LADDER_DIM);
  for (size_t i = 0; i < LADDER_DIM; ++i) {
    for (size_t s = 0; s < i; ++s)
      psi[i].step();
  }

  uint64_t cumulative_attempts = 0;
  uint64_t base = 0;

  while (base <= max_nonce && found.size() < collect_limit) {
    for (size_t i = 0; i < LADDER_DIM && found.size() < collect_limit; ++i) {
      psi[i] = chiral_nonlinear_local(psi[i], kick_strength);

      uint64_t offset = static_cast<uint64_t>(std::abs(psi[i].beta.imag()) *
                                              static_cast<double>(LADDER_DIM)) %
                        LADDER_DIM;
      uint64_t candidate_nonce = base + offset;

      ++cumulative_attempts;
      std::string input = block_header + std::to_string(candidate_nonce);
      std::string digest = sha256_hex(input);
      if (check_hash(digest, difficulty)) {
        auto now = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(now - start).count();
        found.push_back(
            {true, candidate_nonce, digest, i, cumulative_attempts, ms});
      }
    }
    base += LADDER_DIM;
  }
  return found;
}

// ── Euler-kick coherence trace
// ──────────────────────────────────────────────── Shows how the magnitude |β|
// of each oscillator evolves across `steps` ladder steps, illustrating the
// selective amplification on the Im > 0 domain.
static void print_coherence_trace(double kick_strength, size_t steps = 8) {
  // Save stream state so callers see no formatting side effects
  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);

  std::cout << "\n  Oscillator |β| magnitude trace (" << steps
            << " steps, k=" << std::fixed << std::setprecision(2)
            << kick_strength << "):\n";
  std::cout << "  " << std::setw(6) << "step";
  for (size_t i = 0; i < LADDER_DIM; ++i) {
    std::cout << std::setw(8) << ("osc" + std::to_string(i));
  }
  std::cout << "\n  " << std::string(6 + LADDER_DIM * 8, '-') << "\n";

  std::vector<QState> psi(LADDER_DIM);
  for (size_t i = 0; i < LADDER_DIM; ++i) {
    for (size_t s = 0; s < i; ++s)
      psi[i].step();
  }

  for (size_t step = 0; step < steps; ++step) {
    std::cout << "  " << std::setw(6) << step;
    for (size_t i = 0; i < LADDER_DIM; ++i) {
      psi[i] = chiral_nonlinear_local(psi[i], kick_strength);
      std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                << std::abs(psi[i].beta);
    }
    std::cout << "\n";
  }

  // Restore stream state
  std::cout.copyfmt(old_state);
}

// ── PoC printer
// ───────────────────────────────────────────────────────────────
static void print_poc_results(const std::string &method_label,
                              const std::vector<PoCResult> &results) {
  if (results.empty()) {
    std::cout << "  [" << method_label << "] No valid nonces found in range.\n";
    return;
  }
  for (size_t n = 0; n < results.size(); ++n) {
    const auto &r = results[n];
    std::cout << "  [" << method_label << " #" << (n + 1) << "]\n";
    std::cout << "    nonce     : " << r.nonce << "\n";
    std::cout << "    hash      : " << r.hash << "\n";
    if (r.oscillator_idx < LADDER_DIM) {
      std::cout << "    found by  : oscillator " << r.oscillator_idx << "\n";
    } else {
      std::cout << "    found by  : sequential scan\n";
    }
    std::cout << "    cumulative: " << r.cumulative_attempts << " attempts\n";
    std::cout << "    time      : " << std::fixed << std::setprecision(3)
              << r.elapsed_ms << " ms\n";
  }
}

// ── Benchmark harness
// ─────────────────────────────────────────────────────────

struct BenchmarkRun {
  size_t difficulty;
  uint64_t max_nonce;
  std::vector<SearchResult> brute_results;
  std::vector<SearchResult> hybrid_results;
};

static void print_run_summary(const std::string &label,
                              const std::vector<SearchResult> &results) {
  if (results.empty())
    return;

  size_t found_count = 0;
  double total_ms = 0.0;
  uint64_t total_att = 0;

  for (const auto &r : results) {
    if (r.found)
      ++found_count;
    total_ms += r.elapsed_ms;
    total_att += r.attempts;
  }

  double mean_ms = total_ms / results.size();
  double mean_att = static_cast<double>(total_att) / results.size();
  double success = static_cast<double>(found_count) / results.size() * 100.0;

  std::cout << "  " << std::left << std::setw(20) << label
            << "  success: " << std::fixed << std::setprecision(1)
            << std::setw(5) << success << "%"
            << "  mean attempts: " << std::setprecision(0) << std::setw(10)
            << mean_att << "  mean time: " << std::setprecision(3)
            << std::setw(8) << mean_ms << " ms\n";
}

static BenchmarkRun run_benchmark(size_t difficulty, uint64_t max_nonce,
                                  int trials, const std::string &block_header) {
  BenchmarkRun run;
  run.difficulty = difficulty;
  run.max_nonce = max_nonce;

  std::cout << "\n  difficulty=" << difficulty << "  max_nonce=" << max_nonce
            << "  trials=" << trials << "\n";

  for (int t = 0; t < trials; ++t) {
    // Vary block header slightly across trials to get different nonce targets
    std::string header = block_header + "_trial" + std::to_string(t);
    run.brute_results.push_back(
        brute_force_search(header, max_nonce, difficulty));
    run.hybrid_results.push_back(ladder_search(header, max_nonce, difficulty));
  }

  print_run_summary("brute-force", run.brute_results);
  print_run_summary("hybrid-kernel", run.hybrid_results);
  return run;
}

// ── Main
// ──────────────────────────────────────────────────────────────────────

int main() {
  std::cout
      << "╔═══════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║   Bitcoin PoW Nonce Search — Hybrid Kernel Benchmark Suite   ║\n";
  std::cout
      << "╚═══════════════════════════════════════════════════════════════╝\n";
  std::cout
      << "\nCompares brute-force SHA-256 nonce search against the hybrid\n";
  std::cout
      << "kernel-enhanced search (chiral non-linear gate / Euler kicks).\n";
  std::cout << "\nDifficulty is expressed in leading zero nibbles of the hex "
               "digest.\n";
  std::cout << "LADDER_DIM = " << LADDER_DIM << " oscillators per step.\n";

  // Representative block-header prefix (simulates Bitcoin block data)
  const std::string BLOCK_HEADER =
      "00000000000000000003a1b2c3d4e5f6_height=840000";

  // ── Benchmark 1: Low difficulty (1 leading zero nibble) ──────────────────
  std::cout << "\n╔═══ Benchmark 1: Low Difficulty (1 zero nibble) ═══╗\n";
  run_benchmark(1, 50000, 10, BLOCK_HEADER);

  // ── Benchmark 2: Medium difficulty (2 leading zero nibbles) ──────────────
  std::cout << "\n╔═══ Benchmark 2: Medium Difficulty (2 zero nibbles) ═══╗\n";
  run_benchmark(2, 200000, 5, BLOCK_HEADER);

  // ── Benchmark 3: Higher difficulty (3 leading zero nibbles) ──────────────
  std::cout << "\n╔═══ Benchmark 3: Higher Difficulty (3 zero nibbles) ═══╗\n";
  run_benchmark(3, 500000, 3, BLOCK_HEADER);

  // ── Kick-strength sensitivity sweep (difficulty=1) ───────────────────────
  std::cout
      << "\n╔═══ Benchmark 4: Kick-Strength Sensitivity (difficulty=1) ═══╗\n";
  std::cout
      << "\n  Sweeping Euler kick strength k ∈ {0.00, 0.05, 0.10, 0.20}:\n";
  for (double k : {0.00, 0.05, 0.10, 0.20}) {
    std::vector<SearchResult> results;
    for (int t = 0; t < 5; ++t) {
      std::string header = BLOCK_HEADER + "_kick_trial" + std::to_string(t);
      results.push_back(ladder_search(header, 50000, 1, k));
    }
    std::ostringstream lbl;
    lbl << "k=" << std::fixed << std::setprecision(2) << k;
    print_run_summary(lbl.str(), results);
  }

  // ── Benchmark 5: Adaptive kick strength (difficulty=1) ──────────────────
  std::cout
      << "\n╔═══ Benchmark 5: Adaptive Kick Strength (difficulty=1) ═══╗\n";
  std::cout << "\n  Compares fixed k=0.05 against adaptive kick "
               "(start=0.05, decay=0.95):\n";
  {
    std::vector<SearchResult> fixed_results, adaptive_results;
    for (int t = 0; t < 5; ++t) {
      std::string header = BLOCK_HEADER + "_adapt_trial" + std::to_string(t);
      fixed_results.push_back(ladder_search(header, 50000, 1, 0.05));
      adaptive_results.push_back(
          ladder_search_adaptive(header, 50000, 1, 0.05, 0.95, 0.1));
    }
    print_run_summary("fixed k=0.05", fixed_results);
    print_run_summary("adaptive k", adaptive_results);
  }

  // ── Benchmark 6: High difficulty (4 leading zero nibbles) ────────────────
  std::cout << "\n╔═══ Benchmark 6: High Difficulty (4 zero nibbles) ═══╗\n";
  std::cout << "\n  Stress-testing scalability at difficulty=4 (expected gap "
               "~65536).\n";
  run_benchmark(4, 2000000, 2, BLOCK_HEADER);

  std::cout << "\n╔════════════════════════════════════════════════════════════"
               "═══╗\n";
  std::cout
      << "║                     Benchmark Suite Complete                 ║\n";
  std::cout
      << "╚═══════════════════════════════════════════════════════════════╝\n";
  std::cout << "\nNotes:\n";
  std::cout << "  • SHA-256 PoW is computationally random; the hybrid kernel "
               "does not\n";
  std::cout << "    break this hardness but explores directed amplification "
               "patterns.\n";
  std::cout << "  • Kick strength k=0 makes ladder_search equivalent to a "
               "strided scan.\n";
  std::cout
      << "  • Higher k increases phase dispersion across oscillator states,\n";
  std::cout << "    broadening the candidate nonce distribution per window.\n";
  std::cout
      << "  • Adaptive kick decays k when dispersion exceeds threshold,\n";
  std::cout << "    preserving late-stage phase structure and preventing |β| "
               "growth.\n";

  // ── PoC 1: Coherence trace ────────────────────────────────────────────────
  std::cout << "\n╔═══ Proof-of-Concept 1: Euler-Kick Coherence Trace ═══╗\n";
  std::cout
      << "\nShows how the chiral non-linear gate amplifies |β| on the Im > 0\n";
  std::cout
      << "domain (Euler kick) vs. preserves it on Im ≤ 0 (linear gate).\n";
  print_coherence_trace(0.05);

  // ── PoC 2: Continuous multi-nonce discovery — brute-force ─────────────────
  std::cout << "\n╔═══ Proof-of-Concept 2: Continuous Nonce Search "
               "(brute-force) ═══╗\n";
  std::cout << "\nCollecting first 5 valid nonces with difficulty=1 "
               "(max_nonce=200000):\n";
  auto bf_poc = brute_force_poc(BLOCK_HEADER, 200000, 1, 5);
  print_poc_results("brute-force", bf_poc);

  // ── PoC 3: Continuous multi-nonce discovery — hybrid kernel ───────────────
  std::cout << "\n╔═══ Proof-of-Concept 3: Continuous Nonce Search (hybrid "
               "kernel) ═══╗\n";
  std::cout << "\nCollecting first 5 valid nonces with difficulty=1 "
               "(max_nonce=200000, k=0.05):\n";
  auto hybrid_poc = ladder_poc(BLOCK_HEADER, 200000, 1, 0.05, 5);
  print_poc_results("hybrid", hybrid_poc);

  // ── PoC 4: Side-by-side comparison ────────────────────────────────────────
  std::cout << "\n╔═══ Proof-of-Concept 4: Side-by-Side PoC Comparison ═══╗\n";
  std::cout << "\n  Block header : " << BLOCK_HEADER << "\n";
  std::cout << "  Difficulty   : 1 leading zero nibble\n";
  std::cout << "  Nonce range  : 0 – 200000\n\n";

  // Build a comparable set: first find for each method across 5 block headers
  std::cout
      << "  "
         "┌────────────────────┬──────────────┬──────────────┬────────────┐\n";
  std::cout << "  │ Method             │ Nonce        │ Attempts     │ Time "
               "(ms)  │\n";
  std::cout
      << "  "
         "├────────────────────┼──────────────┼──────────────┼────────────┤\n";

  for (int t = 0; t < 5; ++t) {
    std::string hdr = BLOCK_HEADER + "_poc" + std::to_string(t);

    SearchResult bf = brute_force_search(hdr, 200000, 1);
    SearchResult hy = ladder_search(hdr, 200000, 1);

    auto fmt_nonce = [](const SearchResult &r) {
      return r.found ? std::to_string(r.nonce) : "—";
    };

    std::cout << "  │ brute-force        │ " << std::left << std::setw(12)
              << fmt_nonce(bf) << " │ " << std::setw(12) << bf.attempts << " │ "
              << std::setw(10) << std::fixed << std::setprecision(3)
              << bf.elapsed_ms << " │\n";
    std::cout << "  │ hybrid (k=0.05)    │ " << std::left << std::setw(12)
              << fmt_nonce(hy) << " │ " << std::setw(12) << hy.attempts << " │ "
              << std::setw(10) << std::fixed << std::setprecision(3)
              << hy.elapsed_ms << " │\n";
    if (t < 4) {
      std::cout << "  "
                   "├────────────────────┼──────────────┼──────────────┼───────"
                   "─────┤\n";
    }
  }
  std::cout
      << "  "
         "└────────────────────┴──────────────┴──────────────┴────────────┘\n";

  std::cout << "\n╔════════════════════════════════════════════════════════════"
               "═══╗\n";
  std::cout
      << "║                  Proof-of-Concept Complete                   ║\n";
  std::cout
      << "╚═══════════════════════════════════════════════════════════════╝\n";
  std::cout << "\nSummary:\n";
  std::cout
      << "  • Each valid nonce above is verifiable: SHA-256(header + nonce)\n";
  std::cout << "    produces a digest whose leading nibble is '0'.\n";
  std::cout
      << "  • The hybrid kernel explores nonce space via phase-staggered\n";
  std::cout
      << "    oscillators; the Euler kick (k>0) biases Im > 0 half-domain.\n";
  std::cout
      << "  • Coherence trace (PoC 1) confirms selective |β| amplification\n";
  std::cout << "    on positive-imaginary steps vs. flat magnitude on Im ≤ 0 "
               "steps.\n";

  return 0;
}
