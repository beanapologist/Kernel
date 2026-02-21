/*
 * Bitcoin Proof-of-Work Nonce Search Benchmarking Suite
 *
 * Benchmarks a hybrid kernel-enhanced nonce search against classical brute-force,
 * leveraging the chiral non-linear gate for directed amplification (Euler kicks).
 *
 * The hybrid approach uses a "ladder" of QState oscillators whose imaginary
 * components bias the nonce candidate selection.  Each ladder step applies the
 * chiral non-linear map (Section 2 / ChiralNonlinearGate.hpp), causing selective
 * quadratic amplification on the Im > 0 half-domain, which seeds the nonce
 * candidates before the SHA-256 PoW check.
 *
 * Metrics recorded per run:
 *   - Wall-clock time (milliseconds)
 *   - Total hash attempts
 *   - First valid nonce found (if any)
 *   - Success rate across multiple trials
 */

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <openssl/sha.h>

// ── Constants (mirrors quantum_kernel_v2.cpp) ────────────────────────────────
constexpr double ETA             = 0.70710678118654752440;  // 1/√2
constexpr double COHERENCE_TOLERANCE = 1e-9;

using Cx = std::complex<double>;
const Cx MU{ -ETA, ETA };  // µ = e^{i3π/4}

// ── Minimal QState (self-contained; does not include quantum_kernel_v2.cpp) ──
struct QState {
    Cx alpha{ ETA, 0.0 };
    Cx beta { -0.5, 0.5 };  // e^{i3π/4}/√2

    double radius() const {
        return std::abs(alpha) > COHERENCE_TOLERANCE
             ? std::abs(beta) / std::abs(alpha) : 0.0;
    }

    void step() { beta *= MU; }
};

// ── Chiral non-linear gate (inline; avoids ODR conflict with kernel header) ──
static const Cx CHIRAL_MU{ -ETA, ETA };

static inline QState chiral_nonlinear_local(QState state, double kick_strength) {
    const bool positive_imag = (state.beta.imag() > 0.0);
    state.beta *= CHIRAL_MU;
    if (positive_imag && kick_strength != 0.0) {
        state.beta += kick_strength * state.beta * std::abs(state.beta);
    }
    return state;
}

// ── SHA-256 helpers ───────────────────────────────────────────────────────────

// Returns lowercase hex SHA-256 of the input string.
static std::string sha256_hex(const std::string& input) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.c_str()),
           input.size(), digest);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return oss.str();
}

// Returns true when the hex digest has at least `difficulty` leading zero
// nibbles (half-bytes), giving a simple adjustable PoW difficulty knob.
static bool check_hash(const std::string& hex_digest, size_t difficulty) {
    if (difficulty > hex_digest.size()) return false;
    for (size_t i = 0; i < difficulty; ++i) {
        if (hex_digest[i] != '0') return false;
    }
    return true;
}

// ── Benchmark result types ────────────────────────────────────────────────────

struct SearchResult {
    bool     found;
    uint64_t nonce;
    uint64_t attempts;
    double   elapsed_ms;
};

// ── Brute-force search ────────────────────────────────────────────────────────

SearchResult brute_force_search(const std::string& block_header,
                                uint64_t           max_nonce,
                                size_t             difficulty) {
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t attempts = 0;

    for (uint64_t nonce = 0; nonce <= max_nonce; ++nonce) {
        ++attempts;
        std::string candidate = block_header + std::to_string(nonce);
        if (check_hash(sha256_hex(candidate), difficulty)) {
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            return { true, nonce, attempts, elapsed };
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return { false, 0, attempts, elapsed };
}

// ── Ladder / hybrid kernel search ────────────────────────────────────────────
// LADDER_DIM controls how many QState oscillators are maintained per step.
// Each oscillator's imaginary component seeds a candidate nonce offset.

constexpr size_t LADDER_DIM = 16;

// One ladder step: evolve every oscillator through the chiral non-linear gate
// and test the resulting candidate nonces against the PoW target.
// Returns the first valid nonce found, or 0 with found=false if none matched.
static SearchResult ladder_search(const std::string& block_header,
                                  uint64_t           max_nonce,
                                  size_t             difficulty,
                                  double             kick_strength = 0.05) {
    auto start = std::chrono::high_resolution_clock::now();

    // Initialise ladder oscillators with phase-staggered states
    std::vector<QState> psi(LADDER_DIM);
    for (size_t i = 0; i < LADDER_DIM; ++i) {
        // Rotate each oscillator by i steps to spread initial phases
        for (size_t s = 0; s < i; ++s) {
            psi[i].step();
        }
    }

    uint64_t attempts  = 0;
    uint64_t base      = 0;  // Current nonce window base

    while (base <= max_nonce) {
        // One pass: each oscillator proposes a candidate nonce offset
        for (size_t i = 0; i < LADDER_DIM; ++i) {
            // Apply chiral non-linear gate (Euler kick on Im > 0 domain)
            psi[i] = chiral_nonlinear_local(psi[i], kick_strength);

            // Derive candidate nonce: |Im(β)| maps to an offset within the window.
            // Use modulo to keep offset in [0, LADDER_DIM) regardless of |Im(β)| magnitude.
            uint64_t offset = static_cast<uint64_t>(std::abs(psi[i].beta.imag())
                                                     * static_cast<double>(LADDER_DIM))
                              % LADDER_DIM;
            uint64_t candidate_nonce = base + offset;


            ++attempts;
            std::string input = block_header + std::to_string(candidate_nonce);
            if (check_hash(sha256_hex(input), difficulty)) {
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
                return { true, candidate_nonce, attempts, elapsed };
            }
        }

        // Advance window by LADDER_DIM nonces
        base += LADDER_DIM;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return { false, 0, attempts, elapsed };
}

// ── Benchmark harness ─────────────────────────────────────────────────────────

struct BenchmarkRun {
    size_t              difficulty;
    uint64_t            max_nonce;
    std::vector<SearchResult> brute_results;
    std::vector<SearchResult> hybrid_results;
};

static void print_run_summary(const std::string& label, const std::vector<SearchResult>& results) {
    if (results.empty()) return;

    size_t found_count   = 0;
    double total_ms      = 0.0;
    uint64_t total_att   = 0;

    for (const auto& r : results) {
        if (r.found) ++found_count;
        total_ms  += r.elapsed_ms;
        total_att += r.attempts;
    }

    double mean_ms  = total_ms  / results.size();
    double mean_att = static_cast<double>(total_att) / results.size();
    double success  = static_cast<double>(found_count) / results.size() * 100.0;

    std::cout << "  " << std::left << std::setw(20) << label
              << "  success: " << std::fixed << std::setprecision(1)
              << std::setw(5) << success << "%"
              << "  mean attempts: " << std::setprecision(0)
              << std::setw(10) << mean_att
              << "  mean time: " << std::setprecision(3)
              << std::setw(8) << mean_ms << " ms\n";
}

static BenchmarkRun run_benchmark(size_t   difficulty,
                                  uint64_t max_nonce,
                                  int      trials,
                                  const std::string& block_header) {
    BenchmarkRun run;
    run.difficulty = difficulty;
    run.max_nonce  = max_nonce;

    std::cout << "\n  difficulty=" << difficulty
              << "  max_nonce=" << max_nonce
              << "  trials=" << trials << "\n";

    for (int t = 0; t < trials; ++t) {
        // Vary block header slightly across trials to get different nonce targets
        std::string header = block_header + "_trial" + std::to_string(t);
        run.brute_results.push_back(brute_force_search(header, max_nonce, difficulty));
        run.hybrid_results.push_back(ladder_search(header, max_nonce, difficulty));
    }

    print_run_summary("brute-force",  run.brute_results);
    print_run_summary("hybrid-kernel", run.hybrid_results);
    return run;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Bitcoin PoW Nonce Search — Hybrid Kernel Benchmark Suite   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nCompares brute-force SHA-256 nonce search against the hybrid\n";
    std::cout << "kernel-enhanced search (chiral non-linear gate / Euler kicks).\n";
    std::cout << "\nDifficulty is expressed in leading zero nibbles of the hex digest.\n";
    std::cout << "LADDER_DIM = " << LADDER_DIM << " oscillators per step.\n";

    // Representative block-header prefix (simulates Bitcoin block data)
    const std::string BLOCK_HEADER = "00000000000000000003a1b2c3d4e5f6_height=840000";

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
    std::cout << "\n╔═══ Benchmark 4: Kick-Strength Sensitivity (difficulty=1) ═══╗\n";
    std::cout << "\n  Sweeping Euler kick strength k ∈ {0.00, 0.05, 0.10, 0.20}:\n";
    for (double k : { 0.00, 0.05, 0.10, 0.20 }) {
        std::vector<SearchResult> results;
        for (int t = 0; t < 5; ++t) {
            std::string header = BLOCK_HEADER + "_kick_trial" + std::to_string(t);
            results.push_back(ladder_search(header, 50000, 1, k));
        }
        std::ostringstream lbl;
        lbl << "k=" << std::fixed << std::setprecision(2) << k;
        print_run_summary(lbl.str(), results);
    }

    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                     Benchmark Suite Complete                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nNotes:\n";
    std::cout << "  • SHA-256 PoW is computationally random; the hybrid kernel does not\n";
    std::cout << "    break this hardness but explores directed amplification patterns.\n";
    std::cout << "  • Kick strength k=0 makes ladder_search equivalent to a strided scan.\n";
    std::cout << "  • Higher k increases phase dispersion across oscillator states,\n";
    std::cout << "    broadening the candidate nonce distribution per window.\n";

    return 0;
}
