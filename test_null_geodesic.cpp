/*
 * Test Suite for Null Geodesic Scheduler
 *
 * March 2026 Addendum: Stochastic Noise Resilience
 *
 * Validates:
 *   - Null metric: ds²(ξ(r)) = tanh²(ln r), null iff r=1
 *   - Coherent state constructor ξ_k(r) = r·µ^k
 *   - Z/8Z symmetric rotators: step^8 = identity
 *   - Geodesic invariants: ds²+C²=1, C(r)=sech(ln r), R(r) monotone
 *   - Recovery mechanism driving r → 1 (coherence restoration)
 *   - Stochastic noise resilience with Gaussian perturbations
 *   - IPC coherence-gate blocking under decoherence
 *   - Orbit stability and convergence benchmarks
 */

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdint>

// ── Constants (mirrored from quantum_kernel_v2.cpp) ───────────────────────────
constexpr double ETA         = 0.70710678118654752440;  // 1/√2
constexpr double DELTA_S     = 2.41421356237309504880;  // δ_S = 1+√2
constexpr double DELTA_CONJ  = 0.41421356237309504880;  // √2-1 = 1/δ_S
constexpr double PI          = 3.14159265358979323846;

constexpr double COHERENCE_TOLERANCE = 1e-9;
constexpr double RADIUS_TOLERANCE    = 1e-9;
constexpr double TIGHT_TOL           = 1e-12;

constexpr double DECOHERENCE_MINOR   = 0.05;
constexpr double DECOHERENCE_MAJOR   = 0.15;
constexpr double MIN_RECOVERY_RADIUS = 0.1;
constexpr double MAX_RECOVERY_RADIUS = 10.0;
constexpr double NULL_LOG_EPSILON    = 1e-15;  // Floor for log(r) guard

using Cx = std::complex<double>;
const Cx MU{ -ETA, ETA };  // µ = e^{i3π/4}

// ── Core math (mirrored from quantum_kernel_v2.cpp) ───────────────────────────
double coherence(double r)        { return (2.0 * r) / (1.0 + r * r); }
double lyapunov(double r)         { return std::log(r); }
double coherence_sech(double lam) { return 1.0 / std::cosh(lam); }
double palindrome_residual(double r) { return (1.0 / DELTA_S) * (r - 1.0 / r); }

// ── Null metric: ds²(ξ(r)) = tanh²(ln r) ─────────────────────────────────────
// Identity: ds²(ξ(r)) + C(r)² = tanh²(ln r) + sech²(ln r) = 1
// Null condition: ds²(ξ(1)) = 0
double null_metric(double r) {
    double lnr = std::log(r > 0 ? r : NULL_LOG_EPSILON);
    double t = std::tanh(lnr);
    return t * t;
}

// ── Decoherence classification ────────────────────────────────────────────────
enum class DecoherenceLevel { NONE, MINOR, MAJOR, CRITICAL };

DecoherenceLevel measure_decoherence(double r) {
    double dev = std::abs(r - 1.0);
    if (dev <= RADIUS_TOLERANCE)   return DecoherenceLevel::NONE;
    if (dev <= DECOHERENCE_MINOR)  return DecoherenceLevel::MINOR;
    if (dev <= DECOHERENCE_MAJOR)  return DecoherenceLevel::MAJOR;
    return DecoherenceLevel::CRITICAL;
}

// ── NullGeodesicState ─────────────────────────────────────────────────────────
struct NullGeodesicState {
    double  r;   // Radial parameter (r=1 is balanced/null)
    uint8_t k;   // Z/8Z position (0..7)

    NullGeodesicState() : r(1.0), k(0) {}
    NullGeodesicState(double r_, uint8_t k_ = 0) : r(r_), k(k_ % 8) {}

    // ξ_k(r) = r · µ^k
    Cx xi() const {
        Cx mu_power{1.0, 0.0};
        for (int i = 0; i < k; ++i) mu_power *= MU;
        return r * mu_power;
    }

    double ds_squared()       const { return null_metric(r); }
    double geodesic_coherence() const { return coherence_sech(lyapunov(r > 0 ? r : NULL_LOG_EPSILON)); }
    double residual()         const { return palindrome_residual(r); }
    bool   is_null()          const { return ds_squared() < RADIUS_TOLERANCE * RADIUS_TOLERANCE; }

    NullGeodesicState rotated() const {
        return NullGeodesicState{r, static_cast<uint8_t>((k + 1) % 8)};
    }
};

// ── GeodesicRotator ───────────────────────────────────────────────────────────
class GeodesicRotator {
public:
    static NullGeodesicState step(const NullGeodesicState& g) { return g.rotated(); }

    static NullGeodesicState full_orbit(const NullGeodesicState& g) {
        NullGeodesicState result = g;
        for (int i = 0; i < 8; ++i) result = step(result);
        return result;
    }

    static bool verify_periodicity(const NullGeodesicState& g) {
        return full_orbit(g).k == g.k;
    }

    static NullGeodesicState recover(const NullGeodesicState& g,
                                     double rate = 0.5,
                                     DecoherenceLevel level = DecoherenceLevel::MINOR) {
        double C      = g.geodesic_coherence();
        double defect = 1.0 - C;
        double mult   = 1.0;
        switch (level) {
            case DecoherenceLevel::MINOR:    mult = 0.5; break;
            case DecoherenceLevel::MAJOR:    mult = 0.8; break;
            case DecoherenceLevel::CRITICAL: mult = 1.0; break;
            case DecoherenceLevel::NONE:     return g;
        }
        double new_r = g.r + (1.0 - g.r) * rate * mult * defect;
        if (new_r < MIN_RECOVERY_RADIUS) new_r = MIN_RECOVERY_RADIUS;
        if (new_r > MAX_RECOVERY_RADIUS) new_r = MAX_RECOVERY_RADIUS;
        return NullGeodesicState{new_r, g.k};
    }
};

// ── GeodesicScheduler ─────────────────────────────────────────────────────────
class GeodesicScheduler {
public:
    struct ScheduledGeodesic {
        uint32_t          id;
        NullGeodesicState geo;
        uint64_t          scheduled_at;
        uint64_t          interrupts = 0;
        bool              active     = true;
    };

    struct Config {
        double   noise_sigma     = 0.05;
        double   recovery_rate   = 0.5;
        bool     enable_recovery = true;
        uint32_t seed            = 42;
    };

    GeodesicScheduler() : config_(Config{}), rng_(42) {}
    explicit GeodesicScheduler(const Config& cfg) : config_(cfg), rng_(cfg.seed) {}

    uint32_t schedule(double r = 1.0, uint8_t k = 0) {
        uint32_t id = next_id_++;
        geodesics_.push_back({id, NullGeodesicState{r, k}, tick_, 0, true});
        return id;
    }

    void tick() {
        ++tick_;
        for (auto& sg : geodesics_)
            if (sg.active) sg.geo = GeodesicRotator::step(sg.geo);
    }

    void apply_noise(double sigma = -1.0) {
        if (sigma < 0.0) sigma = config_.noise_sigma;
        std::normal_distribution<double> noise(0.0, sigma);
        for (auto& sg : geodesics_) {
            if (!sg.active) continue;
            double new_r = sg.geo.r * (1.0 + noise(rng_));
            if (new_r < MIN_RECOVERY_RADIUS) new_r = MIN_RECOVERY_RADIUS;
            if (new_r > MAX_RECOVERY_RADIUS) new_r = MAX_RECOVERY_RADIUS;
            sg.geo.r = new_r;
        }
    }

    uint32_t recover_all() {
        if (!config_.enable_recovery) return 0;
        uint32_t count = 0;
        for (auto& sg : geodesics_) {
            if (!sg.active) continue;
            double old_dev = std::abs(sg.geo.r - 1.0);
            DecoherenceLevel level = measure_decoherence(sg.geo.r);
            if (level != DecoherenceLevel::NONE) {
                ++sg.interrupts;
                sg.geo = GeodesicRotator::recover(sg.geo, config_.recovery_rate, level);
                if (std::abs(sg.geo.r - 1.0) < old_dev) ++count;
            }
        }
        return count;
    }

    double mean_deviation() const {
        double sum = 0.0; uint32_t n = 0;
        for (const auto& sg : geodesics_)
            if (sg.active) { sum += std::abs(sg.geo.r - 1.0); ++n; }
        return n > 0 ? sum / n : 0.0;
    }

    uint32_t null_count() const {
        uint32_t n = 0;
        for (const auto& sg : geodesics_)
            if (sg.active && sg.geo.is_null()) ++n;
        return n;
    }

    const std::vector<ScheduledGeodesic>& geodesics() const { return geodesics_; }
    uint64_t tick_count() const { return tick_; }
    Config config_;

private:
    std::vector<ScheduledGeodesic> geodesics_;
    uint32_t next_id_ = 1;
    uint64_t tick_    = 0;
    std::mt19937 rng_;
};

// ── Test infrastructure ────────────────────────────────────────────────────────
int test_count = 0, passed = 0, failed = 0;

void test_assert(bool condition, const std::string& name) {
    ++test_count;
    if (condition) { std::cout << "  ✓ " << name << "\n"; ++passed; }
    else           { std::cout << "  ✗ FAILED: " << name << "\n"; ++failed; }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — Null metric ds²(ξ(r)) = tanh²(ln r)
// ══════════════════════════════════════════════════════════════════════════════
void test_null_metric() {
    std::cout << "\n╔═══ Test 1: Null Metric ds²(ξ(r)) = tanh²(ln r) ═══╗\n";

    // Null condition: ds²(ξ(1)) = 0 exactly
    test_assert(std::abs(null_metric(1.0)) < TIGHT_TOL,
                "ds²(ξ(1)) = 0 exactly (null geodesic)");

    // ds²(ξ(r)) > 0 for r ≠ 1
    for (double r : {0.5, 0.8, 1.1, 1.5, 2.0}) {
        test_assert(null_metric(r) > COHERENCE_TOLERANCE,
                    "ds²(ξ(" + std::to_string(r) + ")) > 0 for r≠1");
    }

    // Symmetry: ds²(ξ(r)) = ds²(ξ(1/r))  [even function of ln r]
    for (double r : {0.5, 0.8, 1.5, 2.0}) {
        test_assert(std::abs(null_metric(r) - null_metric(1.0 / r)) < TIGHT_TOL,
                    "ds²(ξ(r)) = ds²(ξ(1/r)) for r=" + std::to_string(r));
    }

    // Identity: ds²(ξ(r)) + C(r)² = 1  [tanh² + sech² = 1]
    for (double r : {0.5, 0.9, 1.0, 1.1, 2.0}) {
        double ds2 = null_metric(r);
        double C   = coherence_sech(lyapunov(r));
        test_assert(std::abs(ds2 + C * C - 1.0) < TIGHT_TOL,
                    "ds²+C²=1 for r=" + std::to_string(r));
    }

    // Monotone: ds²(r) increases as |r-1| increases from 1
    test_assert(null_metric(1.5) > null_metric(1.2),
                "ds² increases with deviation: r=1.5 > r=1.2");
    test_assert(null_metric(0.5) > null_metric(0.8),
                "ds² increases with deviation: r=0.5 > r=0.8");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — Coherent state constructor ξ_k(r) = r·µ^k
// ══════════════════════════════════════════════════════════════════════════════
void test_coherent_state_constructor() {
    std::cout << "\n╔═══ Test 2: Coherent State Constructor ξ_k(r) = r·µ^k ═══╗\n";

    // |ξ_k(1)| = 1 for all k
    for (uint8_t k = 0; k < 8; ++k) {
        NullGeodesicState g{1.0, k};
        test_assert(std::abs(std::abs(g.xi()) - 1.0) < COHERENCE_TOLERANCE,
                    "|ξ_" + std::to_string(k) + "(1)| = 1 exactly");
    }

    // |ξ_k(r)| = r for all k and r
    for (double r : {0.5, 1.0, 1.5, 2.0}) {
        NullGeodesicState g{r, 0};
        test_assert(std::abs(std::abs(g.xi()) - r) < COHERENCE_TOLERANCE,
                    "|ξ_0(" + std::to_string(r) + ")| = r exactly");
    }

    // All 8 geodesic directions are distinct at r=1
    std::vector<Cx> xis;
    for (uint8_t k = 0; k < 8; ++k) xis.push_back(NullGeodesicState{1.0, k}.xi());
    bool all_distinct = true;
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = i + 1; j < 8; ++j)
            if (std::abs(xis[i] - xis[j]) < COHERENCE_TOLERANCE) all_distinct = false;
    test_assert(all_distinct, "All 8 geodesic directions ξ_k are distinct");

    // Null condition: ds²(ξ_k(1)) = 0 for all k
    for (uint8_t k = 0; k < 8; ++k) {
        NullGeodesicState g{1.0, k};
        test_assert(g.is_null(),
                    "Null condition at r=1, k=" + std::to_string(k));
    }

    // arg(ξ_k(r)) = k · 3π/4  (Z/8Z phase structure)
    for (uint8_t k = 0; k < 8; ++k) {
        NullGeodesicState g{1.0, k};
        double expected_arg = k * 3.0 * PI / 4.0;
        // Normalize to [-π, π] range for comparison
        Cx expected = std::polar(1.0, expected_arg);
        test_assert(std::abs(g.xi() - expected) < COHERENCE_TOLERANCE,
                    "arg(ξ_" + std::to_string(k) + ") = k·3π/4");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — Z/8Z symmetric rotators
// ══════════════════════════════════════════════════════════════════════════════
void test_z8z_rotators() {
    std::cout << "\n╔═══ Test 3: Z/8Z Symmetric Rotators ═══╗\n";

    // One rotation: k → (k+1) mod 8
    NullGeodesicState g0{1.0, 0};
    NullGeodesicState g1 = GeodesicRotator::step(g0);
    test_assert(g1.k == 1, "One step: k=0 → k=1");
    test_assert(std::abs(g1.r - g0.r) < TIGHT_TOL, "Step preserves r");

    // Z/8Z wrap: k=7 → k=0
    NullGeodesicState g7{1.0, 7};
    NullGeodesicState g7s = GeodesicRotator::step(g7);
    test_assert(g7s.k == 0, "Z/8Z wrap: step at k=7 → k=0");

    // 8 rotations = identity for every starting position
    for (uint8_t k = 0; k < 8; ++k) {
        NullGeodesicState gk{1.0, k};
        test_assert(GeodesicRotator::verify_periodicity(gk),
                    "step^8 = identity for k=" + std::to_string(k));
    }

    // Rotation preserves null condition (ds² depends only on r, not k)
    for (double r : {0.5, 1.0, 1.5}) {
        NullGeodesicState gr{r, 0};
        NullGeodesicState grs = GeodesicRotator::step(gr);
        test_assert(gr.is_null() == grs.is_null(),
                    "Rotation preserves null condition for r=" + std::to_string(r));
    }

    // Rotation preserves ds², C, R (they depend only on r)
    NullGeodesicState g_r{1.3, 2};
    NullGeodesicState g_rs = GeodesicRotator::step(g_r);
    test_assert(std::abs(g_r.ds_squared()         - g_rs.ds_squared())         < TIGHT_TOL, "Step preserves ds²");
    test_assert(std::abs(g_r.geodesic_coherence() - g_rs.geodesic_coherence()) < TIGHT_TOL, "Step preserves C(r)");
    test_assert(std::abs(g_r.residual()           - g_rs.residual())           < TIGHT_TOL, "Step preserves R(r)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — Geodesic invariants
// ══════════════════════════════════════════════════════════════════════════════
void test_geodesic_invariants() {
    std::cout << "\n╔═══ Test 4: Geodesic Invariants ═══╗\n";

    // C(r) = sech(ln r) via Theorem 14 (sech duality)
    for (double r : {0.5, 0.9, 1.0, 1.1, 2.0}) {
        NullGeodesicState g{r, 0};
        double C_direct = coherence(r);
        double C_geo    = g.geodesic_coherence();
        test_assert(std::abs(C_direct - C_geo) < TIGHT_TOL,
                    "C(r) = sech(ln r) via geodesic at r=" + std::to_string(r));
    }

    // C(r) maximum 1 at r=1 only
    NullGeodesicState g_null{1.0, 0};
    test_assert(std::abs(g_null.geodesic_coherence() - 1.0) < TIGHT_TOL,
                "C(r=1) = 1 (maximum coherence at null geodesic)");

    // R(r) = 0 at r=1 (Corollary 13)
    test_assert(std::abs(g_null.residual()) < TIGHT_TOL,
                "R(r=1) = 0 (palindrome residual at null geodesic)");

    // R(r) sign: < 0 for r<1, > 0 for r>1
    for (double r : {0.5, 0.8}) {
        test_assert(NullGeodesicState{r}.residual() < 0,
                    "R(r) < 0 for r<1, r=" + std::to_string(r));
    }
    for (double r : {1.2, 2.0}) {
        test_assert(NullGeodesicState{r}.residual() > 0,
                    "R(r) > 0 for r>1, r=" + std::to_string(r));
    }

    // Corollary 13 consistency: ds²=0 ↔ C=1 ↔ R=0 ↔ r=1
    for (double r : {0.5, 0.9, 1.0, 1.1, 2.0}) {
        NullGeodesicState g{r, 0};
        bool is_null  = g.is_null();
        bool max_coh  = std::abs(g.geodesic_coherence() - 1.0) < COHERENCE_TOLERANCE;
        bool zero_res = std::abs(g.residual()) < COHERENCE_TOLERANCE;
        test_assert((is_null == max_coh) && (max_coh == zero_res),
                    "Corollary 13 consistency at r=" + std::to_string(r));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — Recovery mechanism (r → 1)
// ══════════════════════════════════════════════════════════════════════════════
void test_recovery_mechanism() {
    std::cout << "\n╔═══ Test 5: Recovery Mechanism (r → 1) ═══╗\n";

    // Recovery drives r toward 1 for r > 1
    NullGeodesicState g_out{1.2, 0};
    auto rec_out = GeodesicRotator::recover(g_out, 0.5, DecoherenceLevel::MINOR);
    test_assert(rec_out.r < g_out.r,
                "Recovery decreases r for r>1");
    test_assert(rec_out.r > 1.0 - COHERENCE_TOLERANCE,
                "Recovery does not overshoot r=1 from above");

    // Recovery drives r toward 1 for r < 1
    NullGeodesicState g_in{0.8, 0};
    auto rec_in = GeodesicRotator::recover(g_in, 0.5, DecoherenceLevel::MINOR);
    test_assert(rec_in.r > g_in.r,
                "Recovery increases r for r<1");
    test_assert(rec_in.r < 1.0 + COHERENCE_TOLERANCE,
                "Recovery does not overshoot r=1 from below");

    // No recovery when level is NONE
    NullGeodesicState g_none{1.3, 0};
    auto rec_none = GeodesicRotator::recover(g_none, 0.5, DecoherenceLevel::NONE);
    test_assert(std::abs(rec_none.r - g_none.r) < TIGHT_TOL,
                "No correction applied when DecoherenceLevel::NONE");

    // Critical recovery stronger than minor (larger step toward r=1)
    NullGeodesicState g_crit{1.5, 0};
    auto rec_minor = GeodesicRotator::recover(g_crit, 0.5, DecoherenceLevel::MINOR);
    auto rec_crit  = GeodesicRotator::recover(g_crit, 0.5, DecoherenceLevel::CRITICAL);
    test_assert(std::abs(rec_crit.r - 1.0) < std::abs(rec_minor.r - 1.0),
                "Critical recovery yields larger correction than minor");

    // Recovery preserves Z/8Z position k
    NullGeodesicState g_k{1.3, 5};
    auto rec_k = GeodesicRotator::recover(g_k, 0.5, DecoherenceLevel::MAJOR);
    test_assert(rec_k.k == 5, "Recovery preserves Z/8Z position k=5");

    // Coherence improves after recovery
    NullGeodesicState g_dev{1.4, 0};
    auto rec_dev = GeodesicRotator::recover(g_dev, 0.5, DecoherenceLevel::MAJOR);
    test_assert(rec_dev.geodesic_coherence() > g_dev.geodesic_coherence(),
                "Coherence C(r) improves after recovery");

    // ds² decreases after recovery
    test_assert(rec_dev.ds_squared() < g_dev.ds_squared(),
                "ds² decreases after recovery");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 6 — Stochastic noise resilience
// ══════════════════════════════════════════════════════════════════════════════
void test_stochastic_noise_resilience() {
    std::cout << "\n╔═══ Test 6: Stochastic Noise Resilience ═══╗\n";

    GeodesicScheduler::Config cfg;
    cfg.noise_sigma    = 0.05;
    cfg.recovery_rate  = 0.6;
    cfg.seed           = 42;
    GeodesicScheduler sched(cfg);

    // Schedule 8 null geodesics (one per Z/8Z position)
    for (uint8_t k = 0; k < 8; ++k) sched.schedule(1.0, k);

    test_assert(sched.null_count() == 8,
                "All 8 geodesics start null (r=1)");
    test_assert(std::abs(sched.mean_deviation()) < TIGHT_TOL,
                "Initial mean deviation = 0");

    // Apply noise: some geodesics deviate from null
    sched.apply_noise(0.1);
    test_assert(sched.mean_deviation() > 0.0,
                "Gaussian noise increases mean deviation");

    // Recovery rounds reduce mean deviation
    double dev_before = sched.mean_deviation();
    for (int i = 0; i < 10; ++i) sched.recover_all();
    double dev_after = sched.mean_deviation();
    test_assert(dev_after < dev_before,
                "Recovery rounds reduce mean deviation toward null");

    // ds²+C²=1 invariant maintained under noise+recovery
    for (const auto& sg : sched.geodesics()) {
        double ds2 = sg.geo.ds_squared();
        double C   = sg.geo.geodesic_coherence();
        test_assert(std::abs(ds2 + C * C - 1.0) < TIGHT_TOL,
                    "ds²+C²=1 invariant maintained for geodesic " +
                    std::to_string(sg.id));
    }

    // Interrupt counter is positive: recoveries were triggered
    uint64_t total_interrupts = 0;
    for (const auto& sg : sched.geodesics()) total_interrupts += sg.interrupts;
    test_assert(total_interrupts > 0,
                "Decoherence interrupts triggered during noise+recovery");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 7 — Scheduler tick and Z/8Z rotation
// ══════════════════════════════════════════════════════════════════════════════
void test_scheduler_tick_and_rotation() {
    std::cout << "\n╔═══ Test 7: Scheduler Tick and Z/8Z Rotation ═══╗\n";

    GeodesicScheduler sched;
    sched.schedule(1.0, 0);
    sched.schedule(1.0, 3);

    sched.tick();
    test_assert(sched.geodesics()[0].geo.k == 1, "k advances 0→1 after tick");
    test_assert(sched.geodesics()[1].geo.k == 4, "k advances 3→4 after tick");
    test_assert(sched.tick_count() == 1, "tick_count increments");

    // After 8 total ticks, k returns to original (Z/8Z period)
    for (int i = 0; i < 7; ++i) sched.tick();
    test_assert(sched.geodesics()[0].geo.k == 0, "After 8 ticks k=0 returns to 0");
    test_assert(sched.geodesics()[1].geo.k == 3, "After 8 ticks k=3 returns to 3");

    // Tick does not change r (no noise applied)
    test_assert(std::abs(sched.geodesics()[0].geo.r - 1.0) < TIGHT_TOL,
                "Tick alone does not change r");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 8 — IPC blocking under decoherence
// ══════════════════════════════════════════════════════════════════════════════
void test_ipc_decoherence_blocking() {
    std::cout << "\n╔═══ Test 8: IPC Blocking Under Decoherence ═══╗\n";

    const double coherence_threshold = 0.7;

    // Null geodesic (r=1): C=1 → passes IPC check
    NullGeodesicState g_null{1.0, 0};
    test_assert(g_null.geodesic_coherence() >= coherence_threshold,
                "Null geodesic (r=1) passes IPC coherence check");

    // Severely decoherent (r=3): C=0.6 < threshold=0.7 → blocked
    NullGeodesicState g_dec{3.0, 0};
    test_assert(g_dec.geodesic_coherence() < coherence_threshold,
                "Decoherent geodesic (r=3) blocked by IPC coherence check");

    // After repeated recovery, coherence improves
    NullGeodesicState rec = g_dec;
    for (int i = 0; i < 5; ++i) {
        DecoherenceLevel lv = measure_decoherence(rec.r);
        if (lv != DecoherenceLevel::NONE)
            rec = GeodesicRotator::recover(rec, 0.5, lv);
    }
    test_assert(rec.geodesic_coherence() > g_dec.geodesic_coherence(),
                "Coherence improves after repeated recovery from r=3");

    // Stochastic scenario: noise + recovery keeps IPC block rate below 50%
    GeodesicScheduler::Config ipc_cfg;
    ipc_cfg.noise_sigma   = 0.2;
    ipc_cfg.recovery_rate = 0.7;
    ipc_cfg.seed          = 123;
    GeodesicScheduler ipc_sched(ipc_cfg);
    for (int i = 0; i < 4; ++i) ipc_sched.schedule(1.0, i);

    uint32_t blocked_count = 0, total_sends = 0;
    for (int round = 0; round < 20; ++round) {
        ipc_sched.apply_noise();
        ipc_sched.recover_all();
        ipc_sched.tick();
        for (const auto& sg : ipc_sched.geodesics()) {
            ++total_sends;
            if (sg.geo.geodesic_coherence() < coherence_threshold) ++blocked_count;
        }
    }
    double block_rate = total_sends > 0 ? (double)blocked_count / total_sends : 1.0;
    test_assert(block_rate < 0.5, "Recovery keeps IPC block rate < 50%");
    std::cout << "    Block rate: " << std::fixed << std::setprecision(1)
              << (block_rate * 100.0) << "% (" << blocked_count << "/"
              << total_sends << ")\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 9 — Orbit stability under persistent noise
// ══════════════════════════════════════════════════════════════════════════════
void test_orbit_stability() {
    std::cout << "\n╔═══ Test 9: Orbit Stability Under Persistent Noise ═══╗\n";

    GeodesicScheduler::Config cfg;
    cfg.noise_sigma   = 0.03;
    cfg.recovery_rate = 0.5;
    cfg.seed          = 777;
    GeodesicScheduler sched(cfg);
    sched.schedule(1.0, 0);

    // Run 80 ticks (10 full Z/8Z orbits) with noise + recovery
    std::vector<double> ds2_vals;
    for (int t = 0; t < 80; ++t) {
        sched.apply_noise();
        sched.recover_all();
        sched.tick();
        ds2_vals.push_back(sched.geodesics()[0].geo.ds_squared());
    }

    // ds² must remain bounded (no divergence under moderate noise)
    double max_ds2 = *std::max_element(ds2_vals.begin(), ds2_vals.end());
    test_assert(max_ds2 < 0.5, "ds² remains bounded under persistent noise");

    // Mean ds² in second half (steady-state) is not much worse than first half
    double mean_first = 0.0, mean_second = 0.0;
    for (int i =  0; i < 40; ++i) mean_first  += ds2_vals[i];
    for (int i = 40; i < 80; ++i) mean_second += ds2_vals[i];
    mean_first /= 40; mean_second /= 40;
    test_assert(mean_second < mean_first * 3.0,
                "Steady-state ds² bounded (system does not diverge)");

    std::cout << "    max ds²=" << std::fixed << std::setprecision(6) << max_ds2
              << "  mean(first 40)=" << mean_first
              << "  mean(last 40)="  << mean_second << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 10 — Recovery convergence toward null geodesic
// ══════════════════════════════════════════════════════════════════════════════
void test_recovery_convergence() {
    std::cout << "\n╔═══ Test 10: Recovery Convergence Toward Null Geodesic ═══╗\n";

    // From each initial r, repeated recovery converges toward r=1
    for (double r0 : {0.5, 0.7, 1.3, 1.5, 2.0}) {
        NullGeodesicState g{r0, 0};
        for (int i = 0; i < 20; ++i) {
            DecoherenceLevel lv = measure_decoherence(g.r);
            if (lv != DecoherenceLevel::NONE)
                g = GeodesicRotator::recover(g, 0.5, lv);
        }
        test_assert(std::abs(g.r - 1.0) < std::abs(r0 - 1.0),
                    "20 recovery steps converge from r=" + std::to_string(r0));
        test_assert(g.geodesic_coherence() > coherence(r0),
                    "Coherence improves from r=" + std::to_string(r0));
    }

    // Monotone convergence: each recovery step strictly reduces |r-1|
    NullGeodesicState g_mono{1.8, 0};
    double prev_dev = std::abs(g_mono.r - 1.0);
    bool monotone = true;
    for (int i = 0; i < 10; ++i) {
        DecoherenceLevel lv = measure_decoherence(g_mono.r);
        if (lv != DecoherenceLevel::NONE)
            g_mono = GeodesicRotator::recover(g_mono, 0.5, lv);
        double new_dev = std::abs(g_mono.r - 1.0);
        if (new_dev >= prev_dev) monotone = false;
        prev_dev = new_dev;
    }
    test_assert(monotone, "Monotone convergence: each step reduces |r-1|");
}

// ══════════════════════════════════════════════════════════════════════════════
// Main test runner
// ══════════════════════════════════════════════════════════════════════════════
int main() {
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Null Geodesic Scheduler — Test Suite                ║\n";
    std::cout << "║  March 2026 Addendum: Stochastic Noise Resilience    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    test_null_metric();
    test_coherent_state_constructor();
    test_z8z_rotators();
    test_geodesic_invariants();
    test_recovery_mechanism();
    test_stochastic_noise_resilience();
    test_scheduler_tick_and_rotation();
    test_ipc_decoherence_blocking();
    test_orbit_stability();
    test_recovery_convergence();

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Test Summary                                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "  Total tests:  " << test_count << "\n";
    std::cout << "  Passed:       " << passed << " ✓\n";
    std::cout << "  Failed:       " << failed << " ✗\n";

    if (failed == 0) {
        std::cout << "\n✓ All null geodesic tests passed!\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed\n";
        return 1;
    }
}
