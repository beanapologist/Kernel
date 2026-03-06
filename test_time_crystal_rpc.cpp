/*
 * test_time_crystal_rpc.cpp — Test Suite for TimeCrystalRPC
 *
 * Validates all layers of the RPC interface:
 *
 *   1. TimeCrystalSimulation — backend: T_eff, ε_F, Floquet evolution,
 *      constraint validation, feedback_step, reset.
 *   2. RPCResponse — status codes, ok/error semantics.
 *   3. TimeCrystalRPCServer — init, step, feedback_step, query_state,
 *      reset, destroy, list_simulations, get_config, get_backend_name.
 *   4. Input validation — bad R, E, T, g, alpha are rejected.
 *   5. Not-found errors — unknown IDs return 404.
 *   6. Concurrency — multiple simulations run independently without corruption.
 *   7. Stress test — Breaking-Campaign endpoint runs and returns coherent output.
 *   8. Pluggable backend — a custom SimulationBackend is accepted via
 *      init_with_backend().
 *
 * Test style matches test_ohm_coherence.cpp and test_battery_analogy.cpp.
 */

#include "TimeCrystalRPC.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace kernel::tc;
using namespace kernel::rpc;

// ── Test framework ────────────────────────────────────────────────────────────
static int test_count = 0;
static int passed     = 0;
static int failed     = 0;

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

static constexpr double TOL = 1e-9;
static constexpr double TEST_PI = 3.14159265358979323846;

// ── 1. TimeCrystalSimulation backend ─────────────────────────────────────────
void test_simulation_backend() {
    std::cout << "\n\u2554\u2550\u2550\u2550 1. TimeCrystalSimulation backend \u2550\u2550\u2550\u2557\n";

    SimulationConfig cfg;
    cfg.N      = 8;
    cfg.R_init = 0.8;
    cfg.E_init = 0.1;
    cfg.T_base = 1.0;
    cfg.g      = 0.3;
    cfg.alpha  = 1.0;

    TimeCrystalSimulation sim(cfg);
    FloquetState s0 = sim.query();

    test_assert(s0.step_count == 0, "Initial step_count = 0");
    test_assert(s0.psi_abs() > 0.0, "Initial |ψ| > 0");

    // Floquet quasi-energy round-trip: ε_F * T_eff = π
    double lhs = s0.epsilon_F * s0.T_eff;
    test_assert(std::abs(lhs - TEST_PI) < TOL, "ε_F * T_eff = π (quasi-energy)");

    // T_eff = T_base / R
    double expected_T_eff = cfg.T_base / s0.R;
    test_assert(std::abs(s0.T_eff - expected_T_eff) < TOL,
                "T_eff = T_base / R");

    // One step: ψ should flip sign (Floquet phase π)
    std::complex<double> psi_before = s0.psi;
    sim.step();
    FloquetState s1 = sim.query();
    test_assert(s1.step_count == 1, "step_count increments to 1");
    // sign flip: s1.psi ≈ −psi_before  (exact for fixed-point evolution)
    std::complex<double> diff = s1.psi + psi_before; // should be ~0
    test_assert(std::abs(diff) < TOL,
                "ψ flips sign after one Floquet step (exp(−iπ) = −1)");

    // Two steps: ψ returns to original (period doubling ψ(t+2T)=ψ(t))
    sim.step();
    FloquetState s2 = sim.query();
    test_assert(s2.step_count == 2, "step_count = 2");
    std::complex<double> diff2 = s2.psi - psi_before;
    test_assert(std::abs(diff2) < TOL,
                "ψ(t+2T) = ψ(t)  (period doubling, 2 Floquet steps)");

    // |ψ| is invariant after Floquet evolution (norm preserved)
    test_assert(std::abs(s2.psi_abs() - s0.psi_abs()) < TOL,
                "|ψ| is norm-invariant across Floquet steps");

    // Coherence non-decreasing after feedback steps
    sim.reset();
    double R_prev = sim.query().R;
    bool R_non_decreasing = true;
    for (int i = 0; i < 20; ++i) {
        sim.feedback_step(1.0);
        double R_now = sim.query().R;
        if (R_now < R_prev - TOL)
            R_non_decreasing = false;
        R_prev = R_now;
    }
    test_assert(R_non_decreasing,
                "Coherence R non-decreasing over feedback steps (g ∈ (0,1])");

    // Reset restores initial state
    sim.reset();
    FloquetState sr = sim.query();
    test_assert(sr.step_count == 0, "reset() clears step_count");
    test_assert(std::abs(sr.psi - std::complex<double>(1.0, 0.0)) < TOL,
                "reset() restores ψ = 1 + 0i");

    // backend_name
    test_assert(sim.backend_name() == "TimeCrystalSimulation",
                "backend_name() == \"TimeCrystalSimulation\"");
}

// ── 2. Constraint validation (TimeCrystalSimulation) ─────────────────────────
void test_simulation_constraints() {
    std::cout << "\n\u2554\u2550\u2550\u2550 2. Constraint validation \u2550\u2550\u2550\u2557\n";

    // Bad R_init = 0 (not in (0,1])
    {
        SimulationConfig bad_cfg;
        bad_cfg.R_init = 0.0;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "R_init = 0 throws domain_error");
    }
    // Bad R_init > 1
    {
        SimulationConfig bad_cfg;
        bad_cfg.R_init = 1.5;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "R_init = 1.5 throws domain_error");
    }
    // Bad T_base ≤ 0
    {
        SimulationConfig bad_cfg;
        bad_cfg.T_base = 0.0;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "T_base = 0 throws domain_error");
    }
    // Bad T_base negative
    {
        SimulationConfig bad_cfg;
        bad_cfg.T_base = -1.0;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "T_base = -1 throws domain_error");
    }
    // Bad E_init < 0
    {
        SimulationConfig bad_cfg;
        bad_cfg.E_init = -0.1;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "E_init = -0.1 throws domain_error");
    }
    // Bad g > 1
    {
        SimulationConfig bad_cfg;
        bad_cfg.g = 1.5;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "g = 1.5 throws domain_error");
    }
    // Bad N < 1
    {
        SimulationConfig bad_cfg;
        bad_cfg.N = 0;
        bool threw = false;
        try { bad_cfg.validate(); } catch (const std::domain_error &) { threw = true; }
        test_assert(threw, "N = 0 throws domain_error");
    }
    // Valid boundary values pass
    {
        SimulationConfig ok_cfg;
        ok_cfg.R_init = 1.0;
        ok_cfg.T_base = 1e-6;
        ok_cfg.g      = 1.0;
        ok_cfg.N      = 1;
        bool threw = false;
        try { ok_cfg.validate(); } catch (...) { threw = true; }
        test_assert(!threw, "Boundary-valid config does not throw");
    }
}

// ── 3. RPCServer basic lifecycle ──────────────────────────────────────────────
void test_rpc_lifecycle() {
    std::cout << "\n\u2554\u2550\u2550\u2550 3. RPCServer lifecycle \u2550\u2550\u2550\u2557\n";

    TimeCrystalRPCServer srv;

    SimulationConfig cfg;
    cfg.N = 4;  cfg.R_init = 0.7;  cfg.T_base = 2.0;  cfg.g = 0.4;

    // init
    auto r_init = srv.init(cfg);
    test_assert(r_init.ok, "init() returns ok");
    test_assert(r_init.status == RPCStatus::CREATED, "init() status = CREATED");
    SimulationID id = r_init.data;
    test_assert(id >= 1, "init() returns non-zero ID");
    test_assert(srv.active_count() == 1, "active_count = 1 after init");

    // list
    auto r_list = srv.list_simulations();
    test_assert(r_list.ok && r_list.data.size() == 1, "list returns 1 ID");
    test_assert(r_list.data[0] == id, "listed ID matches returned ID");

    // get_config
    auto r_cfg = srv.get_config(id);
    test_assert(r_cfg.ok, "get_config() ok");
    test_assert(r_cfg.data.N == 4, "get_config returns correct N");

    // get_backend_name
    auto r_name = srv.get_backend_name(id);
    test_assert(r_name.ok, "get_backend_name() ok");
    test_assert(r_name.data == "TimeCrystalSimulation",
                "backend name is TimeCrystalSimulation");

    // step
    auto r_step = srv.step(id);
    test_assert(r_step.ok, "step() ok");
    test_assert(r_step.data >= 0.0, "step() released frustration ≥ 0");

    // feedback_step
    auto r_fb = srv.feedback_step(id, 0.8);
    test_assert(r_fb.ok, "feedback_step() ok");

    // query_state
    auto r_qs = srv.query_state(id);
    test_assert(r_qs.ok, "query_state() ok");
    test_assert(r_qs.data.step_count == 2, "step_count = 2 after step+feedback");
    // ε_F * T_eff = π
    double quasi = r_qs.data.epsilon_F * r_qs.data.T_eff;
    test_assert(std::abs(quasi - TEST_PI) < 1e-6,
                "query: ε_F * T_eff = π");

    // reset
    auto r_reset = srv.reset(id);
    test_assert(r_reset.ok, "reset() ok");
    auto r_qs2 = srv.query_state(id);
    test_assert(r_qs2.ok && r_qs2.data.step_count == 0,
                "reset() clears step_count via RPC");

    // destroy
    auto r_del = srv.destroy(id);
    test_assert(r_del.ok, "destroy() ok");
    test_assert(srv.active_count() == 0, "active_count = 0 after destroy");

    // Operations on destroyed ID return 404
    auto r_miss = srv.step(id);
    test_assert(!r_miss.ok && r_miss.status == RPCStatus::NOT_FOUND,
                "step() on destroyed ID returns NOT_FOUND");
}

// ── 4. RPC input validation ────────────────────────────────────────────────────
void test_rpc_validation() {
    std::cout << "\n\u2554\u2550\u2550\u2550 4. RPC input validation \u2550\u2550\u2550\u2557\n";

    TimeCrystalRPCServer srv;

    // Bad config rejected by init()
    SimulationConfig bad;
    bad.R_init = 0.0; // invalid
    auto r = srv.init(bad);
    test_assert(!r.ok && r.status == RPCStatus::BAD_REQUEST,
                "init() with R_init=0 returns BAD_REQUEST");
    test_assert(srv.active_count() == 0, "no sim created on bad init");

    // Unknown ID → NOT_FOUND
    auto r2 = srv.query_state(9999);
    test_assert(!r2.ok && r2.status == RPCStatus::NOT_FOUND,
                "query unknown ID returns NOT_FOUND");

    // alpha out of range
    SimulationConfig ok_cfg;
    auto r_init = srv.init(ok_cfg);
    SimulationID id = r_init.data;

    auto r_alpha = srv.feedback_step(id, -0.1);
    test_assert(!r_alpha.ok && r_alpha.status == RPCStatus::BAD_REQUEST,
                "feedback_step() with alpha=-0.1 returns BAD_REQUEST");

    auto r_alpha2 = srv.feedback_step(id, 1.5);
    test_assert(!r_alpha2.ok && r_alpha2.status == RPCStatus::BAD_REQUEST,
                "feedback_step() with alpha=1.5 returns BAD_REQUEST");

    // stress_test with 0 waves → BAD_REQUEST
    auto r_st = srv.stress_test(id, 0, 10);
    test_assert(!r_st.ok && r_st.status == RPCStatus::BAD_REQUEST,
                "stress_test() with N_wave=0 returns BAD_REQUEST");

    srv.destroy(id);
}

// ── 5. Concurrent simulations ─────────────────────────────────────────────────
void test_concurrency() {
    std::cout << "\n\u2554\u2550\u2550\u2550 5. Concurrent simulations \u2550\u2550\u2550\u2557\n";

    TimeCrystalRPCServer srv;

    // Create several simulations with different configs.
    const int N_SIMS = 4;
    std::vector<SimulationID> ids(N_SIMS);
    for (int i = 0; i < N_SIMS; ++i) {
        SimulationConfig c;
        c.N      = 4 + i;
        c.R_init = 0.5 + 0.1 * i;
        c.T_base = 1.0 + i * 0.5;
        c.g      = 0.2 + 0.1 * i;
        ids[i] = srv.init(c).data;
    }
    test_assert(srv.active_count() == N_SIMS,
                "All simulations registered");

    // Run steps concurrently from multiple threads.
    std::vector<std::thread> threads;
    threads.reserve(N_SIMS);
    for (int i = 0; i < N_SIMS; ++i) {
        threads.emplace_back([&srv, id = ids[i]]() {
            for (int s = 0; s < 50; ++s)
                srv.feedback_step(id, 0.9);
        });
    }
    for (auto &t : threads)
        t.join();

    // Each simulation advanced independently.
    bool all_stepped = true;
    for (int i = 0; i < N_SIMS; ++i) {
        auto qs = srv.query_state(ids[i]);
        if (!qs.ok || qs.data.step_count != 50)
            all_stepped = false;
    }
    test_assert(all_stepped,
                "Each of 4 concurrent simulations ran 50 steps independently");

    // Verify all simulations survived concurrent stepping without crashing and
    // that their state is accessible (queries succeed).
    bool all_queryable = true;
    for (int i = 0; i < N_SIMS; ++i) {
        auto qs = srv.query_state(ids[i]);
        if (!qs.ok)
            all_queryable = false;
    }
    test_assert(all_queryable,
                "All concurrent simulations survived 50 parallel steps without crash");

    for (auto id : ids)
        srv.destroy(id);
    test_assert(srv.active_count() == 0,
                "All concurrent simulations cleanly destroyed");
}

// ── 6. Stress test (Breaking Campaign) ───────────────────────────────────────
void test_stress() {
    std::cout << "\n\u2554\u2550\u2550\u2550 6. Stress test (Breaking Campaign) \u2550\u2550\u2550\u2557\n";

    TimeCrystalRPCServer srv;
    SimulationConfig cfg;
    cfg.N = 16;  cfg.g = 0.4;  cfg.R_init = 0.6;  cfg.T_base = 1.0;
    auto r_init = srv.init(cfg);
    test_assert(r_init.ok, "Stress test sim initialised");
    SimulationID id = r_init.data;

    auto r_st = srv.stress_test(id, 8, 32);
    test_assert(r_st.ok, "stress_test() completes without error");
    const StressTestResult &st = r_st.data;

    test_assert(st.waves_run == 8, "stress_test reports correct wave count");
    test_assert(st.steps_per_wave == 32,
                "stress_test reports correct steps_per_wave");
    test_assert(st.peak_R >= st.min_R,
                "peak_R >= min_R (coherence collapse well-ordered)");
    test_assert(st.coherence_collapse >= 0.0,
                "coherence_collapse ≥ 0");
    test_assert(st.peak_throughput_kHz >= 0.0,
                "peak_throughput_kHz ≥ 0");
    test_assert(!st.notes.empty(), "stress_test notes non-empty");
    std::cout << "    Notes: " << st.notes << "\n";

    srv.destroy(id);
}

// ── 7. Multiple concurrent init/destroy cycles ────────────────────────────────
void test_concurrent_registry() {
    std::cout << "\n\u2554\u2550\u2550\u2550 7. Concurrent init/destroy \u2550\u2550\u2550\u2557\n";

    TimeCrystalRPCServer srv;
    const int N_THREADS = 8;
    const int N_OPS     = 20;

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&srv]() {
            for (int i = 0; i < N_OPS; ++i) {
                SimulationConfig c;
                c.N = 4;  c.R_init = 0.7;
                auto r = srv.init(c);
                if (r.ok) {
                    srv.step(r.data);
                    srv.destroy(r.data);
                }
            }
        });
    }
    for (auto &th : threads)
        th.join();

    // After all threads finish, registry must be empty.
    test_assert(srv.active_count() == 0,
                "Concurrent init/destroy leaves empty registry");
}

// ── 8. Pluggable backend ───────────────────────────────────────────────────────
void test_pluggable_backend() {
    std::cout << "\n\u2554\u2550\u2550\u2550 8. Pluggable backend \u2550\u2550\u2550\u2557\n";

    // A trivial custom backend that always returns a fixed FloquetState.
    class FixedBackend : public SimulationBackend {
    public:
        double step() override { ++steps_; return 0.0; }
        double feedback_step(double /*alpha*/) override { ++steps_; return 0.0; }
        FloquetState query() const override {
            FloquetState fs;
            fs.psi        = std::complex<double>(0.5, 0.5);
            fs.T_eff      = 2.0;
            fs.epsilon_F  = TEST_PI / 2.0;
            fs.R          = 0.5;
            fs.E          = 0.1;
            fs.step_count = steps_;
            return fs;
        }
        void reset() override { steps_ = 0; }
        std::string backend_name() const override { return "FixedBackend"; }
    private:
        std::size_t steps_ = 0;
    };

    TimeCrystalRPCServer srv;
    SimulationConfig cfg; // config is informational only for custom backends
    auto r_init = srv.init_with_backend(
        std::make_unique<FixedBackend>(), cfg);
    test_assert(r_init.ok, "init_with_backend() succeeds");
    SimulationID id = r_init.data;

    auto r_name = srv.get_backend_name(id);
    test_assert(r_name.ok && r_name.data == "FixedBackend",
                "Custom backend name reported correctly");

    srv.step(id);
    auto r_qs = srv.query_state(id);
    test_assert(r_qs.ok && r_qs.data.step_count == 1,
                "FixedBackend::step() counted correctly");

    // Null backend rejected
    auto r_null = srv.init_with_backend(nullptr, cfg);
    test_assert(!r_null.ok && r_null.status == RPCStatus::BAD_REQUEST,
                "null backend rejected with BAD_REQUEST");

    srv.destroy(id);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║   TimeCrystalRPC — Test Suite                           ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    test_simulation_backend();
    test_simulation_constraints();
    test_rpc_lifecycle();
    test_rpc_validation();
    test_concurrency();
    test_stress();
    test_concurrent_registry();
    test_pluggable_backend();

    std::cout << "\n────────────────────────────────────────────────────────\n";
    std::cout << "  Results: " << passed << " / " << test_count << " passed";
    if (failed > 0)
        std::cout << "  (" << failed << " FAILED)";
    std::cout << "\n";

    return failed == 0 ? 0 : 1;
}
