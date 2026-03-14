/*
 * TimeCrystalRPC.hpp — RPC Interface for Coherence-Driven Time-Crystal Simulations
 *
 * Exposes the TimeCrystalSimulation (and any SimulationBackend) through a
 * REST-style in-process RPC layer.  Each simulation instance is identified by
 * a unique SimulationID and managed by the TimeCrystalRPCServer.
 *
 * ── Endpoint catalogue ───────────────────────────────────────────────────────
 *
 *   POST /simulations/init        → create a new simulation, return its ID
 *   POST /simulations/{id}/step   → advance one EMA step
 *   POST /simulations/{id}/feedback_step  → advance one adaptive-feedback step
 *   GET  /simulations/{id}/state  → query FloquetState snapshot
 *   POST /simulations/{id}/reset  → reset to initial conditions
 *   DELETE /simulations/{id}      → destroy the simulation
 *   GET  /simulations             → list all active simulation IDs
 *
 *   POST /simulations/{id}/stress_test → run the super-linear scaling
 *                                        stress test ("Breaking Campaign")
 *
 * ── Concurrency ──────────────────────────────────────────────────────────────
 * All methods on TimeCrystalRPCServer are guarded by a per-server mutex so
 * that concurrent callers from different threads cannot corrupt the registry.
 * Individual simulations additionally carry their own mutex so that concurrent
 * step/query calls on the same simulation are safe.
 *
 * ── Input validation ─────────────────────────────────────────────────────────
 * Every public method that accepts numeric parameters validates them before
 * touching any simulation state.  Constraint violations are reported via
 * RPCResponse::error_message and the ok flag set to false.
 *
 * ── Extensible backend ───────────────────────────────────────────────────────
 * The server accepts any unique_ptr<SimulationBackend>, enabling drop-in
 * replacement of the simulation engine without changing the RPC layer.
 *
 * ── Stress-test / Breaking Campaign ─────────────────────────────────────────
 * stress_test() deliberately explores super-linear coherence regimes by
 * running N_wave concurrent "waves" of feedback_step() calls with
 * aggressively increasing alpha values, measuring wall-clock throughput
 * and coherence collapse.  The results surface simultaneity-breaking
 * behaviour in coherence-driven ensembles.
 */

#pragma once

#include "TimeCrystalSimulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace kernel::rpc {

// ── Constants ─────────────────────────────────────────────────────────────────
/// Threshold factor for super-linear throughput-degradation detection in the
/// Breaking Campaign stress test.  Throughput is considered super-linear when
/// the last wave falls below STRESS_SUPERLINEAR_THRESHOLD of the first wave.
static constexpr double STRESS_SUPERLINEAR_THRESHOLD = 0.90;

// ── Type aliases ──────────────────────────────────────────────────────────────
using SimulationID = std::uint64_t;

// ── RPCStatus ─────────────────────────────────────────────────────────────────
enum class RPCStatus : int {
    OK               = 200,
    CREATED          = 201,
    BAD_REQUEST      = 400,
    NOT_FOUND        = 404,
    INTERNAL_ERROR   = 500,
};

// ── RPCResponse ───────────────────────────────────────────────────────────────
/// Uniform response envelope returned by every server method.
template <typename T>
struct RPCResponse {
    bool        ok            = false;
    RPCStatus   status        = RPCStatus::INTERNAL_ERROR;
    std::string error_message;
    T           data{};

    static RPCResponse success(T value, RPCStatus s = RPCStatus::OK) {
        return RPCResponse{true, s, "", std::move(value)};
    }
    static RPCResponse error(RPCStatus s, std::string msg) {
        return RPCResponse{false, s, std::move(msg), {}};
    }
};

// Specialisation for void responses.
template <>
struct RPCResponse<void> {
    bool        ok            = false;
    RPCStatus   status        = RPCStatus::INTERNAL_ERROR;
    std::string error_message;

    static RPCResponse success(RPCStatus s = RPCStatus::OK) {
        return RPCResponse{true, s, ""};
    }
    static RPCResponse error(RPCStatus s, std::string msg) {
        return RPCResponse{false, s, std::move(msg)};
    }
};

// ── StressTestResult ──────────────────────────────────────────────────────────
/// Summary returned by the Breaking-Campaign stress test.
struct StressTestResult {
    std::size_t waves_run;           ///< Number of stress waves executed
    std::size_t steps_per_wave;      ///< Steps executed per wave
    double      peak_R;              ///< Highest coherence observed
    double      min_R;               ///< Lowest coherence observed (collapse)
    double      peak_throughput_kHz; ///< Peak step throughput (kilo-steps/s)
    double      coherence_collapse;  ///< |peak_R − min_R| — collapse amplitude
    bool        super_linear_detected; ///< True if throughput degrades super-linearly
    std::string notes;               ///< Human-readable summary
};

// ── SimulationEntry ───────────────────────────────────────────────────────────
/// Per-simulation registry entry (backend + its own mutex).
struct SimulationEntry {
    std::unique_ptr<tc::SimulationBackend> backend;
    mutable std::mutex                     sim_mutex;
    tc::SimulationConfig                   config; ///< Config snapshot for introspection

    SimulationEntry(std::unique_ptr<tc::SimulationBackend> b,
                    tc::SimulationConfig cfg)
        : backend(std::move(b)), config(std::move(cfg)) {}
};

// ── TimeCrystalRPCServer ──────────────────────────────────────────────────────
/// Thread-safe registry and dispatcher for time-crystal simulation RPCs.
class TimeCrystalRPCServer {
public:
    TimeCrystalRPCServer() : next_id_(1) {}

    // ── POST /simulations/init ─────────────────────────────────────────────
    /// Create a new simulation with the given configuration.
    /// Returns the new simulation's ID.
    RPCResponse<SimulationID>
    init(const tc::SimulationConfig &cfg) {
        // Validate before creating anything.
        try {
            cfg.validate();
        } catch (const std::exception &ex) {
            return RPCResponse<SimulationID>::error(RPCStatus::BAD_REQUEST,
                                                    ex.what());
        }

        auto backend = std::make_unique<tc::TimeCrystalSimulation>(cfg);
        SimulationID id;
        {
            std::lock_guard<std::mutex> lk(registry_mutex_);
            id = next_id_++;
            registry_.emplace(
                id,
                std::make_unique<SimulationEntry>(std::move(backend), cfg));
        }
        return RPCResponse<SimulationID>::success(id, RPCStatus::CREATED);
    }

    /// Overload accepting a custom backend (for pluggable backends).
    RPCResponse<SimulationID>
    init_with_backend(std::unique_ptr<tc::SimulationBackend> backend,
                      const tc::SimulationConfig &cfg) {
        if (!backend)
            return RPCResponse<SimulationID>::error(
                RPCStatus::BAD_REQUEST, "backend must not be null");
        try {
            cfg.validate();
        } catch (const std::exception &ex) {
            return RPCResponse<SimulationID>::error(RPCStatus::BAD_REQUEST,
                                                    ex.what());
        }

        SimulationID id;
        {
            std::lock_guard<std::mutex> lk(registry_mutex_);
            id = next_id_++;
            registry_.emplace(
                id,
                std::make_unique<SimulationEntry>(std::move(backend), cfg));
        }
        return RPCResponse<SimulationID>::success(id, RPCStatus::CREATED);
    }

    // ── POST /simulations/{id}/step ────────────────────────────────────────
    /// Advance one plain EMA step.
    /// Returns the frustration released this step.
    RPCResponse<double> step(SimulationID id) {
        return with_sim(id, [](SimulationEntry &e) -> RPCResponse<double> {
            std::lock_guard<std::mutex> lk(e.sim_mutex);
            double released = e.backend->step();
            return RPCResponse<double>::success(released);
        });
    }

    // ── POST /simulations/{id}/feedback_step ───────────────────────────────
    /// Advance one adaptive-feedback step.
    /// alpha must be in [0, 1].
    RPCResponse<double> feedback_step(SimulationID id, double alpha = 1.0) {
        if (alpha < 0.0 || alpha > 1.0 + tc::VALIDATION_EPSILON)
            return RPCResponse<double>::error(
                RPCStatus::BAD_REQUEST,
                "alpha must be in [0, 1]; got " + std::to_string(alpha));

        return with_sim(id, [alpha](SimulationEntry &e) -> RPCResponse<double> {
            std::lock_guard<std::mutex> lk(e.sim_mutex);
            double released = e.backend->feedback_step(alpha);
            return RPCResponse<double>::success(released);
        });
    }

    // ── GET /simulations/{id}/state ────────────────────────────────────────
    /// Query the current FloquetState snapshot.
    RPCResponse<tc::FloquetState> query_state(SimulationID id) {
        return with_sim(
            id, [](const SimulationEntry &e) -> RPCResponse<tc::FloquetState> {
                std::lock_guard<std::mutex> lk(e.sim_mutex);
                return RPCResponse<tc::FloquetState>::success(e.backend->query());
            });
    }

    // ── POST /simulations/{id}/reset ───────────────────────────────────────
    RPCResponse<void> reset(SimulationID id) {
        return with_sim(id, [](SimulationEntry &e) -> RPCResponse<void> {
            std::lock_guard<std::mutex> lk(e.sim_mutex);
            e.backend->reset();
            return RPCResponse<void>::success();
        });
    }

    // ── DELETE /simulations/{id} ───────────────────────────────────────────
    RPCResponse<void> destroy(SimulationID id) {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        auto it = registry_.find(id);
        if (it == registry_.end())
            return RPCResponse<void>::error(RPCStatus::NOT_FOUND,
                                            sim_not_found_msg(id));
        registry_.erase(it);
        return RPCResponse<void>::success();
    }

    // ── GET /simulations ───────────────────────────────────────────────────
    /// Return a sorted list of all currently active simulation IDs.
    RPCResponse<std::vector<SimulationID>> list_simulations() {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        std::vector<SimulationID> ids;
        ids.reserve(registry_.size());
        for (const auto &kv : registry_)
            ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        return RPCResponse<std::vector<SimulationID>>::success(std::move(ids));
    }

    // ── POST /simulations/{id}/stress_test — Breaking Campaign ────────────
    /// Super-linear scaling stress test.
    ///
    /// Runs N_wave waves of N_steps_per_wave feedback steps against the
    /// simulation identified by `id`.  Each wave increases alpha toward 1.0
    /// (maximising coherence amplification), then measures throughput and
    /// coherence to detect super-linear collapse.
    ///
    /// @param id              Simulation to stress-test.
    /// @param N_wave          Number of stress waves (default 8).
    /// @param N_steps_per_wave Steps per wave (default 64).
    RPCResponse<StressTestResult>
    stress_test(SimulationID id,
                std::size_t N_wave          = 8,
                std::size_t N_steps_per_wave = 64) {
        if (N_wave == 0 || N_steps_per_wave == 0)
            return RPCResponse<StressTestResult>::error(
                RPCStatus::BAD_REQUEST,
                "N_wave and N_steps_per_wave must be > 0");

        return with_sim(id, [N_wave, N_steps_per_wave](
                                SimulationEntry &e) -> RPCResponse<StressTestResult> {
            std::lock_guard<std::mutex> lk(e.sim_mutex);

            e.backend->reset(); // Start from clean initial conditions.

            StressTestResult result{};
            result.waves_run        = N_wave;
            result.steps_per_wave   = N_steps_per_wave;
            result.peak_R           = 0.0;
            result.min_R            = 1.0;
            result.peak_throughput_kHz = 0.0;

            std::vector<double> throughputs;
            throughputs.reserve(N_wave);

            for (std::size_t w = 0; w < N_wave; ++w) {
                // Ramp alpha super-linearly: α_w = (w+1)² / N_wave²
                double alpha_w = static_cast<double>((w + 1) * (w + 1)) /
                                 static_cast<double>(N_wave * N_wave);
                if (alpha_w > 1.0)
                    alpha_w = 1.0;

                auto t0 = std::chrono::steady_clock::now();
                for (std::size_t s = 0; s < N_steps_per_wave; ++s) {
                    e.backend->feedback_step(alpha_w);
                }
                auto t1 = std::chrono::steady_clock::now();

                double elapsed_us = std::chrono::duration<double, std::micro>(
                                        t1 - t0)
                                        .count();
                double steps_kHz = (elapsed_us > 0.0)
                                   ? static_cast<double>(N_steps_per_wave) /
                                         elapsed_us * 1000.0
                                   : 0.0;
                throughputs.push_back(steps_kHz);
                if (steps_kHz > result.peak_throughput_kHz)
                    result.peak_throughput_kHz = steps_kHz;

                tc::FloquetState snap = e.backend->query();
                if (snap.R > result.peak_R)
                    result.peak_R = snap.R;
                if (snap.R < result.min_R)
                    result.min_R = snap.R;
            }

            result.coherence_collapse = result.peak_R - result.min_R;

            // Super-linear detection: throughput should *fall* as coherence
            // rises (super-linear communication cost).  We flag it when the
            // last-wave throughput is < 90 % of the first-wave throughput.
            result.super_linear_detected =
                (throughputs.size() >= 2) &&
                (throughputs.back() < STRESS_SUPERLINEAR_THRESHOLD * throughputs.front());

            std::ostringstream notes;
            notes << "Stress test (" << N_wave << " waves × "
                  << N_steps_per_wave << " steps): "
                  << "peak_R=" << result.peak_R
                  << " min_R=" << result.min_R
                  << " collapse=" << result.coherence_collapse
                  << " throughput_kHz=" << result.peak_throughput_kHz
                  << (result.super_linear_detected
                          ? " [SUPER-LINEAR DETECTED]"
                          : " [linear/sub-linear]");
            result.notes = notes.str();

            return RPCResponse<StressTestResult>::success(result);
        });
    }

    // ── Introspection ──────────────────────────────────────────────────────
    /// Return the config used to create simulation `id`.
    RPCResponse<tc::SimulationConfig> get_config(SimulationID id) {
        return with_sim(
            id, [](const SimulationEntry &e) -> RPCResponse<tc::SimulationConfig> {
                return RPCResponse<tc::SimulationConfig>::success(e.config);
            });
    }

    /// Return the backend name for simulation `id`.
    RPCResponse<std::string> get_backend_name(SimulationID id) {
        return with_sim(
            id, [](const SimulationEntry &e) -> RPCResponse<std::string> {
                std::lock_guard<std::mutex> lk(e.sim_mutex);
                return RPCResponse<std::string>::success(
                    e.backend->backend_name());
            });
    }

    /// Number of currently active simulations.
    std::size_t active_count() const {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        return registry_.size();
    }

private:
    mutable std::mutex registry_mutex_;
    std::unordered_map<SimulationID, std::unique_ptr<SimulationEntry>> registry_;
    SimulationID next_id_;

    // ── Internal helpers ───────────────────────────────────────────────────

    static std::string sim_not_found_msg(SimulationID id) {
        return "Simulation " + std::to_string(id) + " not found";
    }

    /// Look up `id` in the registry and invoke `fn` on the SimulationEntry.
    /// Returns RPCStatus::NOT_FOUND if the ID is unknown.
    template <typename Fn>
    auto with_sim(SimulationID id, Fn &&fn)
        -> decltype(fn(*registry_.begin()->second)) {
        SimulationEntry *entry = nullptr;
        {
            std::lock_guard<std::mutex> lk(registry_mutex_);
            auto it = registry_.find(id);
            if (it == registry_.end()) {
                using R = decltype(fn(*registry_.begin()->second));
                return R::error(RPCStatus::NOT_FOUND,
                                sim_not_found_msg(id));
            }
            entry = it->second.get();
        }
        return fn(*entry);
    }
};

} // namespace kernel::rpc
