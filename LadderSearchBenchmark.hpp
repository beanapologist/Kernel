/*
 * LadderSearchBenchmark.hpp — Benchmark: Euler Kick vs No-Kick in LadderChiralSearch
 *
 * Measures execution time and amplification rate of LadderChiralSearch::ladder_step
 * with and without the Euler kick (e^y − 1) active, as described in the Chiral
 * Non-Linear Gate specification.
 *
 * Usage:
 *   #include "LadderSearchBenchmark.hpp"
 *   kernel::quantum::benchmark_kick_vs_nokick();
 */

#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstddef>

#include "ChiralNonlinearGate.hpp"
#include "LadderChiralSearch.hpp"

namespace kernel::quantum {

// ── benchmark_kick_vs_nokick ──────────────────────────────────────────────────
// Compare LadderChiralSearch::ladder_step performance and amplification rate
// with and without the Euler kick active.
//
// Parameters
//   steps       — number of ladder steps per run (default: 30)
//   runs        — number of repetitions for timing statistics (default: 10)
//   n_states    — candidate-state register size passed to ladder_step (default: 8)
//
// Output (to stdout):
//   Average execution time per run for both configurations.
//   Average target-state probability (amplification) after all steps.
//
inline void benchmark_kick_vs_nokick(size_t steps    = 30,
                                     size_t runs     = 10,
                                     size_t n_states = 8) {
    std::cout << "\n╔═══ Benchmark: Euler Kick vs No-Kick (LadderChiralSearch) ═══╗\n";
    std::cout << "  steps=" << steps
              << "  runs=" << runs
              << "  n_states=" << n_states << "\n";
    std::cout << std::fixed << std::setprecision(9);

    double total_time_no_kick   = 0.0;
    double total_time_with_kick = 0.0;
    double total_prob_no_kick   = 0.0;
    double total_prob_with_kick = 0.0;

    // ── Baseline: No Euler Kick (kick_base >= 1.0) ────────────────────────────
    {
        LadderChiralSearch search;
        search.target    = 3;
        search.kick_base = 1.0;   // linear regime — kick disabled

        for (size_t run = 0; run < runs; ++run) {
            double last_prob = 0.0;
            auto start = std::chrono::steady_clock::now();
            for (size_t s = 0; s < steps; ++s) {
                last_prob = search.ladder_step(n_states);
            }
            auto end = std::chrono::steady_clock::now();
            total_time_no_kick += std::chrono::duration<double>(end - start).count();
            total_prob_no_kick += last_prob;
        }
    }

    // ── Euler Kick Active (kick_base < 1.0) ───────────────────────────────────
    {
        LadderChiralSearch search;
        search.target    = 3;
        search.kick_base = 0.5;   // exponential regime — kick enabled

        for (size_t run = 0; run < runs; ++run) {
            double last_prob = 0.0;
            auto start = std::chrono::steady_clock::now();
            for (size_t s = 0; s < steps; ++s) {
                last_prob = search.ladder_step(n_states);
            }
            auto end = std::chrono::steady_clock::now();
            total_time_with_kick += std::chrono::duration<double>(end - start).count();
            total_prob_with_kick += last_prob;
        }
    }

    // ── Results ───────────────────────────────────────────────────────────────
    const double avg_no_kick   = total_time_no_kick   / static_cast<double>(runs);
    const double avg_with_kick = total_time_with_kick / static_cast<double>(runs);
    const double avg_prob_no_kick   = total_prob_no_kick   / static_cast<double>(runs);
    const double avg_prob_with_kick = total_prob_with_kick / static_cast<double>(runs);

    std::cout << "\n  Benchmark Results (avg over " << runs << " runs):\n";
    std::cout << "  ┌────────────────────────┬─────────────────────┬────────────────────┐\n";
    std::cout << "  │ Configuration          │ Avg time / run (s)  │ Avg P(target)      │\n";
    std::cout << "  ├────────────────────────┼─────────────────────┼────────────────────┤\n";
    std::cout << "  │ Without Euler's Kick   │ " << std::setw(19) << avg_no_kick
              << " │ " << std::setw(18) << avg_prob_no_kick   << " │\n";
    std::cout << "  │ With Euler's Kick      │ " << std::setw(19) << avg_with_kick
              << " │ " << std::setw(18) << avg_prob_with_kick << " │\n";
    std::cout << "  └────────────────────────┴─────────────────────┴────────────────────┘\n";

    if (avg_no_kick > 0.0 && avg_with_kick > 0.0) {
        const double speedup = avg_no_kick / avg_with_kick;
        std::cout << "  Speed ratio (no_kick / with_kick): " << speedup << "x\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
}

} // namespace kernel::quantum
