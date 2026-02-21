/*
 * LadderSearchBenchmark.hpp — Benchmark: Euler Kick vs No-Kick in
 * LadderChiralSearch
 *
 * Measures execution time, amplification rate, and coherence C(r) of
 * LadderChiralSearch::ladder_step with and without the Euler kick (e^y − 1)
 * active, as described in the Chiral Non-Linear Gate specification.
 *
 * Usage:
 *   #include "LadderSearchBenchmark.hpp"
 *   kernel::quantum::benchmark_kick_vs_nokick();           // single n=8
 *   kernel::quantum::benchmark_kick_vs_nokick_at_scale();  // n = 8, 16, 32,
 * 64, 128
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ChiralNonlinearGate.hpp"
#include "LadderChiralSearch.hpp"

namespace kernel::quantum {

// ── benchmark_kick_vs_nokick
// ────────────────────────────────────────────────── Compare
// LadderChiralSearch::ladder_step performance, amplification rate, and
// coherence C(r) = 2r/(1+r²) with and without the Euler kick active for a
// single register size n_states.
//
// Parameters
//   steps       — number of ladder steps per run (default: 30)
//   runs        — number of repetitions for timing statistics (default: 10)
//   n_states    — candidate-state register size passed to ladder_step (default:
//   8)
//
// Output (to stdout):
//   Average execution time, P(target), and C(r) for both configurations.
//
inline void benchmark_kick_vs_nokick(size_t steps = 30, size_t runs = 10,
                                     size_t n_states = 8) {
  std::cout << "\n  n=" << n_states << "  steps=" << steps << "  runs=" << runs
            << "\n";
  std::cout << std::fixed << std::setprecision(9);

  double total_time_no_kick = 0.0;
  double total_time_with_kick = 0.0;
  double total_prob_no_kick = 0.0;
  double total_prob_with_kick = 0.0;
  double total_coh_no_kick = 0.0;
  double total_coh_with_kick = 0.0;

  // ── Baseline: No Euler Kick (kick_base >= 1.0) ────────────────────────────
  {
    LadderChiralSearch search;
    search.target = n_states / 2;
    search.kick_base = 1.0; // linear regime — kick disabled

    for (size_t run = 0; run < runs; ++run) {
      StepResult last{};
      auto start = std::chrono::steady_clock::now();
      for (size_t s = 0; s < steps; ++s) {
        last = search.ladder_step(n_states);
      }
      auto end = std::chrono::steady_clock::now();
      total_time_no_kick += std::chrono::duration<double>(end - start).count();
      total_prob_no_kick += last.p_target;
      total_coh_no_kick += last.coherence;
    }
  }

  // ── Euler Kick Active (kick_base < 1.0) ───────────────────────────────────
  {
    LadderChiralSearch search;
    search.target = n_states / 2;
    search.kick_base = 0.5; // exponential regime — kick enabled

    for (size_t run = 0; run < runs; ++run) {
      StepResult last{};
      auto start = std::chrono::steady_clock::now();
      for (size_t s = 0; s < steps; ++s) {
        last = search.ladder_step(n_states);
      }
      auto end = std::chrono::steady_clock::now();
      total_time_with_kick +=
          std::chrono::duration<double>(end - start).count();
      total_prob_with_kick += last.p_target;
      total_coh_with_kick += last.coherence;
    }
  }

  // ── Results ───────────────────────────────────────────────────────────────
  const double dbl_runs = static_cast<double>(runs);
  const double avg_no_kick = total_time_no_kick / dbl_runs;
  const double avg_with_kick = total_time_with_kick / dbl_runs;
  const double avg_prob_no_kick = total_prob_no_kick / dbl_runs;
  const double avg_prob_with_kick = total_prob_with_kick / dbl_runs;
  const double avg_coh_no_kick = total_coh_no_kick / dbl_runs;
  const double avg_coh_with_kick = total_coh_with_kick / dbl_runs;

  std::cout << "  "
               "┌────────────────────────┬──────────────────┬──────────────────"
               "──┬────────────────────┐\n";
  std::cout << "  │ Configuration          │ Avg time/run (s) │ Avg P(target)  "
               "    │ Avg C(r) [Thm 11]  │\n";
  std::cout << "  "
               "├────────────────────────┼──────────────────┼──────────────────"
               "──┼────────────────────┤\n";
  std::cout << "  │ Without Euler's Kick   │ " << std::setw(16) << avg_no_kick
            << " │ " << std::setw(18) << avg_prob_no_kick << " │ "
            << std::setw(18) << avg_coh_no_kick << " │\n";
  std::cout << "  │ With Euler's Kick      │ " << std::setw(16) << avg_with_kick
            << " │ " << std::setw(18) << avg_prob_with_kick << " │ "
            << std::setw(18) << avg_coh_with_kick << " │\n";
  std::cout << "  "
               "└────────────────────────┴──────────────────┴──────────────────"
               "──┴────────────────────┘\n";

  if (avg_no_kick > 0.0 && avg_with_kick > 0.0) {
    const double speedup = avg_no_kick / avg_with_kick;
    const double prob_gain =
        (avg_prob_no_kick > 0.0) ? avg_prob_with_kick / avg_prob_no_kick : 0.0;
    const double coh_gain =
        (avg_coh_no_kick > 0.0) ? avg_coh_with_kick / avg_coh_no_kick : 0.0;
    std::cout << "  Speed ratio    (no_kick / with_kick): " << speedup << "x\n";
    std::cout << "  P(target) gain (with / no kick):      " << prob_gain
              << "x\n";
    std::cout << "  C(r) gain      (with / no kick):      " << coh_gain
              << "x\n";
  }
}

// ── benchmark_kick_vs_nokick_at_scale ────────────────────────────────────────
// Run benchmark_kick_vs_nokick across multiple register sizes to show how
// execution time, amplification, and coherence scale with n.
//
// Parameters
//   steps      — number of ladder steps per run (default: 30)
//   runs       — number of repetitions for timing statistics (default: 10)
//   n_values   — register sizes to sweep (default: {8, 16, 32, 64, 128, 256,
//   512, 1024, 2048, 4096})
//
inline void benchmark_kick_vs_nokick_at_scale(
    size_t steps = 30, size_t runs = 10,
    const std::vector<size_t> &n_values = {8, 16, 32, 64, 128, 256, 512, 1024,
                                           2048, 4096}) {
  std::cout
      << "\n╔═══ Benchmark: Euler Kick vs No-Kick — Scale Sweep ═════════╗\n";
  std::cout << "  steps=" << steps << "  runs=" << runs << "\n";

  for (size_t n : n_values) {
    benchmark_kick_vs_nokick(steps, runs, n);
  }

  std::cout
      << "╚══════════════════════════════════════════════════════════════╝\n";
}

} // namespace kernel::quantum
