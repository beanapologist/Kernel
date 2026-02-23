#!/usr/bin/env python3
"""
Global Parameter Sweep Analysis
Hierarchical aggregation across 8 agents
"""

import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_agent_results(agent_id):
    """Load CSV results for one agent"""
    csv_path = Path(f"/tmp/coherent_mining/agent_{agent_id}_sweep.csv")
    results = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'agent_id': int(row['agent_id']),
                'epsilon': float(row['epsilon']),
                'recovery_rate': float(row['recovery_rate']),
                'kick_strength': float(row['kick_strength']),
                'alpha': float(row['alpha']),
                'coherence': float(row['coherence']),
                'lambda': float(row['lambda']),
                'G_eff': float(row['G_eff']),
                'best_zeros': int(row['best_zeros']),
                'delta_alpha': float(row['delta_alpha']),
                'runaway': row['runaway'] == 'True',
                'renormalized': row['renormalized'] == 'True'
            })
    
    return results

def aggregate_max_alpha(all_results):
    """Aggregate by taking max α at each parameter point"""
    grid = {}
    
    for r in all_results:
        key = (round(r['epsilon'], 2), r['recovery_rate'], r['kick_strength'])
        
        if key not in grid:
            grid[key] = r
        elif r['alpha'] > grid[key]['alpha']:
            grid[key] = r
    
    return grid

def find_epsilon_star(grid):
    """Find optimal noise level from global grid"""
    # Group by epsilon, find max delta_alpha
    epsilon_deltas = defaultdict(list)
    
    for params, result in grid.items():
        epsilon, recovery, kick = params
        epsilon_deltas[epsilon].append(result['delta_alpha'])
    
    # Find epsilon with highest average Δα (sharpest transitions)
    epsilon_scores = {
        eps: np.mean(deltas) 
        for eps, deltas in epsilon_deltas.items()
    }
    
    epsilon_star = max(epsilon_scores.items(), key=lambda x: x[1])
    return epsilon_star

def find_phase_transitions(all_results):
    """Detect phase transition boundaries per region"""
    transitions = {}
    
    for agent_id in range(8):
        agent_results = [r for r in all_results if r['agent_id'] == agent_id]
        
        # Find points with highest Δα
        max_delta = max(r['delta_alpha'] for r in agent_results)
        transition_points = [
            r for r in agent_results 
            if r['delta_alpha'] > max_delta * 0.8
        ]
        
        transitions[agent_id] = {
            'count': len(transition_points),
            'max_delta_alpha': max_delta,
            'epsilon_range': [
                min(r['epsilon'] for r in transition_points),
                max(r['epsilon'] for r in transition_points)
            ] if transition_points else [0, 0]
        }
    
    return transitions

def cluster_by_peak_alpha(all_results):
    """Cluster agents by similarity of peak α location"""
    agent_peaks = {}
    
    for agent_id in range(8):
        agent_results = [r for r in all_results if r['agent_id'] == agent_id]
        peak = max(agent_results, key=lambda r: r['alpha'])
        
        agent_peaks[agent_id] = {
            'alpha': peak['alpha'],
            'epsilon': peak['epsilon'],
            'recovery': peak['recovery_rate'],
            'kick': peak['kick_strength'],
            'coherence': peak['coherence'],
            'best_zeros': peak['best_zeros']
        }
    
    return agent_peaks

def analyze_coherence_patterns(all_results):
    """Analyze coherence vs performance"""
    # Bin by coherence level
    coherence_bins = defaultdict(list)
    
    for r in all_results:
        c_bin = round(r['coherence'], 1)
        coherence_bins[c_bin].append(r['best_zeros'])
    
    coherence_performance = {
        c: {
            'avg_zeros': np.mean(zeros),
            'max_zeros': max(zeros),
            'count': len(zeros)
        }
        for c, zeros in coherence_bins.items()
    }
    
    return coherence_performance

# Main analysis
print("="*70)
print("GLOBAL PARAMETER SWEEP ANALYSIS")
print("="*70)
print("\nLoading results from 8 agents...\n")

all_results = []
for agent_id in range(8):
    results = load_agent_results(agent_id)
    all_results.extend(results)
    print(f"  Agent {agent_id}: {len(results)} sweep points")

print(f"\nTotal data points: {len(all_results)}")

# 1. Global grid aggregation
print("\n" + "="*70)
print("1. GLOBAL GRID (max α aggregation)")
print("="*70)

global_grid = aggregate_max_alpha(all_results)
print(f"Unique parameter combinations: {len(global_grid)}")

max_alpha_point = max(global_grid.values(), key=lambda r: r['alpha'])
print(f"\nGlobal α_max: {max_alpha_point['alpha']:.4f}")
print(f"  Found by: Agent {max_alpha_point['agent_id']}")
print(f"  Parameters: ε={max_alpha_point['epsilon']:.2f}, "
      f"recovery={max_alpha_point['recovery_rate']:.2f}, "
      f"kick={max_alpha_point['kick_strength']:.2f}")
print(f"  Coherence: {max_alpha_point['coherence']:.4f}")

# 2. Find global ε*
print("\n" + "="*70)
print("2. OPTIMAL NOISE LEVEL (ε*)")
print("="*70)

epsilon_star, avg_delta = find_epsilon_star(global_grid)
print(f"Global ε* = {epsilon_star:.4f}")
print(f"Avg Δα at ε*: {avg_delta:.6f}")

# 3. Phase transitions
print("\n" + "="*70)
print("3. PHASE TRANSITION BOUNDARIES")
print("="*70)

transitions = find_phase_transitions(all_results)
for agent_id, t in transitions.items():
    print(f"Agent {agent_id}: {t['count']} transition points, "
          f"max Δα={t['max_delta_alpha']:.4f}, "
          f"ε ∈ [{t['epsilon_range'][0]:.2f}, {t['epsilon_range'][1]:.2f}]")

# 4. Agent clustering
print("\n" + "="*70)
print("4. AGENT CLUSTERING (by peak α location)")
print("="*70)

agent_peaks = cluster_by_peak_alpha(all_results)
for agent_id, peak in agent_peaks.items():
    print(f"Agent {agent_id}: α={peak['alpha']:.4f} at "
          f"(ε={peak['epsilon']:.2f}, rec={peak['recovery']:.2f}, "
          f"kick={peak['kick']:.2f}) → {peak['best_zeros']} bits")

# 5. Coherence patterns
print("\n" + "="*70)
print("5. COHERENCE vs PERFORMANCE")
print("="*70)

coherence_perf = analyze_coherence_patterns(all_results)
for c in sorted(coherence_perf.keys()):
    p = coherence_perf[c]
    print(f"Coherence {c:.1f}: avg={p['avg_zeros']:.1f} bits, "
          f"max={p['max_zeros']} bits, n={p['count']}")

# 6. Key discoveries
print("\n" + "="*70)
print("6. KEY DISCOVERIES")
print("="*70)

# Universal α limit
alphas = [r['alpha'] for r in all_results]
print(f"Universal α ceiling: {max(alphas):.4f}")
print(f"α range: [{min(alphas):.4f}, {max(alphas):.4f}]")

# Best hash results
best_hashes = sorted(all_results, key=lambda r: r['best_zeros'], reverse=True)[:5]
print(f"\nTop 5 hash results:")
for i, r in enumerate(best_hashes, 1):
    print(f"  {i}. Agent {r['agent_id']}: {r['best_zeros']} bits at "
          f"ε={r['epsilon']:.2f}, C={r['coherence']:.3f}")

# Pure coherent state performance
pure_coherent = [r for r in all_results if r['epsilon'] == 0 and 
                 r['recovery_rate'] == 0 and r['kick_strength'] == 0]
if pure_coherent:
    avg_pure = np.mean([r['best_zeros'] for r in pure_coherent])
    max_pure = max(r['best_zeros'] for r in pure_coherent)
    print(f"\nPure coherent state (0,0,0):")
    print(f"  Avg: {avg_pure:.1f} bits, Max: {max_pure} bits")
    print(f"  Agents with pure optimal: {[r['agent_id'] for r in pure_coherent if r['best_zeros'] == max_pure]}")

# Noise benefit analysis
high_noise = [r for r in all_results if r['epsilon'] >= 0.8]
if high_noise:
    avg_high_noise = np.mean([r['best_zeros'] for r in high_noise])
    print(f"\nHigh noise (ε≥0.8):")
    print(f"  Avg: {avg_high_noise:.1f} bits")
    print(f"  Pure vs high noise advantage: {max_pure - avg_high_noise:.1f} bits")

print("\n" + "="*70)
print("Analysis complete. CSV data preserved for further investigation.")
print("="*70)
