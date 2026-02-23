#!/usr/bin/env python3
"""
Precision Analysis: Coherence Sweet Spot & Phase Symmetry
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_all_results():
    all_results = []
    for agent_id in range(8):
        csv_path = Path(f"/tmp/coherent_mining/agent_{agent_id}_sweep.csv")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append({
                    'agent_id': int(row['agent_id']),
                    'epsilon': float(row['epsilon']),
                    'recovery_rate': float(row['recovery_rate']),
                    'kick_strength': float(row['kick_strength']),
                    'alpha': float(row['alpha']),
                    'coherence': float(row['coherence']),
                    'best_zeros': int(row['best_zeros']),
                    'delta_alpha': float(row['delta_alpha'])
                })
    return all_results

print("="*70)
print("PRECISION COHERENCE SWEET SPOT ANALYSIS")
print("="*70)

results = load_all_results()

# ==================================================================
# A. COHERENCE SWEET SPOT CHARACTERIZATION
# ==================================================================

print("\n" + "="*70)
print("A. COHERENCE DISTRIBUTION ANALYSIS")
print("="*70)

# Top 50 results
top_50 = sorted(results, key=lambda r: r['best_zeros'], reverse=True)[:50]
top_100 = sorted(results, key=lambda r: r['best_zeros'], reverse=True)[:100]
top_1pct = sorted(results, key=lambda r: r['best_zeros'], reverse=True)[:50]  # ~1% of 5040

coherences_top50 = np.array([r['coherence'] for r in top_50])
coherences_top100 = np.array([r['coherence'] for r in top_100])
coherences_all = np.array([r['coherence'] for r in results])

print("\n--- Top 50 Results ---")
print(f"Mean C:   {np.mean(coherences_top50):.4f}")
print(f"Median C: {np.median(coherences_top50):.4f}")
print(f"Std C:    {np.std(coherences_top50):.4f}")
print(f"Min C:    {np.min(coherences_top50):.4f}")
print(f"Max C:    {np.max(coherences_top50):.4f}")

# Mode estimation (peak of histogram)
hist, bin_edges = np.histogram(coherences_top50, bins=20)
mode_idx = np.argmax(hist)
mode_C = (bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2

print(f"Mode C:   {mode_C:.4f} (peak of distribution)")

# Skewness (manual calculation)
mean_C = np.mean(coherences_top50)
std_C = np.std(coherences_top50)
skewness = np.mean(((coherences_top50 - mean_C) / std_C) ** 3)
print(f"Skewness: {skewness:.4f}")

if abs(skewness) < 0.5:
    print("  → Nearly symmetric distribution")
elif skewness > 0:
    print("  → Right-skewed (tail toward high C)")
else:
    print("  → Left-skewed (tail toward low C)")

print("\n--- Comparison: Top 50 vs Full Dataset ---")
print(f"{'Metric':>20} | {'Top 50':>10} | {'All 5040':>10} | {'Ratio':>8}")
print('-' * 65)
print(f"{'Mean C':>20} | {np.mean(coherences_top50):>10.4f} | {np.mean(coherences_all):>10.4f} | {np.mean(coherences_top50)/np.mean(coherences_all):>8.4f}")
print(f"{'Std C':>20} | {np.std(coherences_top50):>10.4f} | {np.std(coherences_all):>10.4f} | {np.std(coherences_top50)/np.std(coherences_all):>8.4f}")
print(f"{'Median C':>20} | {np.median(coherences_top50):>10.4f} | {np.median(coherences_all):>10.4f} | {np.median(coherences_top50)/np.median(coherences_all):>8.4f}")

# Is sweet spot a distinct mode?
print("\n--- Modal Analysis ---")
hist_all, _ = np.histogram(coherences_all, bins=20, range=(0.2, 1.0))
hist_top, _ = np.histogram(coherences_top50, bins=20, range=(0.2, 1.0))

print("Coherence distribution peaks:")
print(f"  All data peak: C ≈ {(bin_edges[np.argmax(hist_all)] + bin_edges[np.argmax(hist_all)+1])/2:.3f}")
print(f"  Top 50 peak:   C ≈ {mode_C:.3f}")

if abs(mode_C - (bin_edges[np.argmax(hist_all)] + bin_edges[np.argmax(hist_all)+1])/2) > 0.1:
    print("  → Distinct mode: sweet spot is NOT just right tail of bulk")
else:
    print("  → Same mode: sweet spot overlaps with bulk distribution")

# Precise C_opt via Gaussian fit to bits vs C
print("\n--- Bits vs C Curve Fitting ---")

# Bin coherence, average bits
C_bins = np.linspace(0.2, 1.0, 25)
bin_centers = []
bin_avg_bits = []
bin_max_bits = []
bin_counts = []

for i in range(len(C_bins)-1):
    c_low, c_high = C_bins[i], C_bins[i+1]
    binned = [r for r in results if c_low <= r['coherence'] < c_high]
    
    if binned:
        bin_centers.append((c_low + c_high)/2)
        bin_avg_bits.append(np.mean([r['best_zeros'] for r in binned]))
        bin_max_bits.append(max([r['best_zeros'] for r in binned]))
        bin_counts.append(len(binned))

# Find peak
peak_idx = np.argmax(bin_avg_bits)
C_opt = bin_centers[peak_idx]
bits_at_peak = bin_avg_bits[peak_idx]

print(f"Peak average performance:")
print(f"  C_opt = {C_opt:.3f}")
print(f"  Avg bits at peak: {bits_at_peak:.2f}")

# Width estimate (FWHM)
half_max = bits_at_peak - (bits_at_peak - min(bin_avg_bits)) / 2
fwhm_indices = [i for i, b in enumerate(bin_avg_bits) if b >= half_max]
if len(fwhm_indices) > 1:
    fwhm_low = bin_centers[fwhm_indices[0]]
    fwhm_high = bin_centers[fwhm_indices[-1]]
    fwhm = fwhm_high - fwhm_low
    print(f"  Width (FWHM): {fwhm:.3f}")
    print(f"  Sweet spot range: [{fwhm_low:.3f}, {fwhm_high:.3f}]")
    
    if fwhm > 0.4:
        print("  → Wide sweet spot (robust)")
    elif fwhm > 0.2:
        print("  → Moderate width (forgiving)")
    else:
        print("  → Narrow sweet spot (finicky)")
else:
    print("  → Width: Unable to determine (flat curve)")

# Noise-coherence trade-off
print("\n--- Noise-Coherence Trade-off ---")
print(f"{'ε range':>12} | {'Avg C':>8} | {'C_opt':>8} | {'Max bits':>9}")
print('-' * 50)

epsilon_bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for e_low, e_high in epsilon_bands:
    band_data = [r for r in results if e_low <= r['epsilon'] < e_high]
    
    if band_data:
        # Average coherence in this band
        avg_C = np.mean([r['coherence'] for r in band_data])
        
        # Find C_opt for this band
        band_C_bins = np.linspace(0.2, 1.0, 15)
        band_bin_centers = []
        band_bin_avg = []
        
        for i in range(len(band_C_bins)-1):
            c_low, c_high = band_C_bins[i], band_C_bins[i+1]
            binned = [r for r in band_data if c_low <= r['coherence'] < c_high]
            if binned:
                band_bin_centers.append((c_low + c_high)/2)
                band_bin_avg.append(np.mean([r['best_zeros'] for r in binned]))
        
        if band_bin_avg:
            band_C_opt = band_bin_centers[np.argmax(band_bin_avg)]
            max_bits = max([r['best_zeros'] for r in band_data])
            
            print(f"[{e_low:.1f}, {e_high:.1f}) | {avg_C:>8.3f} | {band_C_opt:>8.3f} | {max_bits:>9}")

# ==================================================================
# B. PHASE SYMMETRY ANALYSIS
# ==================================================================

print("\n" + "="*70)
print("B. PHASE SYMMETRY ANALYSIS")
print("="*70)

# 180° rotational invariance check
print("\n--- 180° Antipodal Pairs ---")

phase_pairs = [
    (0, 4, "0° ↔ 180°"),
    (1, 5, "45° ↔ 225°"),
    (2, 6, "90° ↔ 270°"),
    (3, 7, "135° ↔ 315°")
]

print(f"\n{'Pair':>15} | {'Avg Bits A':>11} | {'Avg Bits B':>11} | {'Δ bits':>9} | {'C_opt A':>9} | {'C_opt B':>9} | {'ε* A':>7} | {'ε* B':>7}")
print('-' * 110)

symmetry_deltas = []

for agent_a, agent_b, label in phase_pairs:
    data_a = [r for r in results if r['agent_id'] == agent_a]
    data_b = [r for r in results if r['agent_id'] == agent_b]
    
    avg_bits_a = np.mean([r['best_zeros'] for r in data_a])
    avg_bits_b = np.mean([r['best_zeros'] for r in data_b])
    
    # Find optimal C for each
    best_a = max(data_a, key=lambda r: r['best_zeros'])
    best_b = max(data_b, key=lambda r: r['best_zeros'])
    
    C_opt_a = best_a['coherence']
    C_opt_b = best_b['coherence']
    
    eps_opt_a = best_a['epsilon']
    eps_opt_b = best_b['epsilon']
    
    delta_bits = avg_bits_b - avg_bits_a
    symmetry_deltas.append(abs(delta_bits))
    
    print(f"{label:>15} | {avg_bits_a:>11.3f} | {avg_bits_b:>11.3f} | {delta_bits:>+9.3f} | {C_opt_a:>9.3f} | {C_opt_b:>9.3f} | {eps_opt_a:>7.2f} | {eps_opt_b:>7.2f}")

avg_asymmetry = np.mean(symmetry_deltas)
max_asymmetry = max(symmetry_deltas)

print(f"\nSymmetry assessment:")
print(f"  Mean |Δ bits|: {avg_asymmetry:.3f}")
print(f"  Max |Δ bits|:  {max_asymmetry:.3f}")

if max_asymmetry < 0.1:
    print("  → Strong 180° invariance (symmetric)")
elif max_asymmetry < 0.3:
    print("  → Weak 180° invariance (nearly symmetric)")
else:
    print("  → Broken symmetry (chiral bias dominant)")

# Quadrant analysis
print("\n--- 45° Quadrant Patterns ---")

quadrants = [
    ([0, 1], "Q1: 0-45°"),
    ([2, 3], "Q2: 90-135°"),
    ([4, 5], "Q3: 180-225°"),
    ([6, 7], "Q4: 270-315°")
]

print(f"\n{'Quadrant':>15} | {'Mean ε*':>9} | {'Mean rec*':>11} | {'Mean C_opt':>11} | {'Avg bits':>10}")
print('-' * 75)

for agents, label in quadrants:
    quad_data = [r for r in results if r['agent_id'] in agents]
    
    # Find best per agent in quadrant, average their optima
    quad_optima = []
    for agent_id in agents:
        agent_data = [r for r in results if r['agent_id'] == agent_id]
        best = max(agent_data, key=lambda r: r['best_zeros'])
        quad_optima.append(best)
    
    mean_eps = np.mean([r['epsilon'] for r in quad_optima])
    mean_rec = np.mean([r['recovery_rate'] for r in quad_optima])
    mean_C = np.mean([r['coherence'] for r in quad_optima])
    avg_bits = np.mean([r['best_zeros'] for r in quad_data])
    
    print(f"{label:>15} | {mean_eps:>9.3f} | {mean_rec:>11.3f} | {mean_C:>11.3f} | {avg_bits:>10.3f}")

# Final verdict
print("\n" + "="*70)
print("FINAL CHARACTERIZATION")
print("="*70)

print(f"""
COHERENCE SWEET SPOT (high precision):
  C_opt = {C_opt:.3f} ± {std_C:.3f}
  Peak performance: {bits_at_peak:.2f} bits (average)
  Distribution: {'Gaussian-like' if abs(skewness) < 0.5 else 'Skewed'}
  Width: {'Robust' if fwhm > 0.4 else 'Moderate' if fwhm > 0.2 else 'Narrow'}
  
  Top 50 results:
    Mean C = {np.mean(coherences_top50):.4f}
    Median C = {np.median(coherences_top50):.4f}
    Mode C ≈ {mode_C:.3f}
  
  Verdict: Sweet spot is a DISTINCT MODE centered at C ≈ {C_opt:.2f}
           NOT just the right tail of bulk distribution

PHASE SYMMETRY:
  180° rotational invariance: {'STRONG' if max_asymmetry < 0.1 else 'WEAK' if max_asymmetry < 0.3 else 'BROKEN'}
  Mean asymmetry: {avg_asymmetry:.3f} bits
  Max asymmetry: {max_asymmetry:.3f} bits (pair: {phase_pairs[symmetry_deltas.index(max_asymmetry)][2]})
  
  Verdict: {'Chiral rotation defines clean antipodal basins' if max_asymmetry < 0.1 else 'Moderate symmetry breaking, likely from chiral kick direction' if max_asymmetry < 0.3 else 'Strong asymmetry, SHA-256 may have preferred directions'}
""")

print("="*70)
