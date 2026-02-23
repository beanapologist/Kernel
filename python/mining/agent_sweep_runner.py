#!/usr/bin/env python3
"""
Agent Sweep Runner - Individual agent execution
Performs systematic parameter sweeps with adaptive exploration
"""

import json
import csv
import argparse
import numpy as np
import hashlib
import struct
import time
from pathlib import Path
from typing import List, Tuple

class CoherentSweepMiner:
    """
    Single-agent miner with parameter sweep capabilities
    Implements adaptive epsilon stepping and safety mechanisms
    """
    
    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.agent_id = self.config['agent_id']
        self.region = self.config['region_assignment']
        
        # Bitcoin setup
        self.wallet = "1cYCgCm9wiTzkHxSaEh2TVF9bnd1E9zbT"
        self.target_bits = 0x17050000
        self.target = self._bits_to_target(self.target_bits)
        
        # Phase state (initialized to region center)
        self.phase = self.region['phase_center']
        self.phase_velocity = 0.0
        
        # Coherence tracking
        self.alpha = self.region['alpha_center']
        self.coherence = 1.0
        
        # Results
        self.sweep_results = []
        self.checkpoint_data = []
        
        print(f"\n{'='*60}")
        print(f"Agent {self.agent_id} Sweep Runner")
        print(f"{'='*60}")
        print(f"Phase region: [{self.region['phase_range'][0]:.3f}, {self.region['phase_range'][1]:.3f}]")
        print(f"Starting α: {self.alpha:.3f}")
        print(f"{'='*60}\n")
    
    def _bits_to_target(self, bits):
        exponent = bits >> 24
        mantissa = bits & 0xFFFFFF
        return mantissa * (256 ** (exponent - 3))
    
    def _sech(self, x):
        """sech(x) = 1/cosh(x)"""
        return 1.0 / np.cosh(x)
    
    def _compute_G_eff(self, lambda_val):
        """G_eff = sech(λ) - Ohm-coherence duality"""
        return self._sech(lambda_val)
    
    def _mine_sample(self, phase, num_hashes=1000):
        """Quick mining sample at given phase"""
        header = self._get_block_template()
        
        best_hash_int = 2**256
        hashes_computed = 0
        
        # Phase-dependent nonce selection
        nonce_base = int((np.sin(phase) + 1) * 2**31)
        
        for i in range(num_hashes):
            nonce = (nonce_base + i) % (2**32)
            block = header + struct.pack('<I', nonce)
            hash1 = hashlib.sha256(block).digest()
            hash2 = hashlib.sha256(hash1).digest()
            hash_int = int.from_bytes(hash2, 'little')
            
            hashes_computed += 1
            if hash_int < best_hash_int:
                best_hash_int = hash_int
        
        best_zeros = 256 - best_hash_int.bit_length()
        return best_zeros, hashes_computed
    
    def _get_block_template(self):
        """Minimal Bitcoin block header"""
        version = struct.pack('<I', 0x20000000)
        prev_block = b'\x00' * 32
        merkle_root = hashlib.sha256(self.wallet.encode()).digest()
        timestamp = struct.pack('<I', int(time.time()))
        bits = struct.pack('<I', self.target_bits)
        return version + prev_block + merkle_root + timestamp + bits
    
    def _adaptive_epsilon_step(self, current_eps, delta_alpha):
        """
        Adjust epsilon step size based on Δα sensitivity
        High Δα → smaller steps (near transition)
        Low Δα → larger steps (flat region)
        """
        threshold = self.config['delta_alpha_threshold']
        
        if delta_alpha > threshold * 2:
            return current_eps * 0.5  # Slow down near sharp transition
        elif delta_alpha < threshold * 0.5:
            return min(current_eps * 1.5, 0.1)  # Speed up in flat regions
        else:
            return current_eps
    
    def _check_runaway(self, alpha):
        """Detect runaway α (kick-induced drift)"""
        max_alpha = self.config['max_alpha']
        return alpha > max_alpha
    
    def _renormalize(self):
        """Auto-renormalization to prevent runaway"""
        print(f"    [RENORMALIZE] α={self.alpha:.3f} → 1.0")
        self.alpha = 1.0
        self.phase_velocity *= 0.5  # Damp velocity too
    
    def sweep_parameter_space(self):
        """
        Main sweep loop across (ε, recovery_rate, kick_strength)
        """
        epsilon_sweep = self.config['noise_epsilon_sweep']
        recovery_sweep = self.config['recovery_rate_sweep']
        kick_sweep = self.config['kick_strength_sweep']
        
        total_combinations = len(epsilon_sweep) * len(recovery_sweep) * len(kick_sweep)
        completed = 0
        
        print(f"Starting parameter sweep: {total_combinations} combinations\n")
        
        for recovery_rate in recovery_sweep:
            for kick_strength in kick_sweep:
                
                # Adaptive epsilon sweep
                eps_idx = 0
                eps_step = epsilon_sweep[1] - epsilon_sweep[0] if len(epsilon_sweep) > 1 else 0.05
                
                prev_alpha = self.alpha
                
                while eps_idx < len(epsilon_sweep):
                    epsilon = epsilon_sweep[eps_idx]
                    
                    # Reset phase to region center for each parameter combo
                    self.phase = self.region['phase_center']
                    
                    # Apply noise
                    phase_noise = np.random.normal(0, epsilon)
                    noisy_phase = self.phase + phase_noise
                    
                    # Compute lambda from phase
                    lambda_val = abs(np.sin(noisy_phase)) * 2.0
                    
                    # Apply recovery
                    phase_correction = -phase_noise * recovery_rate
                    recovered_phase = noisy_phase + phase_correction
                    
                    # Optional kick
                    if kick_strength > 0:
                        kick = kick_strength * np.cos(recovered_phase)
                        self.phase_velocity += kick
                    
                    final_phase = recovered_phase + self.phase_velocity
                    
                    # Compute G_eff and coherence
                    G_eff = self._compute_G_eff(lambda_val)
                    self.coherence = G_eff
                    
                    # Update alpha (simplified dynamics)
                    self.alpha = 1.0 + (1.0 - G_eff) * 0.5
                    
                    # Mine sample
                    best_zeros, hashes = self._mine_sample(final_phase, num_hashes=500)
                    
                    # Compute Δα
                    delta_alpha = abs(self.alpha - prev_alpha)
                    
                    # Check for runaway
                    runaway = self._check_runaway(self.alpha)
                    renormalized = False
                    
                    if runaway and self.config['auto_renormalize']:
                        self._renormalize()
                        renormalized = True
                    
                    # Record result
                    result = {
                        'agent_id': self.agent_id,
                        'epsilon': epsilon,
                        'recovery_rate': recovery_rate,
                        'kick_strength': kick_strength,
                        'alpha': self.alpha,
                        'coherence': self.coherence,
                        'lambda': lambda_val,
                        'G_eff': G_eff,
                        'best_zeros': best_zeros,
                        'hashes': hashes,
                        'phase_start': self.region['phase_center'],
                        'phase_end': final_phase,
                        'delta_alpha': delta_alpha,
                        'runaway': runaway,
                        'renormalized': renormalized
                    }
                    
                    self.sweep_results.append(result)
                    
                    # Adaptive epsilon stepping
                    if self.config['adaptive_epsilon_steps']:
                        new_step = self._adaptive_epsilon_step(eps_step, delta_alpha)
                        if new_step != eps_step:
                            print(f"  [ADAPTIVE] Δα={delta_alpha:.4f} → step {eps_step:.4f}→{new_step:.4f}")
                            eps_step = new_step
                    
                    prev_alpha = self.alpha
                    eps_idx += 1
                    completed += 1
                    
                    # Progress
                    if completed % 20 == 0:
                        pct = 100 * completed / total_combinations
                        print(f"  Progress: {completed}/{total_combinations} ({pct:.1f}%) | "
                              f"α={self.alpha:.3f}, C={self.coherence:.3f}, zeros={best_zeros}")
                    
                    # Checkpoint
                    if completed % self.config['checkpoint_interval'] == 0:
                        self._save_checkpoint()
        
        print(f"\nSweep complete! {completed} points explored")
        return self.sweep_results
    
    def _save_checkpoint(self):
        """Save checkpoint for potential rollback"""
        checkpoint = {
            'alpha': self.alpha,
            'phase': self.phase,
            'phase_velocity': self.phase_velocity,
            'results_count': len(self.sweep_results)
        }
        self.checkpoint_data.append(checkpoint)
    
    def save_results(self, output_path: Path):
        """Write results to CSV"""
        fieldnames = [
            'agent_id', 'epsilon', 'recovery_rate', 'kick_strength',
            'alpha', 'coherence', 'lambda', 'G_eff',
            'best_zeros', 'hashes', 'phase_start', 'phase_end',
            'delta_alpha', 'runaway', 'renormalized'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.sweep_results)
        
        print(f"\n💾 Results saved: {output_path}")
        
        # Summary stats
        max_alpha = max(r['alpha'] for r in self.sweep_results)
        max_zeros = max(r['best_zeros'] for r in self.sweep_results)
        avg_coherence = np.mean([r['coherence'] for r in self.sweep_results])
        
        # Find optimal parameters
        best_result = max(self.sweep_results, key=lambda r: r['alpha'])
        
        print(f"\n{'='*60}")
        print(f"Agent {self.agent_id} Summary")
        print(f"{'='*60}")
        print(f"Peak α: {max_alpha:.4f}")
        print(f"Best hash: {max_zeros} leading zeros")
        print(f"Avg coherence: {avg_coherence:.4f}")
        print(f"\nOptimal parameters:")
        print(f"  ε* = {best_result['epsilon']:.4f}")
        print(f"  recovery* = {best_result['recovery_rate']:.2f}")
        print(f"  kick* = {best_result['kick_strength']:.2f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-id', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config_path = Path(args.config)
    output_dir = config_path.parent
    
    print(__doc__)
    
    miner = CoherentSweepMiner(config_path)
    results = miner.sweep_parameter_space()
    
    output_path = output_dir / f"agent_{args.agent_id}_sweep.csv"
    miner.save_results(output_path)
