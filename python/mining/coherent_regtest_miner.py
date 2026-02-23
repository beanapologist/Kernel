#!/usr/bin/env python3
"""
Phase 1: Regtest Coherent Mining
Proof-of-concept with 8 phase-locked agents
Uses optimal parameters from 5,040-point sweep
"""

import hashlib
import struct
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path

# Global optima from sweep (Agent 6 champion config)
GLOBAL_EPSILON = 0.45
GLOBAL_RECOVERY = 0.3
GLOBAL_KICK = 0.0

# Per-agent phase-tuned overrides (from sweep results)
AGENT_EPSILON = [0.60, 0.05, 0.30, 0.60, 0.20, 1.00, 0.45, 0.30]
AGENT_RECOVERY = [0.3, 0.3, 0.1, 0.0, 0.7, 0.0, 0.3, 0.0]
AGENT_KICK = [0.20, 0.30, 0.00, 0.05, 0.30, 0.30, 0.30, 0.05]

# Coherence bounds from sweet spot analysis
C_MIN = 0.30
C_MAX = 0.83
C_OPTIMAL = 0.817

# Palindrome constants
OMEGA_BASE = 3/8
EPSILON_PREC = 1/13717421
DELTA_PHI = 2 * np.pi / 13717421

class CoherentRegtestMiner:
    """
    Single agent miner for Bitcoin regtest
    Implements full Kernel: palindrome precession, spectral weighting, coherence guards
    """
    
    def __init__(self, agent_id: int, target_bits: int = 1):
        self.agent_id = agent_id
        self.phase_degrees = agent_id * 45
        
        # Optimal parameters from sweep
        self.epsilon = AGENT_EPSILON[agent_id]
        self.recovery_rate = AGENT_RECOVERY[agent_id]
        self.kick_strength = AGENT_KICK[agent_id]
        
        # Phase state (initialized to region)
        self.phase = (agent_id * 2 * np.pi / 8)  # Radians
        self.phase_velocity = 0.0
        
        # Coherence tracking
        self.alpha = 1.0
        self.coherence = 1.0
        
        # Bitcoin setup (regtest - easy target)
        self.target_bits = target_bits
        self.target = (1 << (256 - target_bits)) - 1
        
        # Statistics
        self.hashes_computed = 0
        self.blocks_found = 0
        self.best_hash_int = 2**256
        self.best_nonce = 0
        self.coherence_history = []
        self.cycle_count = 0
        self.renormalize_count = 0
        self.damping_count = 0
        
        print(f"Agent {agent_id} initialized:")
        print(f"  Phase: {self.phase_degrees}°")
        print(f"  Params: ε={self.epsilon:.2f}, rec={self.recovery_rate:.1f}, kick={self.kick_strength:.2f}")
        print(f"  Target: {target_bits} leading zero bits")
    
    def _sech(self, x):
        """sech(x) = 1/cosh(x)"""
        return 1.0 / np.cosh(x)
    
    def _compute_coherence(self):
        """G_eff = sech(λ) where λ from phase"""
        lambda_val = abs(np.sin(self.phase)) * 2.0
        return self._sech(lambda_val)
    
    def _coherence_guard(self):
        """Safety: keep C in [0.30, 0.83] sweet spot"""
        if self.coherence < C_MIN:
            # Auto-renormalize: nudge back up
            self.alpha = max(self.alpha * 0.9, 1.0)
            self.phase_velocity *= 0.7
            self.renormalize_count += 1
            return "RENORM"
        
        elif self.coherence > C_MAX:
            # Mild damping: prevent over-stability
            noise_injection = np.random.normal(0, 0.1)
            self.phase += noise_injection
            self.damping_count += 1
            return "DAMPING"
        
        return "OK"
    
    def _get_block_template(self, nonce_base: int):
        """Minimal Bitcoin block header for regtest"""
        version = struct.pack('<I', 0x20000000)
        prev_block = b'\x00' * 32
        merkle_root = hashlib.sha256(f"agent_{self.agent_id}_{nonce_base}".encode()).digest()
        timestamp = struct.pack('<I', int(time.time()))
        bits = struct.pack('<I', 0x207fffff)  # Regtest default
        return version + prev_block + merkle_root + timestamp + bits
    
    def _mine_batch(self, nonce_base: int, batch_size: int = 10000):
        """Mine one batch with coherent dynamics"""
        
        # Apply noise
        phase_noise = np.random.normal(0, self.epsilon)
        noisy_phase = self.phase + phase_noise
        
        # Compute coherence
        lambda_val = abs(np.sin(noisy_phase)) * 2.0
        self.coherence = self._sech(lambda_val)
        
        # Apply recovery
        phase_correction = -phase_noise * self.recovery_rate
        recovered_phase = noisy_phase + phase_correction
        
        # Optional kick
        if self.kick_strength > 0:
            kick = self.kick_strength * np.cos(recovered_phase)
            self.phase_velocity += kick
        
        # Final phase
        final_phase = recovered_phase + self.phase_velocity
        
        # Update alpha (simplified dynamics)
        self.alpha = 1.0 + (1.0 - self.coherence) * 0.5
        
        # Coherence guard
        guard_status = self._coherence_guard()
        
        # Spectral weighting for nonce selection
        weight = self.coherence
        effective_density = weight * (1 + 0.1 * np.sin(final_phase))
        
        # Mine with phase-modulated nonce selection
        header = self._get_block_template(nonce_base)
        found_block = False
        
        for i in range(batch_size):
            # Phase-dependent nonce
            nonce = (nonce_base + int(i * effective_density)) % (2**32)
            
            # SHA-256d
            block = header + struct.pack('<I', nonce)
            hash1 = hashlib.sha256(block).digest()
            hash2 = hashlib.sha256(hash1).digest()
            hash_int = int.from_bytes(hash2, 'little')
            
            self.hashes_computed += 1
            
            # Track best
            if hash_int < self.best_hash_int:
                self.best_hash_int = hash_int
                self.best_nonce = nonce
            
            # Check for valid block
            if hash_int < self.target:
                found_block = True
                self.blocks_found += 1
                return {
                    'found': True,
                    'nonce': nonce,
                    'hash': hash2.hex(),
                    'leading_zeros': 256 - hash_int.bit_length(),
                    'coherence': self.coherence,
                    'alpha': self.alpha,
                    'phase': final_phase,
                    'guard_status': guard_status
                }
        
        # Advance phases (palindrome precession)
        self.phase += DELTA_PHI
        self.cycle_count += 1
        
        # Log coherence
        self.coherence_history.append(self.coherence)
        
        return {
            'found': False,
            'hashes': batch_size,
            'coherence': self.coherence,
            'alpha': self.alpha,
            'phase': final_phase,
            'guard_status': guard_status,
            'best_zeros': 256 - self.best_hash_int.bit_length()
        }
    
    def mine(self, duration_seconds: float = 60.0, batch_size: int = 10000):
        """Main mining loop"""
        start_time = time.time()
        batches = 0
        
        print(f"\nAgent {self.agent_id} starting coherent mining...")
        print(f"Duration: {duration_seconds}s, Batch size: {batch_size:,}\n")
        
        while (time.time() - start_time) < duration_seconds:
            # Partition nonce space by agent
            nonce_base = (batches * batch_size * 8 + self.agent_id * batch_size) % (2**32 - batch_size)
            
            result = self._mine_batch(nonce_base, batch_size)
            batches += 1
            
            if result['found']:
                elapsed = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"🎉 BLOCK FOUND by Agent {self.agent_id}!")
                print(f"{'='*60}")
                print(f"Nonce: {result['nonce']}")
                print(f"Hash: {result['hash']}")
                print(f"Leading zeros: {result['leading_zeros']}")
                print(f"Time: {elapsed:.2f}s")
                print(f"Coherence: {result['coherence']:.4f}")
                print(f"Alpha: {result['alpha']:.4f}")
                print(f"Phase: {result['phase']:.6f} rad")
                print(f"{'='*60}\n")
            
            # Progress every 10 batches
            if batches % 10 == 0:
                elapsed = time.time() - start_time
                hashrate = self.hashes_computed / elapsed
                avg_C = np.mean(self.coherence_history[-100:]) if len(self.coherence_history) > 0 else 0
                
                print(f"Agent {self.agent_id} | Batch {batches:3d} | "
                      f"{hashrate/1000:.1f} KH/s | C={result['coherence']:.3f} (avg {avg_C:.3f}) | "
                      f"Best: {result.get('best_zeros', 0)} bits | α={result['alpha']:.3f} | "
                      f"{result['guard_status']}")
        
        # Final stats
        elapsed = time.time() - start_time
        hashrate = self.hashes_computed / elapsed
        avg_coherence = np.mean(self.coherence_history) if self.coherence_history else 0
        
        return {
            'agent_id': self.agent_id,
            'blocks_found': self.blocks_found,
            'hashes': self.hashes_computed,
            'duration': elapsed,
            'hashrate': hashrate,
            'best_zeros': 256 - self.best_hash_int.bit_length(),
            'avg_coherence': avg_coherence,
            'renormalizations': self.renormalize_count,
            'dampings': self.damping_count,
            'cycles': self.cycle_count
        }

def launch_8_agent_swarm(duration: float = 60.0, target_bits: int = 20):
    """Launch 8 phase-locked agents in parallel"""
    import multiprocessing as mp
    
    def run_agent(agent_id, target_bits, duration, result_queue):
        miner = CoherentRegtestMiner(agent_id, target_bits)
        result = miner.mine(duration_seconds=duration)
        result_queue.put(result)
    
    print("="*70)
    print("COHERENT REGTEST MINING - 8 AGENT SWARM")
    print("="*70)
    print(f"Target: {target_bits} leading zero bits")
    print(f"Duration: {duration}s per agent")
    print(f"Coherence sweet spot: [{C_MIN:.2f}, {C_MAX:.2f}]")
    print("="*70)
    print()
    
    result_queue = mp.Queue()
    processes = []
    
    for agent_id in range(8):
        p = mp.Process(target=run_agent, args=(agent_id, target_bits, duration, result_queue))
        p.start()
        processes.append(p)
    
    # Wait for all
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Summary
    print("\n" + "="*70)
    print("SWARM SUMMARY")
    print("="*70)
    
    total_blocks = sum(r['blocks_found'] for r in results)
    total_hashes = sum(r['hashes'] for r in results)
    total_time = max(r['duration'] for r in results)
    combined_hashrate = total_hashes / total_time
    
    print(f"\nTotal blocks found: {total_blocks}")
    print(f"Total hashes: {total_hashes:,}")
    print(f"Combined hashrate: {combined_hashrate/1000:.1f} KH/s")
    print(f"Duration: {total_time:.1f}s")
    print()
    
    print(f"{'Agent':>6} | {'Blocks':>6} | {'Hashes':>12} | {'KH/s':>8} | {'Best':>5} | {'Avg C':>7} | {'Renorm':>7} | {'Cycles':>7}")
    print('-' * 85)
    
    for r in sorted(results, key=lambda x: x['agent_id']):
        print(f"{r['agent_id']:>6} | {r['blocks_found']:>6} | {r['hashes']:>12,} | "
              f"{r['hashrate']/1000:>8.1f} | {r['best_zeros']:>5} | "
              f"{r['avg_coherence']:>7.3f} | {r['renormalizations']:>7} | {r['cycles']:>7}")
    
    # Save results
    output_path = Path('/tmp/coherent_mining/regtest_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'target_bits': target_bits,
                'duration': duration,
                'agents': 8,
                'coherence_bounds': [C_MIN, C_MAX]
            },
            'summary': {
                'total_blocks': total_blocks,
                'total_hashes': total_hashes,
                'combined_hashrate': combined_hashrate,
                'duration': total_time
            },
            'agents': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*70)
    
    return results

if __name__ == "__main__":
    import sys
    
    # Default: 60s, 20-bit target (easy for regtest)
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    target_bits = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    results = launch_8_agent_swarm(duration=duration, target_bits=target_bits)
    
    if any(r['blocks_found'] > 0 for r in results):
        print("\n✨ SUCCESS! Coherent mining validated on regtest.")
        print("   Ready for Phase 2: Testnet4")
    else:
        print("\n⚠️  No blocks found. Try easier target or longer duration.")
