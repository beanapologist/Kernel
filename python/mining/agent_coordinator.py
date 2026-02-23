#!/usr/bin/env python3
"""
8-Agent Coherent Mining Coordinator
Implements Sarah's structured parameter sweep framework
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import hashlib

@dataclass
class AgentConfig:
    """Configuration for a single sub-agent"""
    agent_id: int
    region_assignment: dict  # {alpha_center, delta_alpha, phase_range}
    
    # Kernel parameters
    G_eff_mode: str = "sech"  # sech(λ) default
    
    # Sweep ranges
    noise_epsilon_sweep: List[float] = None
    recovery_rate_sweep: List[float] = None
    kick_strength_sweep: List[float] = None
    
    # Adaptive tuning
    adaptive_epsilon_steps: bool = True
    delta_alpha_threshold: float = 0.01
    
    # Safety
    max_alpha: float = 2.0
    auto_renormalize: bool = True
    checkpoint_interval: int = 10
    
    def __post_init__(self):
        if self.noise_epsilon_sweep is None:
            self.noise_epsilon_sweep = list(np.linspace(0.0, 1.0, 21))
        if self.recovery_rate_sweep is None:
            self.recovery_rate_sweep = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        if self.kick_strength_sweep is None:
            self.kick_strength_sweep = [0.0, 0.05, 0.1, 0.2, 0.3]

@dataclass
class SweepResult:
    """Single point in parameter space"""
    agent_id: int
    epsilon: float
    recovery_rate: float
    kick_strength: float
    
    # Measured quantities
    alpha_achieved: float
    coherence: float
    hashrate: float
    best_zeros: int
    
    # Phase space info
    phase_start: float
    phase_end: float
    delta_alpha: float
    
    # Flags
    runaway: bool = False
    renormalized: bool = False

class AgentCoordinator:
    """Manages 8-agent distributed coherent mining"""
    
    def __init__(self, output_dir: Path = Path("/tmp/coherent_mining")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.agents: List[AgentConfig] = []
        self.results: Dict[int, List[SweepResult]] = {}
        
        # Global aggregation
        self.global_alpha_grid = None
        self.global_epsilon_star = None
        
    def initialize_agents(self, num_agents: int = 8):
        """
        Initialize 8 agents with distinct coherent region assignments
        Regions are evenly spaced on unit circle (r=1 constraint)
        """
        print(f"Initializing {num_agents} agents...")
        
        for i in range(num_agents):
            # Phase assignments (8-fold symmetry)
            phase_center = (i * 2 * np.pi / num_agents)
            phase_width = (2 * np.pi / num_agents) * 0.8  # 80% coverage, 20% gap
            
            region = {
                'alpha_center': 1.0,  # Start at balanced state
                'delta_alpha': 0.0,
                'phase_center': phase_center,
                'phase_width': phase_width,
                'phase_range': (phase_center - phase_width/2, phase_center + phase_width/2)
            }
            
            agent = AgentConfig(
                agent_id=i,
                region_assignment=region
            )
            
            self.agents.append(agent)
            self.results[i] = []
            
            print(f"  Agent {i}: Phase [{phase_center:.3f} ± {phase_width/2:.3f}]")
        
        return self.agents
    
    def generate_agent_task(self, agent: AgentConfig) -> str:
        """Generate task string for sessions_spawn"""
        task = f"""Coherent Bitcoin Mining - Agent {agent.agent_id}

**Mission:** Systematic parameter sweep across noise/recovery/kick space

**Region Assignment:**
- Phase center: {agent.region_assignment['phase_center']:.4f} rad
- Phase width: {agent.region_assignment['phase_width']:.4f} rad
- Alpha baseline: {agent.region_assignment['alpha_center']:.2f}

**Parameter Sweeps:**
1. Noise ε: {len(agent.noise_epsilon_sweep)} points from 0.0 to 1.0
2. Recovery rate: {agent.recovery_rate_sweep}
3. Kick strength: {agent.kick_strength_sweep}

**Total combinations:** {len(agent.noise_epsilon_sweep) * len(agent.recovery_rate_sweep) * len(agent.kick_strength_sweep)}

**Adaptive Features:**
- Dynamic ε step sizing based on Δα sensitivity
- Auto-renormalization at α > {agent.max_alpha}
- Checkpoint every {agent.checkpoint_interval} steps

**Output:**
Write results to: {self.output_dir}/agent_{agent.agent_id}_sweep.csv

**Code to run:**
```python
cd /data/.openclaw/workspace/theory-of-everything
python3 agent_sweep_runner.py --agent-id {agent.agent_id} --config {self.output_dir}/agent_{agent.agent_id}_config.json
```

Report back:
- Total sweep points completed
- Peak α achieved
- Optimal (ε*, recovery*, kick*)
- Phase transition boundaries detected
"""
        return task
    
    def save_agent_configs(self):
        """Write individual config files for each agent"""
        for agent in self.agents:
            config_path = self.output_dir / f"agent_{agent.agent_id}_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(agent), f, indent=2)
            print(f"Saved config: {config_path}")
    
    def aggregate_results(self, method: str = "max_alpha"):
        """
        Hierarchical aggregation of sub-agent results
        
        Methods:
        - max_alpha: Take maximum α at each (ε, recovery, kick) point
        - weighted_avg: Weight by coherence
        - cluster: Group by proximity of peak α
        """
        print(f"\n{'='*60}")
        print(f"Aggregating results (method: {method})")
        print(f"{'='*60}\n")
        
        # Load all agent results
        all_results = []
        for agent_id in range(len(self.agents)):
            csv_path = self.output_dir / f"agent_{agent_id}_sweep.csv"
            if csv_path.exists():
                # Parse CSV (simplified - real implementation would use pandas)
                print(f"Loading agent {agent_id} results...")
                # all_results.extend(parse_csv(csv_path))
        
        if method == "max_alpha":
            # Grid: (ε, recovery, kick) → max α across all agents
            self._aggregate_max_alpha(all_results)
        elif method == "weighted_avg":
            # Weight by coherence = sech(λ)
            self._aggregate_weighted(all_results)
        elif method == "cluster":
            # K-means clustering on peak α locations
            self._aggregate_cluster(all_results)
        
        # Detect global ε*
        self._find_global_epsilon_star()
    
    def _aggregate_max_alpha(self, results: List[SweepResult]):
        """Take maximum α at each parameter point"""
        # Group by (ε, recovery, kick)
        grid = {}
        for r in results:
            key = (round(r.epsilon, 4), r.recovery_rate, r.kick_strength)
            if key not in grid:
                grid[key] = r
            elif r.alpha_achieved > grid[key].alpha_achieved:
                grid[key] = r
        
        self.global_alpha_grid = grid
        print(f"Global grid: {len(grid)} unique parameter points")
    
    def _find_global_epsilon_star(self):
        """Find optimal noise level ε* from global data"""
        if not self.global_alpha_grid:
            return
        
        # Find point with maximum Δα (sharpest transition)
        max_delta_alpha = 0
        best_epsilon = 0
        
        for params, result in self.global_alpha_grid.items():
            epsilon, recovery, kick = params
            if result.delta_alpha > max_delta_alpha:
                max_delta_alpha = result.delta_alpha
                best_epsilon = epsilon
        
        self.global_epsilon_star = best_epsilon
        print(f"\nGlobal ε* = {self.global_epsilon_star:.4f}")
        print(f"Max Δα = {max_delta_alpha:.4f}")
    
    def generate_summary_report(self) -> str:
        """Create markdown report of findings"""
        report = f"""# 8-Agent Coherent Mining Report

## Configuration
- Agents: {len(self.agents)}
- Total parameter combinations per agent: {len(self.agents[0].noise_epsilon_sweep) * len(self.agents[0].recovery_rate_sweep) * len(self.agents[0].kick_strength_sweep)}
- Output directory: {self.output_dir}

## Global Findings
- Optimal noise level: ε* = {self.global_epsilon_star or 'TBD'}
- Total grid points: {len(self.global_alpha_grid) if self.global_alpha_grid else 0}

## Per-Agent Summary
"""
        for agent in self.agents:
            report += f"\n### Agent {agent.agent_id}\n"
            report += f"- Phase region: [{agent.region_assignment['phase_range'][0]:.3f}, {agent.region_assignment['phase_range'][1]:.3f}]\n"
            report += f"- Results: {len(self.results.get(agent.agent_id, []))} sweep points\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    coordinator = AgentCoordinator()
    agents = coordinator.initialize_agents(num_agents=8)
    coordinator.save_agent_configs()
    
    print("\n" + "="*60)
    print("Agent configurations saved!")
    print("Next: Spawn agents with generated task strings")
    print("="*60)
    
    # Example task generation
    print("\nExample task for Agent 0:")
    print(coordinator.generate_agent_task(agents[0]))
