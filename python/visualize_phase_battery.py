#!/usr/bin/env python3
"""
visualize_phase_battery.py — Interactive prototype for PhaseBattery geometric
learning dynamics (frustration / coherence / adaptive-α phase space).

Implements the same mathematics as PhaseBattery::feedback_step() in
ohm_coherence_duality.hpp (PR #51) entirely in Python so the app runs
standalone on simulated data, or on a CSV exported by write_debug_csv().

Visualization features
──────────────────────
1. **3D Phase Space** — E(t), R(t), α(t) trajectory (rotate / zoom).
2. **Node Deviations Heatmap** — per-node δθ_j values for every time step.
3. **Interactive Controls** — sliders for c₁ and c₂ (α sensitivity
   parameters) that re-simulate and update all plots in real time.

Usage
─────
    python3 python/visualize_phase_battery.py            # simulated data
    python3 python/visualize_phase_battery.py debug.csv  # load CSV from C++

Open http://127.0.0.1:8050 in a browser.

Dependencies
────────────
    pip install dash plotly numpy
"""

import math
import sys
import csv
import os
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

# ──────────────────────────────────────────────────────────────────────────────
# PhaseBattery simulation (mirrors ohm_coherence_duality.hpp)
# ──────────────────────────────────────────────────────────────────────────────

_TWO_PI = 2.0 * math.pi


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (−π, π] — mirrors wrap_angle() in the C++ header."""
    return a - _TWO_PI * np.floor((a + math.pi) / _TWO_PI)


def simulate_phase_battery(
    n_nodes: int = 12,
    gain: float = 0.35,
    alpha_base: float = 0.5,
    c1: float = 0.2,
    c2: float = 0.3,
    n_steps: int = 40,
    seed: int = 7,
) -> dict:
    """
    Simulate PhaseBattery::feedback_step() with adaptive-α for *n_steps*
    iterations, recording E, R, α, and per-node δθ_j at every step.

    Adaptive-α formula (from PR #51):
        ΔE          = E_before − E_mid          (frustration decay, sub-step 1)
        ΔR          = R_mid    − R_before        (coherence gain,   sub-step 1)
        α_adaptive  = clamp(alpha_base + c1·ΔE + c2·ΔR, 0, 1)

    Returns a dict with keys:
        iterations  — list[int]
        E           — list[float]   frustration after each step
        R           — list[float]   coherence after each step
        alpha       — list[float]   adaptive α each step
        delta_theta — list[list[float]]  per-node deviations (n_steps × n_nodes)
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-math.pi, math.pi, n_nodes)

    iterations, E_hist, R_hist, alpha_hist, dtheta_hist = [], [], [], [], []

    g = gain

    for step in range(n_steps):
        # ── Capture per-node deviations before sub-step 1 ──────────────────
        cx = np.cos(phases).mean()
        cy = np.sin(phases).mean()
        psi_bar = math.atan2(cy, cx)
        delta_theta = _wrap(phases - psi_bar)
        dtheta_hist.append(delta_theta.tolist())

        # ── Measure E and R before sub-step 1 ──────────────────────────────
        E_before = float(np.mean(delta_theta**2))
        R_before = math.sqrt(cx**2 + cy**2)

        # ── Sub-step 1: standard EMA contraction ───────────────────────────
        phases = phases - g * delta_theta

        cx_mid = np.cos(phases).mean()
        cy_mid = np.sin(phases).mean()
        psi_bar_mid = math.atan2(cy_mid, cx_mid)
        dtheta_mid = _wrap(phases - psi_bar_mid)

        E_mid = float(np.mean(dtheta_mid**2))
        R_mid = math.sqrt(cx_mid**2 + cy_mid**2)

        # ── Adaptive α ─────────────────────────────────────────────────────
        delta_E = E_before - E_mid
        delta_R = R_mid - R_before
        alpha_adaptive = float(
            min(1.0, max(0.0, alpha_base + c1 * delta_E + c2 * delta_R))
        )

        # ── Sub-step 2: coherence-amplified feedback pass ──────────────────
        g_fb = g * alpha_adaptive * R_mid
        phases = phases - g_fb * _wrap(phases - psi_bar_mid)

        # ── Observables after the full step ────────────────────────────────
        cx_out = np.cos(phases).mean()
        cy_out = np.sin(phases).mean()
        psi_bar_out = math.atan2(cy_out, cx_out)
        E_after = float(np.mean(_wrap(phases - psi_bar_out) ** 2))
        R_after = math.sqrt(cx_out**2 + cy_out**2)

        iterations.append(step)
        E_hist.append(E_after)
        R_hist.append(R_after)
        alpha_hist.append(alpha_adaptive)

    return {
        "iterations": iterations,
        "E": E_hist,
        "R": R_hist,
        "alpha": alpha_hist,
        "delta_theta": dtheta_hist,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CSV loader (for data exported by PhaseBattery::write_debug_csv())
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> Optional[dict]:
    """
    Load a CSV written by PhaseBattery::write_debug_csv().
    Columns: step, E, R, alpha
    Returns the same dict shape as simulate_phase_battery(), with empty
    delta_theta (node-level data is not written by write_debug_csv).
    Returns None if the file cannot be parsed.
    """
    if not os.path.isfile(path):
        return None
    iterations, E_hist, R_hist, alpha_hist = [], [], [], []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iterations.append(int(row["step"]))
                E_hist.append(float(row["E"]))
                R_hist.append(float(row["R"]))
                alpha_hist.append(float(row["alpha"]))
    except (OSError, ValueError, KeyError) as exc:
        print(f"[warn] Could not parse {path}: {exc}", file=sys.stderr)
        return None
    n = len(iterations)
    return {
        "iterations": iterations,
        "E": E_hist,
        "R": R_hist,
        "alpha": alpha_hist,
        "delta_theta": [[] for _ in range(n)],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure builders
# ──────────────────────────────────────────────────────────────────────────────

def build_3d_phase_space(data: dict) -> go.Figure:
    """3-D scatter plot of E(t), R(t), α(t) coloured by iteration."""
    iters = data["iterations"]
    E = data["E"]
    R = data["R"]
    alpha = data["alpha"]

    scatter = go.Scatter3d(
        x=E,
        y=R,
        z=alpha,
        mode="lines+markers",
        marker=dict(
            size=5,
            color=iters,
            colorscale="Viridis",
            colorbar=dict(title="Iteration t", len=0.6),
            showscale=True,
        ),
        line=dict(color="rgba(100,100,200,0.5)", width=2),
        text=[
            f"t={t}<br>E={e:.4f}<br>R={r:.4f}<br>α={a:.4f}"
            for t, e, r, a in zip(iters, E, R, alpha)
        ],
        hovertemplate="%{text}<extra></extra>",
        name="trajectory",
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title="3D Phase Space — E(t) · R(t) · α(t) Trajectory",
        scene=dict(
            xaxis_title="Frustration E(t)",
            yaxis_title="Coherence R(t)",
            zaxis_title="Adaptive α(t)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
    )
    return fig


def build_heatmap(data: dict) -> go.Figure:
    """2-D heatmap of per-node δθ_j (rows = time steps, cols = nodes)."""
    dtheta = data["delta_theta"]

    # Filter out empty rows (CSV-loaded data has no node deviations)
    filled = [row for row in dtheta if len(row) > 0]
    if not filled:
        fig = go.Figure()
        fig.update_layout(
            title="Node Deviations δθ_j — unavailable (CSV mode has no node data)",
            height=400,
        )
        return fig

    z = np.array(filled)          # shape (T, N)
    T, N = z.shape
    heatmap = go.Heatmap(
        z=z,
        colorscale="RdBu",
        zmid=0.0,
        colorbar=dict(title="δθ_j (rad)"),
        xaxis="x",
        yaxis="y",
        hovertemplate="t=%{y}  node=%{x}<br>δθ=%{z:.4f}<extra></extra>",
    )
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="Node Deviations Heatmap — δθ_j(t) per node",
        xaxis=dict(title="Node index j", dtick=max(1, N // 10)),
        yaxis=dict(title="Iteration t", dtick=max(1, T // 10)),
        height=400,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


def build_time_series(data: dict) -> go.Figure:
    """Overlay time-series of E(t), R(t), α(t) on a shared x-axis."""
    iters = data["iterations"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=data["E"], mode="lines+markers",
                             name="E — Frustration", line=dict(color="#e74c3c")))
    fig.add_trace(go.Scatter(x=iters, y=data["R"], mode="lines+markers",
                             name="R — Coherence", line=dict(color="#2ecc71")))
    fig.add_trace(go.Scatter(x=iters, y=data["alpha"], mode="lines+markers",
                             name="α — Adaptive alpha", line=dict(color="#3498db")))
    fig.update_layout(
        title="Time-Series: E(t), R(t), α(t)",
        xaxis_title="Iteration t",
        yaxis_title="Value",
        legend=dict(orientation="h", y=1.12),
        height=350,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Dash application layout
# ──────────────────────────────────────────────────────────────────────────────

_CSV_PATH: str | None = sys.argv[1] if len(sys.argv) > 1 else None

app = Dash(__name__, title="PhaseBattery Geometric Learning Visualizer")

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "0 auto",
           "padding": "20px"},
    children=[
        html.H1("PhaseBattery — Geometric Learning Dynamics",
                style={"textAlign": "center", "color": "#2c3e50"}),

        html.P(
            "Interactive prototype visualizing the relationship between frustration E(t), "
            "coherence R(t), and adaptive α(t) as described in PR #51 "
            "(ohm_coherence_duality.hpp — adaptive-α feedback_step).",
            style={"textAlign": "center", "color": "#7f8c8d"},
        ),

        # ── Simulation parameters ──────────────────────────────────────────
        html.Div(
            style={"background": "#f8f9fa", "borderRadius": "8px",
                   "padding": "20px", "marginBottom": "20px",
                   "border": "1px solid #dee2e6"},
            children=[
                html.H3("Simulation Parameters", style={"marginTop": 0, "color": "#495057"}),
                html.Div(
                    style={"display": "grid",
                           "gridTemplateColumns": "1fr 1fr",
                           "gap": "30px"},
                    children=[
                        html.Div([
                            html.Label("c₁ — frustration-decay sensitivity",
                                       style={"fontWeight": "bold"}),
                            html.Div(id="c1-display",
                                     style={"color": "#e74c3c", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="c1-slider",
                                min=-1.0, max=1.0, step=0.05,
                                value=0.2,
                                marks={v: str(v) for v in [-1.0, -0.5, 0.0, 0.5, 1.0]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]),
                        html.Div([
                            html.Label("c₂ — coherence-gain sensitivity",
                                       style={"fontWeight": "bold"}),
                            html.Div(id="c2-display",
                                     style={"color": "#2ecc71", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="c2-slider",
                                min=-1.0, max=1.0, step=0.05,
                                value=0.3,
                                marks={v: str(v) for v in [-1.0, -0.5, 0.0, 0.5, 1.0]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]),
                        html.Div([
                            html.Label("EMA gain g = G_eff",
                                       style={"fontWeight": "bold"}),
                            html.Div(id="gain-display",
                                     style={"color": "#8e44ad", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="gain-slider",
                                min=0.05, max=0.95, step=0.05,
                                value=0.35,
                                marks={v: str(round(v, 2))
                                       for v in [0.05, 0.25, 0.5, 0.75, 0.95]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]),
                        html.Div([
                            html.Label("Base α (before adaptive tuning)",
                                       style={"fontWeight": "bold"}),
                            html.Div(id="alpha-display",
                                     style={"color": "#3498db", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="alpha-slider",
                                min=0.0, max=1.0, step=0.05,
                                value=0.5,
                                marks={v: str(round(v, 2))
                                       for v in [0.0, 0.25, 0.5, 0.75, 1.0]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]),
                    ],
                ),
            ],
        ),

        # ── Plots ──────────────────────────────────────────────────────────
        dcc.Graph(id="time-series-graph"),
        dcc.Graph(id="phase-space-3d"),
        dcc.Graph(id="heatmap-graph"),

        # ── Footer ─────────────────────────────────────────────────────────
        html.Hr(),
        html.P(
            "Simulating PhaseBattery::feedback_step() with adaptive-α "
            f"{'(simulated data)' if _CSV_PATH is None else f'+ CSV: {_CSV_PATH}'}. "
            "Adjust c₁ / c₂ sliders to see how α sensitivity shapes the trajectories.",
            style={"color": "#95a5a6", "textAlign": "center", "fontSize": "13px"},
        ),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

@callback(
    Output("c1-display", "children"),
    Output("c2-display", "children"),
    Output("gain-display", "children"),
    Output("alpha-display", "children"),
    Output("time-series-graph", "figure"),
    Output("phase-space-3d", "figure"),
    Output("heatmap-graph", "figure"),
    Input("c1-slider", "value"),
    Input("c2-slider", "value"),
    Input("gain-slider", "value"),
    Input("alpha-slider", "value"),
)
def update_all(c1: float, c2: float, gain: float, alpha_base: float):
    """Re-simulate (or reload CSV) and refresh all three plots."""
    if _CSV_PATH is not None:
        data = load_csv(_CSV_PATH)
        if data is None:
            # Fall back to simulation if CSV failed
            data = simulate_phase_battery(gain=gain, alpha_base=alpha_base,
                                          c1=c1, c2=c2)
    else:
        data = simulate_phase_battery(gain=gain, alpha_base=alpha_base,
                                      c1=c1, c2=c2)

    c1_lbl  = f"c₁ = {c1:.2f}"
    c2_lbl  = f"c₂ = {c2:.2f}"
    g_lbl   = f"g = {gain:.2f}"
    alp_lbl = f"α_base = {alpha_base:.2f}"

    return (
        c1_lbl,
        c2_lbl,
        g_lbl,
        alp_lbl,
        build_time_series(data),
        build_3d_phase_space(data),
        build_heatmap(data),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PhaseBattery Geometric Learning Visualizer                 ║")
    print("║                                                              ║")
    print("║  Open http://127.0.0.1:8050 in your browser.                ║")
    print("║  Adjust c₁/c₂ sliders to tune adaptive-α sensitivity.      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if _CSV_PATH:
        print(f"  Loading CSV: {_CSV_PATH}")
    else:
        print("  Using simulated PhaseBattery dynamics (no CSV provided).")
    print()
    app.run(debug=False)
