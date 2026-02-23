"""
Neuro-Edge Silicon Lab Console
Hardware-Accurate ReRAM SNN Simulation Dashboard
"""

import sys
import time
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is available
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crossbar import IdealCrossbar
from src.snn import PoissonEncoder, LIFNeuron, SNNNetwork
from src.hardware.energy_estimator import EnergyEstimator
from src.utils.mnist_loader import get_mnist_path, load_mnist_from_path
from src.utils.metrics import compute_accuracy
from src.utils.weight_io import load_weights

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro-Edge | Silicon Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Silicon Lab CSS Injection ────────────────────────────────────────────────
st.markdown("""
<style>
/* Global Styling */
.stApp {
    background-color: #0B0F1A;
}

/* Glassmorphism Cards */
div.stMetric, .stPlotlyChart {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(0, 240, 255, 0.1);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* Pulsing Header */
@keyframes glow {
  0% { text-shadow: 0 0 5px #00F0FF; }
  50% { text-shadow: 0 0 15px #00F0FF; }
  100% { text-shadow: 0 0 5px #00F0FF; }
}

h1 {
    color: #00F0FF !important;
    animation: glow 3s infinite;
    letter-spacing: 1px;
    font-weight: 700 !important;
}

h2, h3 {
    color: #E5E7EB !important;
    font-weight: 600 !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: #0E1424;
    border-right: 1px solid rgba(0, 240, 255, 0.1);
}

/* Metric Tweak */
[data-testid="stMetricValue"] {
    color: #00F0FF !important;
}
</style>
""", unsafe_allow_html=True)

# ── Branding ──────────────────────────────────────────────────────────────────
# stlite version compat: replace .toggle with .checkbox
is_dev = "dev" in str(Path.cwd()) or st.sidebar.checkbox("Force Dev Mode", value=False)
if is_dev:
    st.warning("🔬 DEVELOPMENT MODE | MLC ENGINE ACTIVE")

st.markdown("""
<h1>Neuro-Edge Silicon Lab Console</h1>
<p style='color:#9CA3AF; margin-top:-1rem; font-size:1.1rem;'>
Hardware-Accurate Neuromorphic Simulation Interface v1.0
</p>
""", unsafe_allow_html=True)

# ── Sidebar Control Panel ─────────────────────────────────────────────────────
st.sidebar.header("🕹️ Lab Control Panel")
rows = st.sidebar.slider("Crossbar Rows", 8, 128, 32)
cols = st.sidebar.slider("Crossbar Columns", 8, 128, 32)
timesteps = st.sidebar.slider("Integration Timesteps (ms)", 20, 200, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Neural Engine")
_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "trained_weights.npy"
_weights_available = _WEIGHTS_PATH.exists()

use_trained = st.sidebar.checkbox(
    "Load Trained Weights",
    value=_weights_available,
    disabled=not _weights_available
)

auto_play = st.sidebar.checkbox("Auto-play Inference", value=False)
play_speed = st.sidebar.slider("Cycle Rate (s)", 0.5, 5.0, 2.0) if auto_play else 2.0

st.sidebar.markdown("---")
st.sidebar.subheader("📡 System Status")
st.sidebar.markdown("""
- 🔋 **Power Rail**: Stable (1.2V)
- ⚙️ **Crossbar**: Active
- 🧠 **SNN Engine**: Spiking
- 📡 **Console**: Online
""")
st.sidebar.success("🟢 Core Initialized")

# ── Session State Handling ────────────────────────────────────────────────────
if "mnist_data" not in st.session_state:
    st.session_state.mnist_data = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None
if "evolution_step" not in st.session_state:
    st.session_state.evolution_step = 0.0

# ── Weight Evolution Feature ──────────────────────────────────────────────────
# stlite version compat: replace .toggle with .checkbox
evolution_mode = st.sidebar.checkbox("🔬 Simulate Weight Evolution", value=False)
if evolution_mode:
    evo_speed = st.sidebar.slider("Learning Rate (Sim)", 0.01, 0.20, 0.05)
    if st.session_state.evolution_step < 1.0:
        st.session_state.evolution_step = min(1.0, st.session_state.evolution_step + evo_speed)
        time.sleep(0.1)
        st.rerun()
    if st.sidebar.button("Reset Evolution"):
        st.session_state.evolution_step = 0.0
        st.rerun()

# ── Dashboard Layout ─────────────────────────────────────────────────────────

# Row 1: Key Metrics
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    if evolution_mode:
        # Interpolated accuracy
        target_acc = st.session_state.accuracy if st.session_state.accuracy else 0.82
        current_acc = 0.10 + (target_acc - 0.10) * st.session_state.evolution_step
        st.metric("In-Situ Accuracy", f"{current_acc:.2%}", delta=f"{st.session_state.evolution_step:.1%} Trained")
    else:
        acc_text = f"{st.session_state.accuracy:.2%}" if st.session_state.accuracy else "DORMANT"
        st.metric("Test Accuracy", acc_text, delta="Optimized" if use_trained else "Baseline")
with m_col2:
    st.metric("Matrix Capacity", f"{rows * cols} Cells")
with m_col3:
    st.metric("Pulse Window", f"{timesteps} Timesteps")
with m_col4:
    st.metric("Hardware Status", "LEARNING..." if (evolution_mode and st.session_state.evolution_step < 1.0) else "STABLE")

st.divider()

# Row 2: Matrix & Event Analysis
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📊 Memristive Crossbar Matrix")
    # Base weight generation
    np.random.seed(42)
    W_init = np.random.rand(rows, cols).astype(np.float32) * 0.05
    
    if evolution_mode and _weights_available:
        # Load trained G (or part of it if size mismatch)
        G_trained_full = load_weights(str(_WEIGHTS_PATH))
        # Resize/crop to fit current UI sliders
        G_target = np.zeros((rows, cols))
        r_f, c_f = min(rows, G_trained_full.shape[0]), min(cols, G_trained_full.shape[1])
        G_target[:r_f, :c_f] = G_trained_full[:r_f, :c_f]
        
        # Interpolate
        alpha = st.session_state.evolution_step
        G_display = (1 - alpha) * W_init + alpha * G_target
    else:
        G_display = W_init
        
    # Replace px.imshow with go.Heatmap for stlite compatibility
    fig_heat = go.Figure(data=go.Heatmap(
        z=G_display,
        colorscale="Turbo",
        colorbar=dict(title="G (S)"),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>G: %{z:.4f} S<extra></extra>"
    ))
    fig_heat.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Column",
        yaxis_title="Row",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E5E7EB",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_right:
    st.subheader("⚡ Spike Event Analyzer")
    # Small demo run
    n_in_s, n_out_s = min(32, rows), min(10, cols)
    cb_s = IdealCrossbar(n_in_s, n_out_s)
    
    # Use current evolving weights for the demo raster too
    W_s_init = np.maximum(np.random.randn(n_in_s, n_out_s) * 0.1, 0)
    if evolution_mode and _weights_available:
        G_tr_s = load_weights(str(_WEIGHTS_PATH))
        # Match dimensions for the small subset
        W_s_target = np.zeros((n_in_s, n_out_s))
        r_s, c_s = min(n_in_s, G_tr_s.shape[0]), min(n_out_s, G_tr_s.shape[1])
        W_s_target[:r_s, :c_s] = G_tr_s[:r_s, :c_s]
        W_s = (1 - st.session_state.evolution_step) * W_s_init + st.session_state.evolution_step * W_s_target
    else:
        W_s = W_s_init
        
    cb_s.set_conductance(W_s)
    xs = np.random.rand(n_in_s).astype(np.float32)
    sp_out, _ = SNNNetwork(n_in_s, n_out_s, cb_s.run, encoder=PoissonEncoder(50.0), timesteps=timesteps).forward(xs, seed=42)
    
    t_coords, n_coords = np.where(sp_out > 0)
    fig_raster = go.Figure()
    fig_raster.add_trace(go.Scatter(
        x=t_coords, y=n_coords, mode='markers',
        marker=dict(size=6, color="#00F0FF", opacity=0.8, symbol="square")
    ))
    fig_raster.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time (Timesteps)", yaxis_title="Neuron ID",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E5E7EB",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_raster, use_container_width=True)

# Row 3: Power & Inference Lab
col_pwr, col_inf = st.columns([1, 1.2])

with col_pwr:
    st.subheader("📉 Dynamic Power Profiling")
    v_demo = np.random.rand(rows).astype(np.float32) * 0.5
    energy = EnergyEstimator().energy_crossbar(v_demo, G_display)
    t_curve = np.linspace(0, timesteps, 50)
    p_base = energy * 1e6
    p_curve = p_base + np.sin(t_curve * 0.5) * (p_base * 0.1)
    
    # Replace px.line with go.Scatter for stlite compatibility
    fig_pwr = go.Figure(data=go.Scatter(
        x=t_curve, y=p_curve,
        mode='lines',
        line=dict(color="#7C3AED", width=3)
    ))
    fig_pwr.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time (ms)", yaxis_title="Power (uW)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E5E7EB",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_pwr, use_container_width=True)

with col_inf:
    st.subheader("🎯 Input Stimulus / Inference")
    if not st.session_state.mnist_data:
        if st.button("🔥 BOOT SNN ENGINE", type="primary"):
            path = get_mnist_path()
            X_tr, y_tr, X_te, y_te, p_in = load_mnist_from_path(path, max_test=200)
            st.session_state.mnist_data = (X_te, y_te, p_in)
            st.rerun()
    else:
        X_test, y_test, p_pixels = st.session_state.mnist_data
        
        # In evolution mode, we use the evolving weights even for the main inference showcase
        if evolution_mode and _weights_available:
            alpha = st.session_state.evolution_step
            G_full_trained = load_weights(str(_WEIGHTS_PATH))
            W_m_init = np.maximum(np.random.randn(p_pixels, 10) * 0.01, 0)
            W_m = (1 - alpha) * W_m_init + alpha * G_full_trained
        elif use_trained and _weights_available:
            W_m = load_weights(str(_WEIGHTS_PATH))
            if W_m.shape != (p_pixels, 10):
                W_m = np.maximum(np.random.randn(p_pixels, 10) * 0.01, 0)
        else:
            W_m = np.maximum(np.random.randn(p_pixels, 10) * 0.01, 0)

        cb_m = IdealCrossbar(p_pixels, 10)
        cb_m.set_conductance(W_m)
        snn_m = SNNNetwork(p_pixels, 10, cb_m.run, encoder=PoissonEncoder(100.0), timesteps=timesteps)

        # Session Accuracy Cache
        if st.session_state.accuracy is None:
            with st.spinner("Analyzing Benchmarks..."):
                logits = np.zeros((len(X_test), 10))
                for i in range(len(X_test)):
                    _, tot = snn_m.forward(X_test[i])
                    logits[i] = tot
                st.session_state.accuracy = compute_accuracy(logits, y_test)
                st.rerun()

        # Inference Display
        idx = st.session_state.current_idx
        _, t_spikes = snn_m.forward(X_test[idx])
        pred = int(np.argmax(t_spikes))
        truth = int(y_test[idx])
        
        inf_c1, inf_c2 = st.columns([1, 2])
        with inf_c1:
            img = (X_test[idx].reshape(28, 28) * 255).astype(np.uint8)
            st.image(img, caption=f"Sample ID: #{idx}", use_container_width=True)
            
            # Result Badge
            status_color = "#00FF9C" if pred == truth else "#FFB020"
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.3); padding:10px; border-radius:10px; border-left:4px solid {status_color}'>
                <span style='color:{status_color}; font-size:1.2rem; font-weight:bold;'>DECISION: {pred}</span><br/>
                <span style='color:#9CA3AF;'>EXPECTED: {truth}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with inf_c2:
            # Replace px.bar with go.Bar for stlite compatibility
            labels = [str(i) for i in range(10)]
            fig_bar = go.Figure(data=go.Bar(
                x=labels, y=t_spikes,
                marker_color="#00F0FF"
            ))
            fig_bar.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Class", yaxis_title="Firing Rate",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color="#E5E7EB",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if auto_play:
            time.sleep(play_speed)
            st.session_state.current_idx = (idx + 1) % len(X_test)
            st.rerun()


st.divider()
st.caption("© 2026 Neuro-Edge ReRAM Research Platform | Silicon Lab Console")




