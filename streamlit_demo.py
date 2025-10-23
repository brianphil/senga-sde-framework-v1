# streamlit_demo_fixed.py
# FIXES: KeyError for 'pending_shipments' and 'engine_status'

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Senga SDE - Sequential Decision Engine",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Helper Functions
@st.cache_data(ttl=5)
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=5)
def get_system_status():
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

@st.cache_data(ttl=5)
def get_recent_cycles(n=20):
    try:
        response = requests.get(f"{API_BASE_URL}/cycles/recent?n={n}", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Handle both list and dict responses
            if isinstance(data, list):
                return data
            return []
        return []
    except:
        return []

def start_autonomous_mode(interval_minutes):
    try:
        response = requests.post(
            f"{API_BASE_URL}/autonomous/start?cycle_interval_minutes={interval_minutes}",
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

def stop_autonomous_mode():
    try:
        response = requests.post(f"{API_BASE_URL}/autonomous/stop", timeout=2)
        return response.status_code == 200
    except:
        return False

def trigger_single_cycle():
    try:
        response = requests.post(f"{API_BASE_URL}/trigger-cycle", timeout=2)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

# Sidebar
st.sidebar.markdown("# 🚚 Senga SDE")
st.sidebar.markdown("**Sequential Decision Analytics**")
st.sidebar.markdown("---")

# API Status Check
api_online = check_api_health()
if api_online:
    st.sidebar.markdown("### Status")
    st.sidebar.markdown('<p class="status-online">✅ API Online</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown("### Status")
    st.sidebar.markdown('<p class="status-offline">❌ API Offline</p>', unsafe_allow_html=True)
    st.sidebar.error("Start the API server:")
    st.sidebar.code("python src/api/main.py", language="bash")
    st.stop()

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Live Decisions", "⚙️ Controls"]
)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

if auto_refresh:
    time.sleep(30)
    st.rerun()

# Main Content
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">Senga Sequential Decision Engine</div>', unsafe_allow_html=True)
    
    # System Status
    status = get_system_status()
    
    if status:
        # Key Metrics Row - FIXED: Use .get() with defaults
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pending = status.get('pending_shipments', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{pending}</h3>
                <p>Pending Shipments</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vehicles = status.get('available_vehicles', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{vehicles}</h3>
                <p>Available Vehicles</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cycles = status.get('current_cycle', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{cycles}</h3>
                <p>Decision Cycles</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            engine_status = status.get('engine_status', 'unknown')
            status_color = "#28a745" if engine_status == "autonomous" else "#ffc107"
            st.markdown(f"""
            <div class="metric-card" style="background: {status_color};">
                <h3>{engine_status.upper()}</h3>
                <p>Engine Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent Activity
        st.markdown("### 📊 Recent Decisions")
        recent_cycles = get_recent_cycles(10)
        
        if recent_cycles and len(recent_cycles) > 0:
            for cycle in recent_cycles[:5]:
                fc = cycle.get('function_class', 'UNKNOWN')
                action = cycle.get('action_type', 'UNKNOWN')
                reward = cycle.get('reward', 0)
                
                color = "#4ecdc4" if fc == 'CFA' else "#ff6b6b" if fc == 'PFA' else "#95e1d3"
                reward_color = "#28a745" if reward > 0 else "#dc3545"
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; 
                     border-left: 4px solid {color}; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong>Cycle {cycle.get('cycle_number', 0)}</strong> - 
                            <span style="color: {color}; font-weight: bold;">{fc}</span> → {action}
                        </div>
                        <div style="color: {reward_color}; font-weight: bold;">
                            {reward:+.0f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No decisions yet. Start autonomous mode or trigger a cycle.")
    
    else:
        st.error("Unable to fetch system status. Check API connection.")

elif page == "📊 Live Decisions":
    st.markdown('<div class="main-header">Live Decision Stream</div>', unsafe_allow_html=True)
    
    n_cycles = st.slider("Number of cycles", 10, 50, 20)
    recent_cycles = get_recent_cycles(n_cycles)
    
    if recent_cycles and len(recent_cycles) > 0:
        for cycle in recent_cycles:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown(f"**Cycle {cycle.get('cycle_number', 0)}**")
            
            with col2:
                fc = cycle.get('function_class', 'UNKNOWN')
                action = cycle.get('action_type', 'UNKNOWN')
                st.markdown(f"**{fc}** → {action}")
            
            with col3:
                reward = cycle.get('reward', 0)
                color = "green" if reward > 0 else "red"
                st.markdown(f"Reward: :{color}[**{reward:+.0f}**]")
            
            st.markdown("---")
        
        # Timeline
        st.markdown("### Decision Timeline")
        df = pd.DataFrame(recent_cycles)
        
        if 'cycle_number' in df.columns and 'reward' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['cycle_number'],
                y=df['reward'],
                mode='lines+markers',
                name='Reward'
            ))
            fig.update_layout(
                xaxis_title="Cycle Number",
                yaxis_title="Reward",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No decisions recorded yet")

elif page == "⚙️ Controls":
    st.markdown('<div class="main-header">System Controls</div>', unsafe_allow_html=True)
    
    status = get_system_status()
    
    if status:
        st.markdown("### 🤖 Autonomous Mode")
        
        # FIXED: Use .get() with default
        engine_status = status.get('engine_status', 'unknown')
        is_autonomous = engine_status == 'autonomous'
        
        if is_autonomous:
            st.success("✅ Autonomous mode is ACTIVE")
            
            if st.button("🛑 Stop Autonomous Mode", type="primary"):
                if stop_autonomous_mode():
                    st.success("Stopped")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to stop")
        else:
            st.info("Autonomous mode is OFF")
            
            interval = st.slider("Decision interval (minutes)", 1, 60, 60)
            
            if st.button("▶️ Start Autonomous Mode", type="primary"):
                if start_autonomous_mode(interval):
                    st.success(f"Started! Decisions every {interval} minutes")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to start")
        
        st.markdown("---")
        st.markdown("### 🔄 Manual Control")
        
        if st.button("⚡ Trigger Single Cycle Now", disabled=is_autonomous):
            with st.spinner("Running..."):
                success, result = trigger_single_cycle()
                if success:
                    st.success("✅ Cycle completed!")
                    st.json(result)
                else:
                    st.error(f"❌ Failed: {result.get('error', 'Unknown')}")
    else:
        st.error("Unable to fetch system status")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d;">
    <strong>Senga Sequential Decision Engine</strong><br>
    <em>Sequential Decision Analytics for African Logistics</em>
</div>
""", unsafe_allow_html=True)