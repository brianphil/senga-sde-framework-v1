"""
Senga SDE Demo Dashboard
Interactive demonstration of Sequential Decision Analytics for African logistics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Senga SDE Demo",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .decision-box {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'demo_initialized' not in st.session_state:
    st.session_state.demo_initialized = False
    st.session_state.current_orders = []
    st.session_state.decision_history = []
    st.session_state.learning_metrics = []

def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def initialize_demo_data():
    """Load pre-configured Kenyan demo data"""
    try:
        response = requests.post(f"{API_BASE_URL}/demo/initialize")
        if response.status_code == 200:
            st.session_state.demo_initialized = True
            return True, "Demo data initialized successfully"
        else:
            return False, f"Failed to initialize: {response.text}"
    except Exception as e:
        return False, f"API Error: {str(e)}"

def create_order(order_data: Dict) -> Dict:
    """Create a new order via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/orders", json=order_data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def get_decision(order_id: str, context: Dict) -> Dict:
    """Get routing decision from SDE"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/decisions/route",
            json={"order_id": order_id, "context": context}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def get_learning_metrics() -> Dict:
    """Fetch VFA learning metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/learning")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        return {}

def get_system_state() -> Dict:
    """Get current system state"""
    try:
        response = requests.get(f"{API_BASE_URL}/state/current")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        return {}

# Sidebar navigation
st.sidebar.markdown("# 🚚 Senga SDE")
st.sidebar.markdown("**Sequential Decision Analytics**")
st.sidebar.markdown("for African Logistics")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📦 Live Orders", "🧠 Decision Engine", "📊 Learning Metrics", "🔧 System Config"]
)

# API Health Check
api_status = check_api_health()
if api_status:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.markdown("Start the API with:")
    st.sidebar.code("python src/api/main.py", language="bash")

# Main content based on selected page
if page == "🏠 Overview":
    st.markdown('<div class="main-header">Senga Sequential Decision Engine</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Real-time route optimization for African logistics challenges
    
    **What makes Senga different:**
    - 🌍 Designed for African infrastructure reality (unreliable GPS, informal addresses)
    - 🔄 Real-time learning from every delivery
    - 🎯 Multi-city direct delivery optimization (no warehousing)
    - 📱 Offline-capable decision engine
    - 🚦 Cultural pattern adaptation (traffic, availability windows)
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Powell's SDA Framework")
        st.markdown("""
        **4 Function Classes:**
        - **PFA**: Emergency dispatch rules
        - **CFA**: MIP optimization (OR-Tools)
        - **VFA**: 25-feature TD learning
        - **DLA**: Monte Carlo lookahead
        """)
    
    with col2:
        st.markdown("### Mathematical Rigor")
        st.markdown("""
        **No Mock Algorithms:**
        - Actual TD(λ) value function updates
        - Real mixed-integer programming
        - True multi-scenario simulations
        - Measurable convergence metrics
        """)
    
    with col3:
        st.markdown("### African Context")
        st.markdown("""
        **Real Challenges:**
        - Nairobi → Nakuru → Eldoret routes
        - "Near the big tree" addresses
        - 2-hour traffic delays
        - Vehicle breakdown cascades
        - Customer availability patterns
        """)
    
    st.markdown("---")
    
    # Demo initialization
    st.markdown("### 🎬 Demo Setup")
    
    if not st.session_state.demo_initialized:
        if st.button("🚀 Initialize Demo with Kenyan Data", type="primary"):
            with st.spinner("Loading realistic Kenyan logistics scenarios..."):
                success, message = initialize_demo_data()
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.error(message)
    else:
        st.success("✅ Demo data loaded and ready")
        
        # Show current system state
        system_state = get_system_state()
        if system_state:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active Orders", system_state.get("active_orders", 0))
            with col2:
                st.metric("Available Drivers", system_state.get("available_drivers", 0))
            with col3:
                st.metric("Completed Today", system_state.get("completed_today", 0))
            with col4:
                st.metric("Learning Iterations", system_state.get("learning_iterations", 0))

elif page == "📦 Live Orders":
    st.markdown("## 📦 Create & Track Orders")
    
    if not api_status:
        st.error("API not available. Please start the backend service.")
        st.stop()
    
    # Order creation form
    with st.expander("➕ Create New Order", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            pickup_options = {
                "Nairobi CBD (Kenyatta Avenue)": {"lat": -1.2864, "lng": 36.8172},
                "Westlands, Nairobi": {"lat": -1.2676, "lng": 36.8070},
                "Industrial Area, Nairobi": {"lat": -1.3167, "lng": 36.8500},
                "Nakuru Town": {"lat": -0.3031, "lng": 36.0800},
                "Eldoret Town": {"lat": 0.5143, "lng": 35.2698}
            }
            
            pickup_location = st.selectbox("Pickup Location", list(pickup_options.keys()))
            pickup_coords = pickup_options[pickup_location]
            
            customer_name = st.text_input("Customer Name", "John Mwangi")
            customer_phone = st.text_input("Phone", "+254712345678")
        
        with col2:
            delivery_options = {
                "Nakuru Town (Kenyatta Avenue)": {"lat": -0.3031, "lng": 36.0800},
                "Eldoret (Uganda Road)": {"lat": 0.5143, "lng": 35.2698},
                "Kitale Town": {"lat": 1.0157, "lng": 35.0062},
                "Thika Town": {"lat": -1.0332, "lng": 37.0690},
                "Kiambu Town": {"lat": -1.1714, "lng": 36.8356}
            }
            
            delivery_location = st.selectbox("Delivery Location", list(delivery_options.keys()))
            delivery_coords = delivery_options[delivery_location]
            
            package_weight = st.number_input("Package Weight (kg)", min_value=0.5, max_value=1000.0, value=10.0)
            priority = st.selectbox("Priority", ["standard", "urgent", "emergency"])
        
        if st.button("📝 Create Order", type="primary"):
            order_data = {
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "pickup_location": {
                    "address": pickup_location,
                    "latitude": pickup_coords["lat"],
                    "longitude": pickup_coords["lng"]
                },
                "delivery_location": {
                    "address": delivery_location,
                    "latitude": delivery_coords["lat"],
                    "longitude": delivery_coords["lng"]
                },
                "package_weight": package_weight,
                "priority": priority,
                "created_at": datetime.now().isoformat()
            }
            
            with st.spinner("Creating order..."):
                result = create_order(order_data)
                
                if "error" in result:
                    st.error(f"Failed to create order: {result['error']}")
                else:
                    st.success(f"✅ Order created: {result.get('order_id')}")
                    st.session_state.current_orders.append(result)
    
    # Show current orders
    if st.session_state.current_orders:
        st.markdown("### 📋 Current Orders")
        
        orders_df = pd.DataFrame([
            {
                "Order ID": o.get("order_id", "N/A")[:8],
                "Customer": o.get("customer_name", "N/A"),
                "Route": f"{o.get('pickup_location', {}).get('address', 'N/A')[:20]} → {o.get('delivery_location', {}).get('address', 'N/A')[:20]}",
                "Weight (kg)": o.get("package_weight", 0),
                "Priority": o.get("priority", "standard").upper(),
                "Status": o.get("status", "pending").upper()
            }
            for o in st.session_state.current_orders[-10:]
        ])
        
        st.dataframe(orders_df, use_container_width=True)

elif page == "🧠 Decision Engine":
    st.markdown("## 🧠 Sequential Decision Process")
    
    if not api_status:
        st.error("API not available. Please start the backend service.")
        st.stop()
    
    st.markdown("""
    Watch the SDE make real-time routing decisions using all 4 function classes.
    Each decision shows the mathematical reasoning behind driver assignment and route optimization.
    """)
    
    # Select an order to route
    if st.session_state.current_orders:
        order_to_route = st.selectbox(
            "Select Order for Routing Decision",
            range(len(st.session_state.current_orders)),
            format_func=lambda i: f"{st.session_state.current_orders[i].get('order_id', 'N/A')[:8]} - {st.session_state.current_orders[i].get('customer_name', 'N/A')}"
        )
        
        selected_order = st.session_state.current_orders[order_to_route]
        
        # Context configuration
        col1, col2 = st.columns(2)
        with col1:
            time_of_day = st.selectbox("Time of Day", ["morning_rush", "midday", "evening_rush", "night"])
            traffic_level = st.slider("Traffic Level", 0.0, 1.0, 0.5)
        
        with col2:
            weather = st.selectbox("Weather", ["clear", "rain", "heavy_rain"])
            available_drivers = st.number_input("Available Drivers", 1, 20, 5)
        
        context = {
            "time_of_day": time_of_day,
            "traffic_level": traffic_level,
            "weather": weather,
            "available_drivers": available_drivers
        }
        
        if st.button("🎯 Get Routing Decision", type="primary"):
            with st.spinner("Running Sequential Decision Engine..."):
                decision = get_decision(selected_order.get("order_id"), context)
                
                if "error" in decision:
                    st.error(f"Decision failed: {decision['error']}")
                else:
                    st.session_state.decision_history.append(decision)
                    
                    # Display decision breakdown
                    st.markdown("### 📊 Decision Breakdown")
                    
                    # Meta-controller selection
                    st.markdown("#### 1️⃣ Meta-Controller Selection")
                    selected_function = decision.get("function_used", "unknown")
                    function_reason = decision.get("selection_reason", "No reason provided")
                    
                    function_colors = {
                        "PFA": "#dc3545",
                        "CFA": "#007bff", 
                        "VFA": "#28a745",
                        "DLA": "#ffc107"
                    }
                    
                    st.markdown(f"""
                    <div class="decision-box" style="border-left-color: {function_colors.get(selected_function, '#6c757d')}">
                    <strong>Selected Function Class: {selected_function}</strong><br>
                    {function_reason}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Decision details
                    st.markdown("#### 2️⃣ Routing Decision")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Assigned Driver", decision.get("driver_id", "N/A")[:8])
                    with col2:
                        st.metric("Estimated Time", f"{decision.get('estimated_time_minutes', 0):.0f} min")
                    with col3:
                        st.metric("Confidence", f"{decision.get('confidence', 0):.1%}")
                    
                    # Route details
                    if "route" in decision:
                        st.markdown("#### 3️⃣ Optimized Route")
                        route_data = decision["route"]
                        
                        st.markdown(f"""
                        - **Distance**: {route_data.get('distance_km', 0):.1f} km
                        - **Base Time**: {route_data.get('base_time_minutes', 0):.0f} minutes
                        - **Traffic Adjustment**: +{route_data.get('traffic_delay_minutes', 0):.0f} minutes
                        - **Total Cost**: KES {route_data.get('total_cost', 0):.2f}
                        """)
                    
                    # VFA features (if VFA was used)
                    if selected_function == "VFA" and "vfa_features" in decision:
                        with st.expander("📊 VFA Features (25-dimensional state)"):
                            features = decision["vfa_features"]
                            features_df = pd.DataFrame([
                                {"Feature": k, "Value": f"{v:.4f}"} 
                                for k, v in features.items()
                            ])
                            st.dataframe(features_df, use_container_width=True)
                    
                    # DLA scenarios (if DLA was used)
                    if selected_function == "DLA" and "scenarios_evaluated" in decision:
                        with st.expander(f"🎲 Monte Carlo Scenarios ({decision['scenarios_evaluated']} evaluated)"):
                            st.markdown(f"""
                            - **Best Case**: {decision.get('best_case_value', 0):.2f}
                            - **Expected Value**: {decision.get('expected_value', 0):.2f}
                            - **Worst Case**: {decision.get('worst_case_value', 0):.2f}
                            - **Variance**: {decision.get('scenario_variance', 0):.4f}
                            """)
                    
                    st.success("✅ Decision recorded and driver notified")
    
    else:
        st.info("📦 Create some orders first to see routing decisions")
    
    # Decision history
    if st.session_state.decision_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Decisions")
        
        history_df = pd.DataFrame([
            {
                "Timestamp": d.get("timestamp", "N/A")[:19],
                "Function": d.get("function_used", "N/A"),
                "Driver": d.get("driver_id", "N/A")[:8],
                "Time (min)": f"{d.get('estimated_time_minutes', 0):.0f}",
                "Confidence": f"{d.get('confidence', 0):.1%}"
            }
            for d in st.session_state.decision_history[-10:]
        ])
        
        st.dataframe(history_df, use_container_width=True)

elif page == "📊 Learning Metrics":
    st.markdown("## 📊 Value Function Learning")
    
    if not api_status:
        st.error("API not available. Please start the backend service.")
        st.stop()
    
    st.markdown("""
    Real-time visualization of the VFA learning from every delivery.
    This proves the system is actually learning, not using static rules.
    """)
    
    # Fetch learning metrics
    if st.button("🔄 Refresh Metrics"):
        with st.spinner("Fetching learning metrics..."):
            metrics = get_learning_metrics()
            if metrics:
                st.session_state.learning_metrics = metrics
    
    if st.session_state.learning_metrics:
        metrics = st.session_state.learning_metrics
        
        # Key learning indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "TD Errors (Last 100)", 
                f"{metrics.get('td_error_mean', 0):.4f}",
                delta=f"{metrics.get('td_error_trend', 0):.4f}"
            )
        
        with col2:
            st.metric(
                "Learning Rate",
                f"{metrics.get('current_learning_rate', 0):.6f}"
            )
        
        with col3:
            st.metric(
                "Total Updates",
                metrics.get('total_updates', 0)
            )
        
        with col4:
            st.metric(
                "Convergence",
                f"{metrics.get('convergence_score', 0):.1%}"
            )
        
        # TD Error plot
        if "td_error_history" in metrics:
            st.markdown("### 📉 TD Error Convergence")
            
            td_errors = metrics["td_error_history"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=td_errors,
                mode='lines',
                name='TD Error',
                line=dict(color='#1f77b4')
            ))
            fig.update_layout(
                title="Temporal Difference Error Over Time",
                xaxis_title="Update Number",
                yaxis_title="TD Error",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if "feature_weights" in metrics:
            st.markdown("### 🎯 Feature Importance")
            
            weights = metrics["feature_weights"]
            weights_df = pd.DataFrame([
                {"Feature": k, "Weight": v}
                for k, v in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
            ])
            
            fig = px.bar(
                weights_df.head(15),
                x="Weight",
                y="Feature",
                orientation='h',
                title="Top 15 VFA Feature Weights"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance improvement
        if "performance_history" in metrics:
            st.markdown("### 📈 Decision Quality Improvement")
            
            perf = metrics["performance_history"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=perf,
                mode='lines+markers',
                name='Average Reward',
                line=dict(color='#28a745')
            ))
            fig.update_layout(
                title="Average Reward Per Decision (Rolling Window)",
                xaxis_title="Time Period",
                yaxis_title="Average Reward",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📊 No learning metrics available yet. Make some routing decisions first!")

elif page == "🔧 System Config":
    st.markdown("## 🔧 System Configuration")
    
    if not api_status:
        st.error("API not available. Please start the backend service.")
        st.stop()
    
    st.markdown("View and modify system parameters in real-time.")
    
    # VFA Configuration
    with st.expander("🧠 Value Function Approximation (VFA)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.slider("Learning Rate (α)", 0.0001, 0.1, 0.001, format="%.4f")
            discount_factor = st.slider("Discount Factor (γ)", 0.8, 0.99, 0.95, format="%.2f")
            lambda_param = st.slider("Eligibility Trace (λ)", 0.0, 1.0, 0.7, format="%.2f")
        
        with col2:
            epsilon = st.slider("Exploration (ε)", 0.0, 0.3, 0.1, format="%.2f")
            batch_size = st.number_input("Batch Size", 16, 256, 32)
            update_frequency = st.number_input("Update Frequency", 1, 100, 10)
    
    # CFA Configuration
    with st.expander("🎯 Cost Function Approximation (CFA)"):
        col1, col2 = st.columns(2)
        
        with col1:
            fuel_cost_per_km = st.number_input("Fuel Cost (KES/km)", 1.0, 50.0, 15.0)
            driver_cost_per_hour = st.number_input("Driver Cost (KES/hr)", 100.0, 1000.0, 300.0)
            vehicle_depreciation = st.number_input("Vehicle Depreciation (KES/km)", 1.0, 20.0, 5.0)
        
        with col2:
            late_penalty_per_hour = st.number_input("Late Penalty (KES/hr)", 100.0, 5000.0, 500.0)
            priority_bonus = st.number_input("Priority Bonus (KES)", 0.0, 1000.0, 200.0)
            max_route_time = st.number_input("Max Route Time (hours)", 1, 24, 8)
    
    # DLA Configuration
    with st.expander("🎲 Direct Lookahead Approximation (DLA)"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_scenarios = st.slider("Monte Carlo Scenarios", 10, 1000, 100)
            lookahead_horizon = st.slider("Lookahead Horizon (hours)", 1, 24, 4)
        
        with col2:
            traffic_uncertainty = st.slider("Traffic Uncertainty", 0.0, 1.0, 0.3)
            demand_uncertainty = st.slider("Demand Uncertainty", 0.0, 1.0, 0.2)
    
    # Meta-Controller Thresholds
    with st.expander("🎛️ Meta-Controller Thresholds"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PFA Activation:**")
            pfa_urgency = st.slider("Urgency Threshold", 0.0, 1.0, 0.8)
            pfa_driver_availability = st.slider("Min Driver % for PFA", 0.0, 0.5, 0.2)
        
        with col2:
            st.markdown("**DLA Activation:**")
            dla_uncertainty = st.slider("Uncertainty Threshold", 0.0, 1.0, 0.6)
            dla_horizon = st.slider("Min Lookahead Need (hrs)", 1, 12, 4)
    
    if st.button("💾 Save Configuration", type="primary"):
        config_data = {
            "vfa": {
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "lambda": lambda_param,
                "epsilon": epsilon,
                "batch_size": batch_size,
                "update_frequency": update_frequency
            },
            "cfa": {
                "fuel_cost_per_km": fuel_cost_per_km,
                "driver_cost_per_hour": driver_cost_per_hour,
                "vehicle_depreciation": vehicle_depreciation,
                "late_penalty_per_hour": late_penalty_per_hour,
                "priority_bonus": priority_bonus,
                "max_route_time": max_route_time
            },
            "dla": {
                "num_scenarios": num_scenarios,
                "lookahead_horizon": lookahead_horizon,
                "traffic_uncertainty": traffic_uncertainty,
                "demand_uncertainty": demand_uncertainty
            },
            "meta": {
                "pfa_urgency_threshold": pfa_urgency,
                "pfa_driver_threshold": pfa_driver_availability,
                "dla_uncertainty_threshold": dla_uncertainty,
                "dla_horizon_threshold": dla_horizon
            }
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/config/update", json=config_data)
            if response.status_code == 200:
                st.success("✅ Configuration saved successfully")
            else:
                st.error(f"Failed to save: {response.text}")
        except Exception as e:
            st.error(f"API Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <strong>Senga Sequential Decision Engine</strong><br>
    Powered by Powell's SDA Framework | Designed for African Logistics Reality<br>
    <em>Real algorithms. Real learning. Real innovation.</em>
</div>
""", unsafe_allow_html=True)