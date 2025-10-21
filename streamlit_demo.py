"""
Senga SDE Demo Dashboard - Fixed Version
Interactive demonstration of Sequential Decision Analytics with proper data persistence
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
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

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
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
    .waiting-box {
        background-color: #e7f3ff;
        border-left: 4px solid #0056b3;
        padding: 1rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state (minimal - data comes from database)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ============= API Helper Functions =============

def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_system_status() -> Dict:
    """Get current system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def create_order(order_data: Dict) -> Dict:
    """Create a new order via API - persists to database"""
    try:
        response = requests.post(f"{API_BASE_URL}/orders", json=order_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def get_pending_orders() -> List[Dict]:
    """Get all pending orders from database"""
    try:
        response = requests.get(f"{API_BASE_URL}/orders/pending", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Failed to load orders: {e}")
        return []

def trigger_consolidation_cycle(force_dispatch: bool = False, context: Dict = None) -> Dict:
    """Trigger consolidation cycle - processes ALL pending orders"""
    try:
        request_data = {
            "force_dispatch": force_dispatch,
            "context": context or {}
        }
        response = requests.post(
            f"{API_BASE_URL}/decisions/consolidation-cycle",
            json=request_data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

# ============= Sidebar =============

st.sidebar.markdown("# 🚚 Senga SDE")
st.sidebar.markdown("### Sequential Decision Analytics")

# API Status
api_status = check_api_health()
if api_status:
    st.sidebar.success("✓ API Online")
    status = get_system_status()
    if status:
        st.sidebar.metric("Pending Orders", status.get('pending_orders', 0))
        st.sidebar.metric("Available Vehicles", status.get('available_vehicles', 0))
        st.sidebar.metric("Active Routes", status.get('active_routes', 0))
else:
    st.sidebar.error("✗ API Offline")
    st.sidebar.warning("Start backend with: `python -m src.api.main`")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# Page selection
page = st.sidebar.radio(
    "Navigation",
    ["📦 Order Management", "🧠 Consolidation Engine", "📊 Analytics"]
)

# ============= PAGE 1: Order Management =============

if page == "📦 Order Management":
    st.markdown('<div class="main-header">📦 Order Management</div>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ API not available. Please start the backend service.")
        st.info("Run: `python -m src.api.main` in project root")
        st.stop()
    
    # Order creation form
    with st.expander("➕ Create New Order", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pickup Details")
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
            st.subheader("Delivery Details")
            delivery_options = {
                "Nakuru Town (Kenyatta Avenue)": {"lat": -0.3031, "lng": 36.0800},
                "Eldoret (Uganda Road)": {"lat": 0.5143, "lng": 35.2698},
                "Kitale Town": {"lat": 1.0157, "lng": 35.0062},
                "Thika Town": {"lat": -1.0332, "lng": 37.0690},
                "Kiambu Town": {"lat": -1.1714, "lng": 36.8356},
                "Mombasa City": {"lat": -4.0435, "lng": 39.6682}
            }
            
            delivery_location = st.selectbox("Delivery Location", list(delivery_options.keys()))
            delivery_coords = delivery_options[delivery_location]
            
            package_weight = st.number_input("Package Weight (kg)", min_value=0.5, max_value=5000.0, value=10.0)
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
            
            with st.spinner("Creating order and saving to database..."):
                result = create_order(order_data)
                
                if "error" in result:
                    st.error(f"Failed to create order: {result['error']}")
                else:
                    st.success(f"✅ Order {result.get('order_id')} created and saved to database!")
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()
    
    # Display pending orders from database
    st.markdown("### 📋 Pending Orders (from Database)")
    st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
    pending_orders = get_pending_orders()
    
    if pending_orders:
        # Create DataFrame for display
        orders_df = pd.DataFrame([
            {
                "Order ID": o.get("order_id", "N/A")[:12],
                "Customer": o.get("customer_name", "N/A"),
                "Pickup": o.get('pickup_location', {}).get('address', 'N/A')[:25],
                "Delivery": o.get('delivery_location', {}).get('address', 'N/A')[:25],
                "Weight (kg)": f"{o.get('package_weight', 0):.1f}",
                "Priority": o.get("priority", "standard").upper(),
                "Status": o.get("status", "pending").upper(),
                "Hours to Deadline": f"{o.get('time_to_deadline_hours', 0):.1f}"
            }
            for o in pending_orders
        ])
        
        # Color code by priority
        def highlight_priority(row):
            if row['Priority'] == 'EMERGENCY':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Priority'] == 'URGENT':
                return ['background-color: #fff3cd'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = orders_df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", len(pending_orders))
        with col2:
            emergency_count = sum(1 for o in pending_orders if o.get('priority') == 'emergency')
            st.metric("Emergency", emergency_count)
        with col3:
            total_weight = sum(o.get('package_weight', 0) for o in pending_orders)
            st.metric("Total Weight", f"{total_weight:.1f} kg")
        with col4:
            avg_deadline = sum(o.get('time_to_deadline_hours', 0) for o in pending_orders) / len(pending_orders)
            st.metric("Avg Time to Deadline", f"{avg_deadline:.1f} hrs")
    else:
        st.info("📭 No pending orders. Create some orders to get started!")

# ============= PAGE 2: Consolidation Engine =============

elif page == "🧠 Consolidation Engine":
    st.markdown('<div class="main-header">🧠 Consolidation Decision Engine</div>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ API not available. Please start the backend service.")
        st.stop()
    
    st.markdown("""
    This page runs the **Sequential Decision Analytics** engine on ALL pending orders.
    The system uses PFA/CFA/VFA/DLA to decide which orders to batch and dispatch vs. which to wait for consolidation.
    """)
    
    # Get pending orders count
    pending_orders = get_pending_orders()
    
    if not pending_orders:
        st.warning("No pending orders to process. Go to Order Management to create orders first.")
        st.stop()
    
    st.info(f"📦 Currently **{len(pending_orders)}** orders in pending queue ready for consolidation analysis")
    
    # Context configuration
    st.markdown("### ⚙️ Decision Context")
    col1, col2 = st.columns(2)
    
    with col1:
        time_of_day = st.selectbox("Time of Day", ["morning_rush", "midday", "evening_rush", "night"])
        traffic_level = st.slider("Traffic Level", 0.0, 1.0, 0.5)
    
    with col2:
        weather = st.selectbox("Weather", ["clear", "rain", "heavy_rain"])
        force_dispatch = st.checkbox("Force Dispatch (ignore utilization threshold)", value=False)
    
    context = {
        "time_of_day": time_of_day,
        "traffic_level": traffic_level,
        "weather": weather
    }
    
    # Trigger consolidation cycle
    st.markdown("### 🎯 Run Consolidation Cycle")
    
    if st.button("🚀 Run Consolidation Cycle", type="primary", use_container_width=True):
        with st.spinner(f"Running Sequential Decision Engine on {len(pending_orders)} orders..."):
            result = trigger_consolidation_cycle(force_dispatch=force_dispatch, context=context)
            
            if "error" in result:
                st.error(f"❌ Consolidation cycle failed: {result['error']}")
            else:
                st.success("✅ Consolidation cycle completed!")
                
                # Display results
                st.markdown("### 📊 Consolidation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Orders Processed", result.get('total_pending_orders', 0))
                with col2:
                    st.metric("✅ Dispatched", result.get('orders_dispatched', 0))
                with col3:
                    st.metric("⏳ Waiting", result.get('orders_waiting', 0))
                with col4:
                    st.metric("🚚 Batches Created", result.get('batches_created', 0))
                
                # Function class used
                st.markdown("#### 🧠 Decision Function")
                function_used = result.get('function_class_used', 'unknown').upper()
                function_colors = {
                    "PFA": "#dc3545",
                    "CFA": "#007bff",
                    "VFA": "#28a745",
                    "DLA": "#ffc107"
                }
                
                st.markdown(f"""
                <div style="border-left: 4px solid {function_colors.get(function_used, '#6c757d')}; padding-left: 1rem; margin: 1rem 0;">
                <strong>Function Class: {function_used}</strong><br>
                <em>{result.get('reasoning', 'No reasoning provided')}</em>
                </div>
                """, unsafe_allow_html=True)
                
                # Dispatched batches
                if result.get('batches_created', 0) > 0:
                    st.markdown("#### ✅ Dispatched Batches")
                    for i, batch in enumerate(result.get('dispatched_batches', []), 1):
                        with st.expander(f"Batch #{i} - Vehicle {batch.get('vehicle_id', 'N/A')[:12]}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Orders:** {len(batch.get('shipments', []))}")
                                st.write(f"**Distance:** {batch.get('estimated_distance_km', 0):.1f} km")
                            with col2:
                                st.write(f"**Duration:** {batch.get('estimated_duration_hours', 0):.1f} hours")
                                st.write(f"**Stops:** {len(batch.get('route', []))}")
                            
                            if batch.get('shipments'):
                                st.write("**Order IDs:**")
                                st.write(", ".join([s[:12] for s in batch.get('shipments', [])]))
                
                # Waiting orders
                if result.get('orders_waiting', 0) > 0:
                    st.markdown("#### ⏳ Orders Waiting for Consolidation")
                    waiting_df = pd.DataFrame([
                        {
                            "Order ID": o.get("order_id", "N/A")[:12],
                            "Customer": o.get("customer_name", "N/A"),
                            "Weight (kg)": f"{o.get('package_weight', 0):.1f}",
                            "Priority": o.get("priority", "standard").upper(),
                            "Hours to Deadline": f"{o.get('time_to_deadline_hours', 0):.1f}"
                        }
                        for o in result.get('waiting_orders', [])
                    ])
                    st.dataframe(waiting_df, use_container_width=True)
                    
                    st.info("💡 These orders are waiting because dispatching them now would result in low vehicle utilization. They will be included in the next consolidation cycle.")
                
                # Refresh orders
                st.session_state.last_refresh = datetime.now()

# ============= PAGE 3: Analytics =============

elif page == "📊 Analytics":
    st.markdown('<div class="main-header">📊 System Analytics</div>', unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ API not available. Please start the backend service.")
        st.stop()
    
    st.info("Analytics dashboard coming soon - will show learning metrics, utilization trends, and decision history")
    
    # Placeholder for future analytics
    st.markdown("### Planned Features")
    st.markdown("""
    - **VFA Learning Progress**: Track how the value function improves over time
    - **Utilization Trends**: Vehicle and route utilization over time
    - **Decision History**: Audit trail of all consolidation decisions
    - **Cost Optimization**: Fuel costs, driver costs, penalty costs over time
    - **SLA Compliance**: On-time delivery rates by priority level
    """)

# Footer
st.markdown("---")
st.markdown("**Senga SDE** | Sequential Decision Analytics for African Logistics | v1.0.0")