"""
Initialize Senga SDE System
Creates both configuration and state databases with correct schema
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

# Ensure data directory exists
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

config_db_path = data_dir / "senga_config.db"
state_db_path = data_dir / "senga_state.db"

print("=" * 60)
print("Senga SDE System Initialization")
print("=" * 60)
print()

# ============================================================
# 1. Configuration Database (senga_config.db)
# ============================================================

print("📋 Creating configuration database...")

# Remove old database if exists
if config_db_path.exists():
    print(f"   Removing old config database: {config_db_path}")
    config_db_path.unlink()

conn_config = sqlite3.connect(str(config_db_path))
cursor_config = conn_config.cursor()

# Business Configuration Table
cursor_config.execute("""
CREATE TABLE IF NOT EXISTS business_config (
    parameter_key TEXT PRIMARY KEY,
    parameter_value TEXT NOT NULL,
    value_type TEXT NOT NULL,
    description TEXT,
    min_value REAL,
    max_value REAL,
    updated_at TEXT NOT NULL,
    updated_by TEXT
)
""")

# Model Configuration Table
cursor_config.execute("""
CREATE TABLE IF NOT EXISTS model_config (
    parameter_key TEXT PRIMARY KEY,
    parameter_value TEXT NOT NULL,
    value_type TEXT NOT NULL,
    description TEXT,
    min_value REAL,
    max_value REAL,
    updated_at TEXT NOT NULL,
    updated_by TEXT
)
""")

# Insert default business configuration
business_params = [
    # Cost parameters
    ("fuel_cost_per_km", "15.0", "float", "Fuel cost in KES per kilometer", 10.0, 50.0),
    ("driver_cost_per_hour", "300.0", "float", "Driver cost in KES per hour", 100.0, 1000.0),
    ("vehicle_depreciation_per_km", "5.0", "float", "Vehicle depreciation in KES per km", 1.0, 20.0),
    ("late_delivery_penalty_per_hour", "500.0", "float", "Late delivery penalty in KES per hour", 100.0, 5000.0),
    ("priority_bonus", "200.0", "float", "Bonus for priority delivery in KES", 0.0, 1000.0),
    
    # Operational parameters
    ("max_route_duration_hours", "8.0", "float", "Maximum route duration in hours", 1.0, 24.0),
    ("max_stops_per_route", "10", "int", "Maximum stops per route", 1, 50),
    ("average_stop_duration_minutes", "15.0", "float", "Average time per stop in minutes", 5.0, 60.0),
    ("vehicle_capacity_kg", "1000.0", "float", "Default vehicle capacity in kg", 100.0, 5000.0),
    
    # Service level parameters
    ("standard_delivery_sla_hours", "24.0", "float", "Standard delivery SLA in hours", 12.0, 72.0),
    ("urgent_delivery_sla_hours", "6.0", "float", "Urgent delivery SLA in hours", 2.0, 24.0),
    ("emergency_delivery_sla_hours", "2.0", "float", "Emergency delivery SLA in hours", 0.5, 6.0),
]

timestamp = datetime.now().isoformat()
for key, value, vtype, desc, min_val, max_val in business_params:
    cursor_config.execute("""
        INSERT INTO business_config 
        (parameter_key, parameter_value, value_type, description, min_value, max_value, updated_at, updated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (key, value, vtype, desc, min_val, max_val, timestamp, "system"))

# Insert default model configuration
model_params = [
    # VFA parameters
    ("vfa_learning_rate", "0.001", "float", "TD learning rate (alpha)", 0.0001, 0.1),
    ("vfa_discount_factor", "0.95", "float", "Discount factor (gamma)", 0.8, 0.99),
    ("vfa_lambda", "0.7", "float", "Eligibility trace decay (lambda)", 0.0, 1.0),
    ("vfa_epsilon", "0.1", "float", "Exploration rate (epsilon)", 0.0, 0.3),
    ("vfa_batch_size", "32", "int", "Batch size for updates", 16, 256),
    ("vfa_update_frequency", "10", "int", "Updates between model saves", 1, 100),
    
    # CFA parameters
    ("cfa_time_limit_seconds", "30", "int", "MIP solver time limit", 5, 300),
    ("cfa_optimality_gap", "0.01", "float", "MIP optimality gap tolerance", 0.001, 0.1),
    ("cfa_use_warm_start", "true", "bool", "Use warm start from previous solution", None, None),
    
    # DLA parameters
    ("dla_num_scenarios", "100", "int", "Number of Monte Carlo scenarios", 10, 1000),
    ("dla_lookahead_horizon_hours", "4.0", "float", "Lookahead horizon in hours", 1.0, 24.0),
    ("dla_traffic_uncertainty", "0.3", "float", "Traffic uncertainty factor", 0.0, 1.0),
    ("dla_demand_uncertainty", "0.2", "float", "Demand uncertainty factor", 0.0, 1.0),
    
    # Meta-controller parameters
    ("meta_pfa_urgency_threshold", "0.8", "float", "Urgency threshold for PFA", 0.0, 1.0),
    ("meta_pfa_driver_threshold", "0.2", "float", "Min driver availability for PFA", 0.0, 0.5),
    ("meta_dla_uncertainty_threshold", "0.6", "float", "Uncertainty threshold for DLA", 0.0, 1.0),
    ("meta_dla_horizon_threshold", "4.0", "float", "Min horizon for DLA (hours)", 1.0, 12.0),
]

for key, value, vtype, desc, min_val, max_val in model_params:
    cursor_config.execute("""
        INSERT INTO model_config 
        (parameter_key, parameter_value, value_type, description, min_value, max_value, updated_at, updated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (key, value, vtype, desc, min_val, max_val, timestamp, "system"))

conn_config.commit()
print(f"✅ Config database created: {config_db_path}")
print(f"   - Business parameters: {len(business_params)}")
print(f"   - Model parameters: {len(model_params)}")

# ============================================================
# 2. State Database (senga_state.db)
# ============================================================

print()
print("📊 Creating state database...")

# Remove old database if exists
if state_db_path.exists():
    print(f"   Removing old state database: {state_db_path}")
    state_db_path.unlink()

conn_state = sqlite3.connect(str(state_db_path))
cursor_state = conn_state.cursor()

# System State Table
cursor_state.execute("""
CREATE TABLE IF NOT EXISTS system_state (
    state_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    state_type TEXT NOT NULL,
    state_data TEXT NOT NULL,
    context TEXT
)
""")

# Create index for faster queries
cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_system_state_timestamp 
ON system_state(timestamp)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_system_state_type 
ON system_state(state_type)
""")

# VFA Weights Table
cursor_state.execute("""
CREATE TABLE IF NOT EXISTS vfa_weights (
    weight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    weight_value REAL NOT NULL,
    updated_at TEXT NOT NULL,
    iteration INTEGER NOT NULL
)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_vfa_weights_iteration 
ON vfa_weights(iteration)
""")

# Learning Metrics Table
cursor_state.execute("""
CREATE TABLE IF NOT EXISTS learning_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    context TEXT
)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_learning_metrics_timestamp 
ON learning_metrics(timestamp)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_learning_metrics_type 
ON learning_metrics(metric_type)
""")

# Decision History Table
cursor_state.execute("""
CREATE TABLE IF NOT EXISTS decision_history (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    order_id TEXT NOT NULL,
    function_class TEXT NOT NULL,
    decision_data TEXT NOT NULL,
    outcome TEXT,
    reward REAL
)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_decision_history_timestamp 
ON decision_history(timestamp)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_decision_history_function 
ON decision_history(function_class)
""")

# Orders Table
cursor_state.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    customer_name TEXT,
    pickup_location TEXT NOT NULL,
    delivery_location TEXT NOT NULL,
    package_weight REAL NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    assigned_driver TEXT,
    estimated_time REAL,
    actual_time REAL,
    order_data TEXT
)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_orders_status 
ON orders(status)
""")

cursor_state.execute("""
CREATE INDEX IF NOT EXISTS idx_orders_created 
ON orders(created_at)
""")

# Drivers Table
cursor_state.execute("""
CREATE TABLE IF NOT EXISTS drivers (
    driver_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT,
    vehicle_type TEXT,
    capacity_kg REAL,
    current_location TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
)
""")

conn_state.commit()
print(f"✅ State database created: {state_db_path}")
print("   - Tables created: system_state, vfa_weights, learning_metrics,")
print("                     decision_history, orders, drivers")

# ============================================================
# 3. Verify Setup
# ============================================================

print()
print("🔍 Verifying setup...")

# Verify config database
cursor_config.execute("SELECT COUNT(*) FROM business_config")
biz_count = cursor_config.fetchone()[0]
cursor_config.execute("SELECT COUNT(*) FROM model_config")
model_count = cursor_config.fetchone()[0]

print(f"   ✅ Business config parameters: {biz_count}")
print(f"   ✅ Model config parameters: {model_count}")

# Verify state database
cursor_state.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor_state.fetchall()
print(f"   ✅ State database tables: {len(tables)}")

# Close connections
conn_config.close()
conn_state.close()

print()
print("=" * 60)
print("✅ Senga SDE System Initialized Successfully!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Start the API: python src/api/main.py")
print("  2. Start the demo: streamlit run scripts/streamlit_demo.py")
print("  3. Or run both: ./run_demo.bat (Windows) or ./run_demo.sh (Linux/Mac)")
print()