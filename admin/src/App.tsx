import { useState, useEffect, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Plus,
  TruckIcon,
  PackageIcon,
  AlertCircle,
  CheckCircle,
  Clock,
  TrendingUp,
  Activity,
  Settings,
} from "lucide-react";
import "./App.css";

const API_BASE = "/api";

// ============================================================================
// KENYAN LOCATIONS FOR TESTING
// ============================================================================
const KENYAN_LOCATIONS = [
  { name: "Nairobi CBD", lat: -1.286389, lon: 36.817223 },
  { name: "Westlands", lat: -1.266667, lon: 36.805 },
  { name: "Industrial Area", lat: -1.322, lon: 36.847 },
  { name: "Nakuru", lat: -0.303099, lon: 36.066667 },
  { name: "Eldoret", lat: 0.514277, lon: 35.269779 },
  { name: "Kisumu", lat: -0.091702, lon: 34.767956 },
  { name: "Mombasa", lat: -4.04374, lon: 39.668207 },
  { name: "Thika", lat: -1.033333, lon: 37.083333 },
];

// ============================================================================
// MAIN APP COMPONENT
// ============================================================================
export default function App() {
  // System state
  const [systemStatus, setSystemStatus] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);

  // Simulation state
  const [simulatorRunning, setSimulatorRunning] = useState(false);
  const [orderRate, setOrderRate] = useState(2); // orders per minute
  const [cycleInterval, setCycleInterval] = useState(10); // minutes

  // Data state
  const [pendingOrders, setPendingOrders] = useState([]);
  const [activeRoutes, setActiveRoutes] = useState([]);
  const [recentCycles, setRecentCycles] = useState([]);
  const [learningMetrics, setLearningMetrics] = useState(null);
  const [eventLog, setEventLog] = useState([]);

  // UI state
  const [activeTab, setActiveTab] = useState("simulator");
  const [showOrderForm, setShowOrderForm] = useState(false);

  // ============================================================================
  // API CALLS
  // ============================================================================

  const checkApiHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      setApiOnline(response.ok);
      return response.ok;
    } catch {
      setApiOnline(false);
      return false;
    }
  }, []);

  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
      }
    } catch (error) {
      console.error("Failed to fetch system status:", error);
    }
  }, []);

  const fetchPendingOrders = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/orders/pending`);
      if (response.ok) {
        const data = await response.json();
        setPendingOrders(Array.isArray(data) ? data : []);
      }
    } catch (error) {
      console.error("Failed to fetch pending orders:", error);
    }
  }, []);

  const fetchActiveRoutes = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/routes/active`);
      if (response.ok) {
        const data = await response.json();
        setActiveRoutes(data.routes || []);
      }
    } catch (error) {
      console.error("Failed to fetch active routes:", error);
    }
  }, []);

  const fetchLearningMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/learning/vfa-metrics`);
      if (response.ok) {
        const data = await response.json();
        setLearningMetrics(data);
      }
    } catch (error) {
      console.error("Failed to fetch learning metrics:", error);
    }
  }, []);

  // Event logging - Memoized to prevent re-renders breaking intervals
  const addEvent = useCallback((type, message, level = "info") => {
    const event = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      type,
      message,
      level,
    };
    setEventLog((prev) => [event, ...prev].slice(0, 100));
  }, []);

  // CREATE ORDER
  const createOrder = useCallback(
    async (orderData) => {
      console.log("Creating order:", orderData);
      try {
        const response = await fetch(`${API_BASE}/orders`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(orderData),
        });

        if (response.ok) {
          const result = await response.json();
          addEvent(
            "order_created",
            `Order ${result.order_id} created`,
            "success"
          );
          await fetchPendingOrders();
          return result;
        } else {
          const errorText = await response.text();
          addEvent("error", `Failed to create order: ${errorText}`, "error");
        }
      } catch (error) {
        addEvent("error", `API Error: ${error.message}`, "error");
      }
    },
    [fetchPendingOrders, addEvent]
  );

  // TRIGGER CONSOLIDATION CYCLE
  const triggerConsolidationCycle = useCallback(async () => {
    console.log("Triggering consolidation cycle");
    try {
      addEvent("cycle_start", "Triggering consolidation cycle...", "info");

      const response = await fetch(
        `${API_BASE}/decisions/consolidation-cycle`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            force_dispatch: false,
            context: null,
          }),
        }
      );

      if (response.ok) {
        const result = await response.json();

        addEvent(
          "cycle_complete",
          `Cycle completed: ${result.function_class_used} - ${result.orders_dispatched} dispatched`,
          "success"
        );

        setRecentCycles((prev) =>
          [
            {
              timestamp: result.timestamp,
              function_class: result.function_class_used,
              orders_dispatched: result.orders_dispatched,
              batches_created: result.batches_created,
            },
            ...prev,
          ].slice(0, 10)
        );

        await Promise.all([
          fetchPendingOrders(),
          fetchActiveRoutes(),
          fetchLearningMetrics(),
        ]);

        return result;
      } else {
        const errorText = await response.text();
        addEvent("error", `Cycle failed: ${errorText}`, "error");
      }
    } catch (error) {
      addEvent("error", `Cycle error: ${error.message}`, "error");
    }
  }, [fetchPendingOrders, fetchActiveRoutes, fetchLearningMetrics, addEvent]);

  // ============================================================================
  // SIMULATION LOGIC
  // ============================================================================

  const generateRandomOrder = useCallback(() => {
    let pickup, delivery;
    let attempts = 0;
    do {
      pickup =
        KENYAN_LOCATIONS[Math.floor(Math.random() * KENYAN_LOCATIONS.length)];
      delivery =
        KENYAN_LOCATIONS[Math.floor(Math.random() * KENYAN_LOCATIONS.length)];
      attempts++;
    } while (pickup.name === delivery.name && attempts < 10);

    if (pickup.name === delivery.name) {
      pickup = KENYAN_LOCATIONS[0];
      delivery = KENYAN_LOCATIONS[1];
    }

    const weights = [100, 250, 500, 750, 1000];
    const priorities = ["standard", "standard", "urgent"];

    return {
      customer_name: `Customer ${Math.floor(Math.random() * 100)}`,
      customer_phone: `+254${Math.floor(
        Math.random() * 900000000 + 100000000
      )}`,
      pickup_location: {
        address: pickup.name,
        latitude: pickup.lat,
        longitude: pickup.lon,
      },
      delivery_location: {
        address: delivery.name,
        latitude: delivery.lat,
        longitude: delivery.lon,
      },
      package_weight: weights[Math.floor(Math.random() * weights.length)],
      volume_m3: parseFloat((Math.random() * 2 + 0.5).toFixed(2)),
      priority: priorities[Math.floor(Math.random() * priorities.length)],
    };
  }, []);

  const startSimulator = () => {
    setSimulatorRunning(true);
    addEvent("simulator_start", "Simulator started", "info");
  };

  const stopSimulator = () => {
    setSimulatorRunning(false);
    addEvent("simulator_stop", "Simulator stopped", "info");
  };

  const resetSimulation = async () => {
    stopSimulator();
    setPendingOrders([]);
    setActiveRoutes([]);
    setRecentCycles([]);
    setEventLog([]);
    addEvent("simulator_reset", "Simulation reset", "info");
  };

  // ============================================================================
  // SIMULATION LOOPS (FIXED)
  // ============================================================================

  // Order generation loop
  useEffect(() => {
    if (!simulatorRunning) return;

    const intervalMs = (60 / orderRate) * 1000;

    const generate = async () => {
      console.log("Generating order...");
      const order = generateRandomOrder();
      await createOrder(order);
    };

    generate(); // Immediate first order

    const interval = setInterval(generate, intervalMs);
    return () => clearInterval(interval);
  }, [simulatorRunning, orderRate, generateRandomOrder, createOrder]);

  // Auto-consolidation loop
  useEffect(() => {
    if (!simulatorRunning) return;

    const consolidate = async () => {
      console.log("Running auto consolidation...");
      await triggerConsolidationCycle();
    };

    consolidate(); // Immediate first cycle

    const interval = setInterval(consolidate, cycleInterval * 60 * 1000);
    return () => clearInterval(interval);
  }, [simulatorRunning, cycleInterval, triggerConsolidationCycle]);

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  useEffect(() => {
    checkApiHealth();
    const healthInterval = setInterval(checkApiHealth, 5000);
    return () => clearInterval(healthInterval);
  }, [checkApiHealth]);

  useEffect(() => {
    if (apiOnline) {
      fetchSystemStatus();
      fetchPendingOrders();
      fetchActiveRoutes();
      fetchLearningMetrics();

      const refreshInterval = setInterval(() => {
        fetchSystemStatus();
        fetchPendingOrders();
        fetchActiveRoutes();
      }, 5000);

      return () => clearInterval(refreshInterval);
    }
  }, [
    apiOnline,
    fetchSystemStatus,
    fetchPendingOrders,
    fetchActiveRoutes,
    fetchLearningMetrics,
  ]);

  // ============================================================================
  // RENDER
  // ============================================================================

  if (!apiOnline) {
    return (
      <div className="error-screen">
        <AlertCircle size={64} className="error-icon" />
        <h1>API Offline</h1>
        <p>Cannot connect to Senga SDE backend</p>
        <p>Please ensure the API is running at http://localhost:8000</p>
        <code>python src/api/main.py</code>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <TruckIcon size={32} />
            <div>
              <h1>Senga Operations</h1>
              <p>Real-time freight consolidation & routing</p>
            </div>
          </div>

          <div className="header-status">
            <StatusBadge status={systemStatus?.engine_status} />
            <span className="cycle-count">
              Cycle #{systemStatus?.current_cycle || 0}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Tabs */}
        <div className="tabs">
          <button
            className={`tab ${activeTab === "simulator" ? "active" : ""}`}
            onClick={() => setActiveTab("simulator")}
          >
            <Play size={16} /> Simulator
          </button>
          <button
            className={`tab ${activeTab === "workflow" ? "active" : ""}`}
            onClick={() => setActiveTab("workflow")}
          >
            <Activity size={16} /> Workflow
          </button>
          <button
            className={`tab ${activeTab === "learning" ? "active" : ""}`}
            onClick={() => setActiveTab("learning")}
          >
            <TrendingUp size={16} /> Learning
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === "simulator" && (
            <SimulatorTab
              simulatorRunning={simulatorRunning}
              orderRate={orderRate}
              cycleInterval={cycleInterval}
              onStart={startSimulator}
              onStop={stopSimulator}
              onReset={resetSimulation}
              onOrderRateChange={setOrderRate}
              onCycleIntervalChange={setCycleInterval}
              onCreateOrder={() => setShowOrderForm(true)}
              onTriggerCycle={triggerConsolidationCycle}
              pendingOrders={pendingOrders}
              eventLog={eventLog}
            />
          )}

          {activeTab === "workflow" && (
            <WorkflowTab
              systemStatus={systemStatus}
              pendingOrders={pendingOrders}
              activeRoutes={activeRoutes}
              recentCycles={recentCycles}
            />
          )}

          {activeTab === "learning" && (
            <LearningTab
              metrics={learningMetrics}
              recentCycles={recentCycles}
            />
          )}
        </div>
      </div>

      {/* Manual Order Form Modal */}
      {showOrderForm && (
        <OrderFormModal
          onClose={() => setShowOrderForm(false)}
          onSubmit={createOrder}
          locations={KENYAN_LOCATIONS}
        />
      )}
    </div>
  );
}

// ============================================================================
// TAB COMPONENTS
// ============================================================================

function SimulatorTab({
  simulatorRunning,
  orderRate,
  cycleInterval,
  onStart,
  onStop,
  onReset,
  onOrderRateChange,
  onCycleIntervalChange,
  onCreateOrder,
  onTriggerCycle,
  pendingOrders,
  eventLog,
}) {
  return (
    <div className="simulator-tab">
      <div className="card">
        <h2>Simulation Controls</h2>

        <div className="control-row">
          {!simulatorRunning ? (
            <button className="btn btn-primary btn-large" onClick={onStart}>
              <Play size={20} /> Start Simulator
            </button>
          ) : (
            <button className="btn btn-secondary btn-large" onClick={onStop}>
              <Pause size={20} /> Stop Simulator
            </button>
          )}

          <button
            className="btn btn-danger"
            onClick={onReset}
            disabled={simulatorRunning}
          >
            <RotateCcw size={16} /> Reset
          </button>
        </div>

        <div className="settings-grid">
          <div className="setting">
            <label>Order Generation Rate</label>
            <div className="range-input">
              <input
                type="range"
                min="1"
                max="10"
                value={orderRate}
                onChange={(e) => onOrderRateChange(Number(e.target.value))}
                disabled={simulatorRunning}
              />
              <span>{orderRate} orders/min</span>
            </div>
          </div>

          <div className="setting">
            <label>Consolidation Interval</label>
            <div className="range-input">
              <input
                type="range"
                min="1"
                max="60"
                value={cycleInterval}
                onChange={(e) => onCycleIntervalChange(Number(e.target.value))}
                disabled={simulatorRunning}
              />
              <span>{cycleInterval} minutes</span>
            </div>
          </div>
        </div>

        <div className="manual-controls">
          <button className="btn btn-outline" onClick={onCreateOrder}>
            <Plus size={16} /> Create Manual Order
          </button>
          <button className="btn btn-outline" onClick={onTriggerCycle}>
            <Activity size={16} /> Trigger Cycle Now
          </button>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard
          icon={<PackageIcon />}
          label="Pending Orders"
          value={pendingOrders.length}
          color="blue"
        />
        <StatCard
          icon={<Clock />}
          label="Avg Order Age"
          value={calculateAvgAge(pendingOrders)}
          suffix=" hrs"
          color="orange"
        />
        <StatCard
          icon={<TruckIcon />}
          label="Total Weight"
          value={calculateTotalWeight(pendingOrders)}
          suffix=" kg"
          color="green"
        />
      </div>

      <div className="card">
        <h2>Live Event Log</h2>
        <div className="event-log">
          {eventLog.length === 0 ? (
            <div className="empty-state">
              No events yet. Start the simulator to begin.
            </div>
          ) : (
            eventLog.map((event) => (
              <div key={event.id} className={`event event-${event.level}`}>
                <span className="event-time">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
                <span className="event-message">{event.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function WorkflowTab({
  systemStatus,
  pendingOrders,
  activeRoutes,
  recentCycles,
}) {
  return (
    <div className="workflow-tab">
      <div className="metrics-grid">
        <MetricCard
          label="Pending Shipments"
          value={systemStatus?.pending_shipments || 0}
        />
        <MetricCard
          label="Available Vehicles"
          value={systemStatus?.available_vehicles || 0}
        />
        <MetricCard
          label="Fleet Utilization"
          value="0.0%"
          subtitle="Capacity available"
        />
        <MetricCard
          label="SLA Compliance"
          value="0.0%"
          subtitle="On-time deliveries"
        />
      </div>

      <div className="card">
        <h2>Pending Shipments</h2>
        {pendingOrders.length === 0 ? (
          <div className="empty-state">
            <PackageIcon size={48} />
            <p>No pending shipments</p>
          </div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Order ID</th>
                  <th>Origin → Destination</th>
                  <th>Weight</th>
                  <th>Age</th>
                  <th>Deadline</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                {pendingOrders.map((order) => {
                  const ageHours = order.created_at
                    ? (new Date() - new Date(order.created_at)) /
                      (1000 * 60 * 60)
                    : 0;

                  return (
                    <tr key={order.order_id}>
                      <td>
                        <code>{order.order_id?.substring(0, 8)}</code>
                      </td>
                      <td className="route-cell">
                        {order.pickup_location?.address?.substring(0, 15) ||
                          "–"}{" "}
                        →{" "}
                        {order.delivery_location?.address?.substring(0, 15) ||
                          "–"}
                      </td>
                      <td>{order.package_weight || 0} kg</td>
                      <td>{ageHours.toFixed(1)} hrs</td>
                      <td>
                        {order.deadline
                          ? new Date(order.deadline).toLocaleDateString()
                          : "–"}
                      </td>
                      <td>
                        <span className={`badge badge-${order.priority}`}>
                          {order.priority?.toUpperCase() || "STANDARD"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h2>Recent Decisions</h2>
        <div className="subtitle">Last 10 consolidation cycles</div>

        {recentCycles.length === 0 ? (
          <div className="empty-state">
            <Clock size={48} />
            <p>No cycles yet</p>
          </div>
        ) : (
          <div className="decisions-list">
            {recentCycles.map((cycle, idx) => (
              <div key={idx} className="decision-item">
                <div className="decision-header">
                  <span className="decision-time">
                    {new Date(cycle.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`function-badge fc-${cycle.function_class}`}>
                    {cycle.function_class}
                  </span>
                </div>
                <div className="decision-body">
                  <strong>{cycle.orders_dispatched}</strong> orders dispatched
                </div>
                <div className="decision-footer">
                  {cycle.batches_created} batches created
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Fleet Status</h2>
        <div className="subtitle">0 vehicles</div>
        <div className="empty-state">
          <TruckIcon size={48} />
          <p>No vehicles</p>
        </div>
      </div>
    </div>
  );
}

function LearningTab({ metrics, recentCycles }) {
  if (!metrics) {
    return (
      <div className="learning-tab">
        <div className="card">
          <div className="empty-state">
            <TrendingUp size={48} />
            <p>No learning metrics available yet</p>
            <p className="subtitle">Make some routing decisions first</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="learning-tab">
      <div className="card">
        <h2>Value Function Approximation (VFA)</h2>
        <div className="metrics-row">
          <div className="metric-box">
            <div className="metric-label">Updates</div>
            <div className="metric-value">{metrics.num_updates || 0}</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Learning Rate</div>
            <div className="metric-value">
              {metrics.learning_rate?.toFixed(4) || "0.0000"}
            </div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Avg TD Error</div>
            <div className="metric-value">
              {metrics.recent_avg_error?.toFixed(2) || "0.00"}
            </div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Convergence</div>
            <div className="metric-value">
              {metrics.convergence_score
                ? (metrics.convergence_score * 100).toFixed(1) + "%"
                : "0.0%"}
            </div>
          </div>
        </div>
      </div>
      {/* Feature Weights */}
      {metrics.feature_weights && (
        <div className="card">
          <h2>Top Feature Weights</h2>
          <div className="feature-weights">
            {Object.entries(metrics.feature_weights)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 10)
              .map(([feature, weight]) => (
                <div key={feature} className="feature-row">
                  <span className="feature-name">{feature}</span>
                  <div className="feature-bar-container">
                    <div
                      className={`feature-bar ${
                        weight >= 0 ? "positive" : "negative"
                      }`}
                      style={{
                        width: `${Math.min(Math.abs(weight) * 10, 100)}%`,
                      }}
                    />
                  </div>
                  <span className="feature-weight">{weight.toFixed(3)}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* TD Error History */}
      {metrics.td_error_history && metrics.td_error_history.length > 0 && (
        <div className="card">
          <h2>TD Error Convergence</h2>
          <SimpleLineChart data={metrics.td_error_history} label="TD Error" />
        </div>
      )}

      {/* Performance History */}
      {metrics.performance_history &&
        metrics.performance_history.length > 0 && (
          <div className="card">
            <h2>Average Reward Trend</h2>
            <SimpleLineChart
              data={metrics.performance_history}
              label="Avg Reward"
              color="#28a745"
            />
          </div>
        )}
    </div>
  );
}

// ============================================================================
// UTILITY COMPONENTS
// ============================================================================

function StatusBadge({ status }) {
  const statusConfig = {
    idle: { color: "yellow", label: "IDLE" },
    running: { color: "green", label: "RUNNING" },
    paused: { color: "orange", label: "PAUSED" },
    error: { color: "red", label: "ERROR" },
  };

  const config = statusConfig[status] || statusConfig.idle;

  return (
    <span className={`status-badge status-${config.color}`}>
      {config.label}
    </span>
  );
}

function StatCard({ icon, label, value, suffix = "", color = "blue" }) {
  return (
    <div className={`stat-card stat-${color}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <div className="stat-label">{label}</div>
        <div className="stat-value">
          {value}
          {suffix && <span className="stat-suffix">{suffix}</span>}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, subtitle }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );
}

function SimpleLineChart({ data, label, color = "#007bff" }) {
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;

  return (
    <div className="simple-chart">
      <div className="chart-container">
        {data.map((value, idx) => {
          const height = ((value - min) / range) * 100;
          return (
            <div
              key={idx}
              className="chart-bar"
              style={{
                height: `${Math.max(height, 2)}%`,
                backgroundColor: color,
              }}
              title={`${label}: ${value.toFixed(2)}`}
            />
          );
        })}
      </div>
      <div className="chart-label">{label} over time</div>
    </div>
  );
}

function OrderFormModal({ onClose, onSubmit, locations }) {
  const [formData, setFormData] = useState({
    customer_id: `CUST-${Math.floor(Math.random() * 1000)}`,
    pickup_location: "",
    delivery_location: "",
    weight_kg: 500,
    volume_m3: 1.5,
    declared_value: 25000,
    priority: "standard",
    deadline_hours: 48,
  });

  const handleSubmit = (e) => {
    e.preventDefault();

    const now = new Date();
    const deadline = new Date(
      now.getTime() + formData.deadline_hours * 3600 * 1000
    );

    const order = {
      order_id: `ORD-${Date.now()}`,
      customer_id: formData.customer_id,
      pickup_location: formData.pickup_location,
      delivery_location: formData.delivery_location,
      weight_kg: formData.weight_kg,
      volume_m3: formData.volume_m3,
      declared_value: formData.declared_value,
      delivery_deadline: deadline.toISOString(),
      priority: formData.priority,
    };

    onSubmit(order);
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Create Manual Order</h2>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Customer ID</label>
            <input
              type="text"
              value={formData.customer_id}
              onChange={(e) =>
                setFormData({ ...formData, customer_id: e.target.value })
              }
              required
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Pickup Location</label>
              <select
                value={formData.pickup_location}
                onChange={(e) =>
                  setFormData({ ...formData, pickup_location: e.target.value })
                }
                required
              >
                <option value="">Select location...</option>
                {locations.map((loc) => (
                  <option key={loc.name} value={loc.name}>
                    {loc.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Delivery Location</label>
              <select
                value={formData.delivery_location}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    delivery_location: e.target.value,
                  })
                }
                required
              >
                <option value="">Select location...</option>
                {locations.map((loc) => (
                  <option key={loc.name} value={loc.name}>
                    {loc.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Weight (kg)</label>
              <input
                type="number"
                value={formData.weight_kg}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    weight_kg: Number(e.target.value),
                  })
                }
                min="1"
                required
              />
            </div>

            <div className="form-group">
              <label>Volume (m³)</label>
              <input
                type="number"
                step="0.1"
                value={formData.volume_m3}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    volume_m3: Number(e.target.value),
                  })
                }
                min="0.1"
                required
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Declared Value (KES)</label>
              <input
                type="number"
                value={formData.declared_value}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    declared_value: Number(e.target.value),
                  })
                }
                min="1"
                required
              />
            </div>

            <div className="form-group">
              <label>Deadline (hours)</label>
              <input
                type="number"
                value={formData.deadline_hours}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    deadline_hours: Number(e.target.value),
                  })
                }
                min="1"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label>Priority</label>
            <select
              value={formData.priority}
              onChange={(e) =>
                setFormData({ ...formData, priority: e.target.value })
              }
            >
              <option value="standard">Standard</option>
              <option value="express">Express</option>
              <option value="emergency">Emergency</option>
            </select>
          </div>

          <div className="form-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Cancel
            </button>
            <button type="submit" className="btn btn-primary">
              Create Order
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function calculateAvgAge(orders) {
  if (orders.length === 0) return 0;
  // Calculate age from created_at
  const now = new Date();
  const sum = orders.reduce((acc, order) => {
    const createdAt = new Date(order.created_at);
    const ageHours = (now - createdAt) / (1000 * 60 * 60);
    return acc + ageHours;
  }, 0);
  return (sum / orders.length).toFixed(1);
}

function calculateTotalWeight(orders) {
  return orders.reduce((acc, order) => acc + (order.package_weight || 0), 0);
}
