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
  MapPin,
  ArrowRight,
} from "lucide-react";
import "./App.css";

const API_BASE = "/api";

// ============================================================================
// REALISTIC SENGA LOCATIONS - NAIROBI PICKUPS + NATIONWIDE DESTINATIONS
// ============================================================================

const NAIROBI_PICKUP_HUBS = [
  {
    id: "NBO_CBD",
    name: "CBD - Kenyatta Avenue",
    lat: -1.2864,
    lon: 36.8172,
    type: "commercial",
  },
  {
    id: "NBO_WEST",
    name: "Westlands - Sarit Centre",
    lat: -1.2636,
    lon: 36.8053,
    type: "commercial",
  },
  {
    id: "NBO_IND",
    name: "Industrial Area - Enterprise Rd",
    lat: -1.3229,
    lon: 36.8467,
    type: "warehouse",
  },
  {
    id: "NBO_UPPER",
    name: "Upperhill - Britam Tower",
    lat: -1.2921,
    lon: 36.8219,
    type: "commercial",
  },
  {
    id: "NBO_EAST",
    name: "Eastleigh - 1st Avenue",
    lat: -1.2816,
    lon: 36.8469,
    type: "commercial",
  },
  {
    id: "NBO_KILIM",
    name: "Kilimani - Yaya Centre",
    lat: -1.2896,
    lon: 36.7823,
    type: "commercial",
  },
];

const KENYA_DESTINATIONS = [
  // Coastal Lane (Mombasa Road)
  {
    id: "MSA_NYA",
    name: "Mombasa - Nyali",
    lat: -4.0435,
    lon: 39.7292,
    distance_km: 480,
    lane: "coastal",
  },
  {
    id: "MSA_CBD",
    name: "Mombasa - CBD",
    lat: -4.0435,
    lon: 39.6682,
    distance_km: 480,
    lane: "coastal",
  },
  {
    id: "MSA_LIK",
    name: "Mombasa - Likoni",
    lat: -4.0832,
    lon: 39.6668,
    distance_km: 490,
    lane: "coastal",
  },
  {
    id: "VOI",
    name: "Voi",
    lat: -3.3967,
    lon: 38.5564,
    distance_km: 340,
    lane: "coastal",
  },
  {
    id: "MAR",
    name: "Mariakani",
    lat: -3.8572,
    lon: 39.4748,
    distance_km: 400,
    lane: "coastal",
  },
  {
    id: "MAL",
    name: "Malindi",
    lat: -3.2175,
    lon: 40.1169,
    distance_km: 550,
    lane: "coastal",
  },

  // Western Lane (Nakuru-Eldoret-Kisumu)
  {
    id: "NAK_TWN",
    name: "Nakuru - Town",
    lat: -0.3031,
    lon: 36.08,
    distance_km: 160,
    lane: "western",
  },
  {
    id: "NAK_IND",
    name: "Nakuru - Industrial",
    lat: -0.2827,
    lon: 36.0664,
    distance_km: 165,
    lane: "western",
  },
  {
    id: "NAI",
    name: "Naivasha",
    lat: -0.7167,
    lon: 36.4333,
    distance_km: 90,
    lane: "western",
  },
  {
    id: "ELD_TWN",
    name: "Eldoret - Town",
    lat: 0.5143,
    lon: 35.2698,
    distance_km: 310,
    lane: "western",
  },
  {
    id: "ELD_IND",
    name: "Eldoret - Industrial",
    lat: 0.5287,
    lon: 35.2415,
    distance_km: 315,
    lane: "western",
  },
  {
    id: "KIS_TWN",
    name: "Kisumu - Town",
    lat: -0.0917,
    lon: 34.768,
    distance_km: 350,
    lane: "western",
  },
  {
    id: "KIT",
    name: "Kitale",
    lat: 1.0157,
    lon: 35.0062,
    distance_km: 380,
    lane: "western",
  },

  // Northern Lane (Thika-Nanyuki-Isiolo)
  {
    id: "THK",
    name: "Thika",
    lat: -1.0332,
    lon: 37.069,
    distance_km: 42,
    lane: "northern",
  },
  {
    id: "MUR",
    name: "Murang'a",
    lat: -0.7167,
    lon: 37.15,
    distance_km: 75,
    lane: "northern",
  },
  {
    id: "NYE",
    name: "Nyeri",
    lat: -0.4167,
    lon: 36.95,
    distance_km: 150,
    lane: "northern",
  },
  {
    id: "NAN",
    name: "Nanyuki",
    lat: -0.0167,
    lon: 37.0667,
    distance_km: 200,
    lane: "northern",
  },
  {
    id: "ISI",
    name: "Isiolo",
    lat: 0.3542,
    lon: 37.5833,
    distance_km: 285,
    lane: "northern",
  },
  {
    id: "MER",
    name: "Meru",
    lat: 0.05,
    lon: 37.65,
    distance_km: 230,
    lane: "northern",
  },
  {
    id: "EMB",
    name: "Embu",
    lat: -0.5333,
    lon: 37.45,
    distance_km: 120,
    lane: "northern",
  },
];

// ============================================================================
// MAIN APP COMPONENT
// ============================================================================

export default function App() {
  // System state
  const [systemStatus, setSystemStatus] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);

  // Workflow state
  const [pendingOrders, setPendingOrders] = useState([]);
  const [consolidationQueue, setConsolidationQueue] = useState([]);
  const [dispatchedRoutes, setDispatchedRoutes] = useState([]);
  const [inTransitRoutes, setInTransitRoutes] = useState([]);
  const [completedRoutes, setCompletedRoutes] = useState([]);

  // Learning state
  const [learningMetrics, setLearningMetrics] = useState(null);
  const [performanceHistory, setPerformanceHistory] = useState([]);

  // Simulation controls
  const [simulatorRunning, setSimulatorRunning] = useState(false);
  const [orderRate, setOrderRate] = useState(3); // orders per minute
  const [consolidationInterval, setConsolidationInterval] = useState(5); // minutes
  const [transitSimSpeed, setTransitSimSpeed] = useState(10); // seconds per hour simulated

  // UI state
  const [activeTab, setActiveTab] = useState("workflow");
  const [eventLog, setEventLog] = useState([]);

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
        // Backend returns array of OrderResponse objects directly
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
        // Backend returns array directly
        setInTransitRoutes(Array.isArray(data) ? data : []);
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

        // Track performance history
        if (data.avg_reward) {
          setPerformanceHistory((prev) => [
            ...prev.slice(-99),
            data.avg_reward,
          ]);
        }
      }
    } catch (error) {
      console.error("Failed to fetch learning metrics:", error);
    }
  }, []);

  const addEvent = useCallback((type, message, level = "info") => {
    const event = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      type,
      message,
      level,
    };
    setEventLog((prev) => [event, ...prev].slice(0, 100));
  }, []);

  // ============================================================================
  // ORDER GENERATION - REALISTIC SENGA PATTERNS
  // ============================================================================

  const generateRealisticOrder = useCallback(() => {
    // Pick random Nairobi pickup hub
    const pickup =
      NAIROBI_PICKUP_HUBS[
        Math.floor(Math.random() * NAIROBI_PICKUP_HUBS.length)
      ];

    // Pick destination - weighted towards common lanes
    const laneWeights = {
      coastal: 0.4, // 40% to coast (Mombasa is major destination)
      western: 0.35, // 35% to western (Nakuru/Eldoret corridor)
      northern: 0.25, // 25% to north (Thika/Nyeri/Meru)
    };

    const rand = Math.random();
    let selectedLane;
    if (rand < 0.4) selectedLane = "coastal";
    else if (rand < 0.75) selectedLane = "western";
    else selectedLane = "northern";

    const laneDestinations = KENYA_DESTINATIONS.filter(
      (d) => d.lane === selectedLane
    );
    const destination =
      laneDestinations[Math.floor(Math.random() * laneDestinations.length)];

    // Realistic package weights - weighted towards common sizes
    const weights = [50, 100, 150, 200, 250, 300, 500, 750, 1000];
    const weightProbs = [0.15, 0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02];

    let cumProb = 0;
    const randWeight = Math.random();
    let selectedWeight = weights[0];

    for (let i = 0; i < weights.length; i++) {
      cumProb += weightProbs[i];
      if (randWeight <= cumProb) {
        selectedWeight = weights[i];
        break;
      }
    }

    // Priority distribution - mostly standard
    const priorities = [
      "standard",
      "standard",
      "standard",
      "urgent",
      "emergency",
    ];
    const priority = priorities[Math.floor(Math.random() * priorities.length)];

    return {
      customer_name: `Customer ${Math.floor(Math.random() * 1000)}`,
      customer_phone: `+254${Math.floor(
        Math.random() * 900000000 + 100000000
      )}`,
      pickup_location: {
        address: pickup.name,
        latitude: pickup.lat,
        longitude: pickup.lon,
      },
      delivery_location: {
        address: destination.name,
        latitude: destination.lat,
        longitude: destination.lon,
      },
      package_weight: selectedWeight,
      volume_m3: parseFloat(
        (selectedWeight / 200 + Math.random() * 0.5).toFixed(2)
      ),
      priority: priority,
    };
  }, []);

  const createOrder = useCallback(
    async (orderData) => {
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
            `Order ${result.order_id} created: ${orderData.pickup_location.address} → ${orderData.delivery_location.address}`,
            "success"
          );
          await fetchPendingOrders();
          return result;
        } else {
          const error = await response.text();
          addEvent("order_failed", `Failed to create order: ${error}`, "error");
        }
      } catch (error) {
        addEvent(
          "order_error",
          `Order creation error: ${error.message}`,
          "error"
        );
      }
    },
    [addEvent, fetchPendingOrders]
  );

  // ============================================================================
  // CONSOLIDATION & DISPATCH
  // ============================================================================

  const triggerConsolidation = useCallback(async () => {
    try {
      addEvent("consolidation_start", "Running consolidation cycle...", "info");

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
          "consolidation_complete",
          `Consolidated: ${result.orders_dispatched || 0} orders, ${
            result.batches_created || 0
          } batches, Function: ${result.function_class_used || "N/A"}`,
          "success"
        );

        // Update queues
        await fetchPendingOrders();
        await fetchActiveRoutes();
        await fetchLearningMetrics();

        return result;
      } else {
        const error = await response.text();
        addEvent(
          "consolidation_failed",
          `Consolidation failed: ${error}`,
          "error"
        );
      }
    } catch (error) {
      addEvent(
        "consolidation_error",
        `Consolidation error: ${error.message}`,
        "error"
      );
    }
  }, [addEvent, fetchPendingOrders, fetchActiveRoutes, fetchLearningMetrics]);

  // ============================================================================
  // ROUTE COMPLETION SIMULATION
  // ============================================================================

  const simulateRouteCompletion = useCallback(
    async (route) => {
      // Simulate route completion with realistic outcomes
      const baseDeliveryTime = route.estimated_duration_hours || 5;
      const actualTime = baseDeliveryTime * (0.9 + Math.random() * 0.3);

      const baseCost = route.estimated_cost || route.total_cost || 15000;
      const actualCost = baseCost * (0.85 + Math.random() * 0.25);

      const shipmentCount = route.shipment_ids?.length || 0;
      const successRate = 0.9 + Math.random() * 0.1;

      const outcome = {
        route_id: route.id,
        completed_at: new Date().toISOString(),
        shipments_delivered: Math.floor(shipmentCount * successRate),
        total_shipments: shipmentCount,
        actual_cost: actualCost,
        predicted_cost: baseCost,
        actual_duration_hours: actualTime,
        predicted_duration_hours: baseDeliveryTime,
        utilization: route.utilization_weight || 0.75,
        sla_compliance: successRate >= 0.95,
        delays: [],
        issues: [],
      };

      try {
        const response = await fetch(
          `${API_BASE}/routes/${route.id}/complete`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(outcome),
          }
        );

        if (response.ok) {
          addEvent(
            "route_completed",
            `Route ${route.id} completed: ${outcome.shipments_delivered}/${outcome.total_shipments} delivered`,
            outcome.sla_compliance ? "success" : "warning"
          );

          setInTransitRoutes((prev) => prev.filter((r) => r.id !== route.id));
          setCompletedRoutes((prev) => [
            ...prev.slice(-19),
            { ...route, ...outcome },
          ]);

          await fetchLearningMetrics();
        }
      } catch (error) {
        addEvent(
          "route_completion_error",
          `Failed to complete route ${route.id}: ${error.message}`,
          "error"
        );
      }
    },
    [addEvent, fetchLearningMetrics]
  );

  // ============================================================================
  // SIMULATION LOOPS
  // ============================================================================

  // Order generation loop
  useEffect(() => {
    if (!simulatorRunning) return;

    const intervalMs = (60 / orderRate) * 1000;

    const generate = async () => {
      const order = generateRealisticOrder();
      await createOrder(order);
    };

    generate(); // Immediate first order

    const interval = setInterval(generate, intervalMs);
    return () => clearInterval(interval);
  }, [simulatorRunning, orderRate, generateRealisticOrder, createOrder]);

  // Consolidation loop
  useEffect(() => {
    if (!simulatorRunning) return;

    const intervalMs = consolidationInterval * 60 * 1000;

    const consolidate = async () => {
      await triggerConsolidation();
    };

    consolidate(); // Immediate first consolidation

    const interval = setInterval(consolidate, intervalMs);
    return () => clearInterval(interval);
  }, [simulatorRunning, consolidationInterval, triggerConsolidation]);

  // Route transit simulation loop
  useEffect(() => {
    if (!simulatorRunning || inTransitRoutes.length === 0) return;

    const interval = setInterval(() => {
      // Randomly complete routes (simulate transit)
      if (inTransitRoutes.length > 0 && Math.random() < 0.3) {
        const routeToComplete =
          inTransitRoutes[Math.floor(Math.random() * inTransitRoutes.length)];
        simulateRouteCompletion(routeToComplete);
      }
    }, transitSimSpeed * 1000);

    return () => clearInterval(interval);
  }, [
    simulatorRunning,
    inTransitRoutes,
    transitSimSpeed,
    simulateRouteCompletion,
  ]);

  // ============================================================================
  // SIMULATION CONTROLS
  // ============================================================================

  const startSimulator = () => {
    setSimulatorRunning(true);
    addEvent("simulator_start", "Workflow simulator started", "info");
  };

  const stopSimulator = () => {
    setSimulatorRunning(false);
    addEvent("simulator_stop", "Workflow simulator stopped", "info");
  };

  const resetSimulation = async () => {
    stopSimulator();
    setPendingOrders([]);
    setConsolidationQueue([]);
    setDispatchedRoutes([]);
    setInTransitRoutes([]);
    setCompletedRoutes([]);
    setEventLog([]);
    setPerformanceHistory([]);
    addEvent("simulator_reset", "Simulation reset", "info");
  };

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
      }, 3000);

      const metricsInterval = setInterval(() => {
        fetchLearningMetrics();
      }, 10000);

      return () => {
        clearInterval(refreshInterval);
        clearInterval(metricsInterval);
      };
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
              <h1>Senga SDE - Learning Engine</h1>
              <p>Real-time consolidation with closed-loop learning</p>
            </div>
          </div>

          <div className="header-status">
            <StatusBadge status={systemStatus?.engine_status || "idle"} />
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
            className={`tab ${activeTab === "workflow" ? "active" : ""}`}
            onClick={() => setActiveTab("workflow")}
          >
            <Activity size={16} /> Workflow
          </button>
          <button
            className={`tab ${activeTab === "learning" ? "active" : ""}`}
            onClick={() => setActiveTab("learning")}
          >
            <TrendingUp size={16} /> Learning Metrics
          </button>
          <button
            className={`tab ${activeTab === "locations" ? "active" : ""}`}
            onClick={() => setActiveTab("locations")}
          >
            <MapPin size={16} /> Locations
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === "workflow" && (
            <WorkflowTab
              simulatorRunning={simulatorRunning}
              orderRate={orderRate}
              consolidationInterval={consolidationInterval}
              transitSimSpeed={transitSimSpeed}
              onStart={startSimulator}
              onStop={stopSimulator}
              onReset={resetSimulation}
              onOrderRateChange={setOrderRate}
              onConsolidationIntervalChange={setConsolidationInterval}
              onTransitSimSpeedChange={setTransitSimSpeed}
              onTriggerConsolidation={triggerConsolidation}
              pendingOrders={pendingOrders}
              inTransitRoutes={inTransitRoutes}
              completedRoutes={completedRoutes}
              eventLog={eventLog}
              systemStatus={systemStatus}
            />
          )}

          {activeTab === "learning" && (
            <LearningTab
              metrics={learningMetrics}
              performanceHistory={performanceHistory}
              completedRoutes={completedRoutes}
            />
          )}

          {activeTab === "locations" && <LocationsTab />}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// WORKFLOW TAB
// ============================================================================

function WorkflowTab({
  simulatorRunning,
  orderRate,
  consolidationInterval,
  transitSimSpeed,
  onStart,
  onStop,
  onReset,
  onOrderRateChange,
  onConsolidationIntervalChange,
  onTransitSimSpeedChange,
  onTriggerConsolidation,
  pendingOrders,
  inTransitRoutes,
  completedRoutes,
  eventLog,
  systemStatus,
}) {
  return (
    <div className="workflow-tab">
      {/* Simulation Controls */}
      <div className="card">
        <h2>Workflow Simulator</h2>

        <div className="control-row">
          {!simulatorRunning ? (
            <button className="btn btn-success" onClick={onStart}>
              <Play size={16} /> Start Simulator
            </button>
          ) : (
            <button className="btn btn-warning" onClick={onStop}>
              <Pause size={16} /> Stop Simulator
            </button>
          )}
          <button className="btn btn-secondary" onClick={onReset}>
            <RotateCcw size={16} /> Reset
          </button>
          <button className="btn btn-primary" onClick={onTriggerConsolidation}>
            <TruckIcon size={16} /> Trigger Consolidation
          </button>
        </div>

        <div className="controls-grid">
          <div className="control-group">
            <label>Order Rate (orders/min)</label>
            <input
              type="range"
              min="1"
              max="10"
              value={orderRate}
              onChange={(e) => onOrderRateChange(Number(e.target.value))}
            />
            <span>{orderRate}</span>
          </div>

          <div className="control-group">
            <label>Consolidation Interval (min)</label>
            <input
              type="range"
              min="1"
              max="15"
              value={consolidationInterval}
              onChange={(e) =>
                onConsolidationIntervalChange(Number(e.target.value))
              }
            />
            <span>{consolidationInterval}</span>
          </div>

          <div className="control-group">
            <label>Transit Speed (sec/hour)</label>
            <input
              type="range"
              min="5"
              max="30"
              value={transitSimSpeed}
              onChange={(e) => onTransitSimSpeedChange(Number(e.target.value))}
            />
            <span>{transitSimSpeed}</span>
          </div>
        </div>
      </div>

      {/* Workflow Stages */}
      <div className="workflow-stages">
        <div className="stage-card">
          <div className="stage-header">
            <PackageIcon size={20} />
            <h3>Pending Orders</h3>
            <span className="stage-count">{pendingOrders.length}</span>
          </div>
          <div className="stage-content">
            {pendingOrders.slice(0, 5).map((order) => (
              <div key={order.order_id} className="order-item">
                <div className="order-route">
                  <span className="location-from">
                    {order.pickup_location.address || "Nairobi"}
                  </span>
                  <ArrowRight size={14} />
                  <span className="location-to">
                    {order.delivery_location.address || "Destination"}
                  </span>
                </div>
                <div className="order-meta">
                  {order.package_weight}kg • {order.priority}
                </div>
              </div>
            ))}
            {pendingOrders.length > 5 && (
              <div className="more-items">+{pendingOrders.length - 5} more</div>
            )}
          </div>
        </div>

        <div className="stage-card">
          <div className="stage-header">
            <Clock size={20} />
            <h3>Consolidation Queue</h3>
            <span className="stage-count">
              {Math.min(pendingOrders.length, 8)}
            </span>
          </div>
          <div className="stage-content">
            <div className="consolidation-info">
              Waiting for consolidation cycle...
            </div>
          </div>
        </div>

        <div className="stage-card">
          <div className="stage-header">
            <TruckIcon size={20} />
            <h3>In Transit</h3>
            <span className="stage-count">{inTransitRoutes.length}</span>
          </div>
          <div className="stage-content">
            {inTransitRoutes.slice(0, 5).map((route) => (
              <div key={route.id} className="route-item">
                <div className="route-id">Route {route.id}</div>
                <div className="route-meta">
                  {route.shipment_ids?.length || 0} orders • Vehicle{" "}
                  {route.vehicle_id}
                </div>
              </div>
            ))}
            {inTransitRoutes.length > 5 && (
              <div className="more-items">
                +{inTransitRoutes.length - 5} more
              </div>
            )}
          </div>
        </div>

        <div className="stage-card">
          <div className="stage-header">
            <CheckCircle size={20} />
            <h3>Completed</h3>
            <span className="stage-count">{completedRoutes.length}</span>
          </div>
          <div className="stage-content">
            {completedRoutes
              .slice(-5)
              .reverse()
              .map((route) => (
                <div
                  key={route.route_id || route.id}
                  className="completed-item"
                >
                  <div className="route-id">
                    Route {route.route_id || route.id}
                  </div>
                  <div className="completion-meta">
                    {route.shipments_delivered ||
                      route.shipment_ids?.length ||
                      0}{" "}
                    delivered
                    {route.sla_compliance && (
                      <span className="success-badge">✓ On-time</span>
                    )}
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Event Log */}
      <div className="card">
        <h2>Event Log</h2>
        <div className="event-log">
          {eventLog.slice(0, 20).map((event) => (
            <div key={event.id} className={`event event-${event.level}`}>
              <span className="event-time">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
              <span className="event-message">{event.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// LEARNING TAB
// ============================================================================

function LearningTab({ metrics, performanceHistory, completedRoutes }) {
  if (!metrics) {
    return (
      <div className="learning-tab">
        <div className="card">
          <p>Loading learning metrics...</p>
        </div>
      </div>
    );
  }

  const avgUtilization =
    completedRoutes.length > 0
      ? completedRoutes.reduce((sum, r) => sum + (r.utilization || 0), 0) /
        completedRoutes.length
      : 0;

  const avgOnTimeRate =
    completedRoutes.length > 0
      ? completedRoutes.filter((r) => r.sla_compliance).length /
        completedRoutes.length
      : 0;

  return (
    <div className="learning-tab">
      {/* Key Learning Indicators */}
      <div className="metrics-grid">
        <MetricCard
          label="VFA Updates"
          value={metrics.num_updates || 0}
          subtitle="Learning iterations"
        />
        <MetricCard
          label="Avg TD Error"
          value={(metrics.avg_error || 0).toFixed(3)}
          subtitle="Prediction accuracy"
        />
        <MetricCard
          label="Convergence"
          value={`${((metrics.convergence_score || 0) * 100).toFixed(1)}%`}
          subtitle="Learning stability"
        />
        <MetricCard
          label="Routes Completed"
          value={completedRoutes.length}
          subtitle="Training samples"
        />
      </div>

      {/* Performance Metrics */}
      <div className="card">
        <h2>Operational Performance</h2>
        <div className="metrics-grid">
          <MetricCard
            label="Avg Utilization"
            value={`${(avgUtilization * 100).toFixed(1)}%`}
            subtitle="Vehicle capacity usage"
          />
          <MetricCard
            label="On-Time Rate"
            value={`${(avgOnTimeRate * 100).toFixed(1)}%`}
            subtitle="SLA compliance"
          />
        </div>
      </div>

      {/* TD Error History */}
      {metrics.td_error_history && metrics.td_error_history.length > 0 && (
        <div className="card">
          <h2>TD Error Convergence</h2>
          <SimpleLineChart
            data={metrics.td_error_history}
            label="TD Error"
            color="#dc3545"
          />
        </div>
      )}

      {/* Performance History */}
      {performanceHistory.length > 0 && (
        <div className="card">
          <h2>Reward Trend</h2>
          <SimpleLineChart
            data={performanceHistory}
            label="Avg Reward"
            color="#28a745"
          />
        </div>
      )}

      {/* Feature Weights */}
      {metrics.feature_weights &&
        Object.keys(metrics.feature_weights).length > 0 && (
          <div className="card">
            <h2>Top Feature Weights (VFA)</h2>
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
    </div>
  );
}

// ============================================================================
// LOCATIONS TAB
// ============================================================================

function LocationsTab() {
  return (
    <div className="locations-tab">
      <div className="card">
        <h2>Nairobi Pickup Hubs</h2>
        <div className="location-list">
          {NAIROBI_PICKUP_HUBS.map((loc) => (
            <div key={loc.id} className="location-item">
              <MapPin size={16} />
              <div className="location-details">
                <div className="location-name">{loc.name}</div>
                <div className="location-coords">
                  {loc.lat.toFixed(4)}, {loc.lon.toFixed(4)} • {loc.type}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Kenya-wide Destinations</h2>

        <h3>Coastal Lane (Mombasa Road)</h3>
        <div className="location-list">
          {KENYA_DESTINATIONS.filter((d) => d.lane === "coastal").map((loc) => (
            <div key={loc.id} className="location-item">
              <MapPin size={16} />
              <div className="location-details">
                <div className="location-name">{loc.name}</div>
                <div className="location-coords">
                  {loc.distance_km}km from Nairobi • {loc.lat.toFixed(4)},{" "}
                  {loc.lon.toFixed(4)}
                </div>
              </div>
            </div>
          ))}
        </div>

        <h3>Western Lane (Nakuru-Eldoret-Kisumu)</h3>
        <div className="location-list">
          {KENYA_DESTINATIONS.filter((d) => d.lane === "western").map((loc) => (
            <div key={loc.id} className="location-item">
              <MapPin size={16} />
              <div className="location-details">
                <div className="location-name">{loc.name}</div>
                <div className="location-coords">
                  {loc.distance_km}km from Nairobi • {loc.lat.toFixed(4)},{" "}
                  {loc.lon.toFixed(4)}
                </div>
              </div>
            </div>
          ))}
        </div>

        <h3>Northern Lane (Thika-Nyeri-Meru)</h3>
        <div className="location-list">
          {KENYA_DESTINATIONS.filter((d) => d.lane === "northern").map(
            (loc) => (
              <div key={loc.id} className="location-item">
                <MapPin size={16} />
                <div className="location-details">
                  <div className="location-name">{loc.name}</div>
                  <div className="location-coords">
                    {loc.distance_km}km from Nairobi • {loc.lat.toFixed(4)},{" "}
                    {loc.lon.toFixed(4)}
                  </div>
                </div>
              </div>
            )
          )}
        </div>
      </div>
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
    autonomous: { color: "blue", label: "AUTONOMOUS" },
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
