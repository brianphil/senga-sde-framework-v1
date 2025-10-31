// src/App.tsx
import { useState, useEffect, useCallback } from "react";
import {
  TruckIcon,
  Activity,
  AlertCircle,
  PlayCircle,
  PauseCircle,
  RefreshCw,
  Package,
  MapPin,
  Clock,
  TrendingUp,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { api } from "./services/api";
import {
  OrderResponse,
  SystemStatusResponse,
  EventLogEntry,
  Route,
  ConsolidationCycleResponse,
} from "./types";
import OrderCreationPanel from "./components/OrderCreationPanel";
import ConsolidationPanel from "./components/ConsolidationPanel";
import OrdersTable from "./components/OrdersTable";
import RoutesTable from "./components/RoutesTable";
import EventLog from "./components/EventLog";
import SystemMetrics from "./components/SystemMetrics";
import "./App.css";

function App() {
  // System state
  const [apiOnline, setApiOnline] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(
    null
  );
  const [loading, setLoading] = useState(false);

  // Data state
  const [pendingOrders, setPendingOrders] = useState<OrderResponse[]>([]);
  const [activeRoutes, setActiveRoutes] = useState<Route[]>([]);
  const [completedRoutes, setCompletedRoutes] = useState<Route[]>([]);

  // UI state
  const [activeTab, setActiveTab] = useState<"orders" | "routes" | "cycles">(
    "orders"
  );
  const [eventLog, setEventLog] = useState<EventLogEntry[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // ============================================================================
  // Event Logging
  // ============================================================================
  const addEvent = useCallback(
    (type: EventLogEntry["type"], message: string, details?: string) => {
      const event: EventLogEntry = {
        id: `${Date.now()}-${Math.random()}`,
        timestamp: new Date().toISOString(),
        type,
        message,
        details,
      };
      setEventLog((prev) => [event, ...prev].slice(0, 100)); // Keep last 100 events
    },
    []
  );

  // ============================================================================
  // API Calls
  // ============================================================================
  const checkHealth = useCallback(async () => {
    try {
      await api.checkHealth();
      setApiOnline(true);
      return true;
    } catch {
      setApiOnline(false);
      return false;
    }
  }, []);

  const refreshSystemStatus = useCallback(async () => {
    try {
      const status = await api.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      addEvent("error", "Failed to fetch system status", String(error));
    }
  }, [addEvent]);

  const refreshPendingOrders = useCallback(async () => {
    try {
      const orders = await api.getPendingOrders();
      setPendingOrders(orders);
    } catch (error) {
      addEvent("error", "Failed to fetch pending orders", String(error));
    }
  }, [addEvent]);

  const refreshActiveRoutes = useCallback(async () => {
    try {
      const routes = await api.getActiveRoutes();
      setActiveRoutes(routes);
    } catch (error) {
      addEvent("error", "Failed to fetch active routes", String(error));
    }
  }, [addEvent]);

  const refreshCompletedRoutes = useCallback(async () => {
    try {
      const routes = await api.getCompletedRoutes(20);
      setCompletedRoutes(routes);
    } catch (error) {
      addEvent("error", "Failed to fetch completed routes", String(error));
    }
  }, [addEvent]);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    await Promise.all([
      refreshSystemStatus(),
      refreshPendingOrders(),
      refreshActiveRoutes(),
      refreshCompletedRoutes(),
    ]);
    setLoading(false);
  }, [
    refreshSystemStatus,
    refreshPendingOrders,
    refreshActiveRoutes,
    refreshCompletedRoutes,
  ]);

  // ============================================================================
  // Order Creation
  // ============================================================================
  const handleOrderCreated = useCallback(
    async (order: OrderResponse) => {
      addEvent(
        "success",
        `Order ${order.order_id} created`,
        `${order.pickup_location.address} → ${order.delivery_location.address}`
      );
      await refreshPendingOrders();
      await refreshSystemStatus();
    },
    [addEvent, refreshPendingOrders, refreshSystemStatus]
  );

  // ============================================================================
  // Consolidation
  // ============================================================================
  const handleConsolidationComplete = useCallback(
    async (result: ConsolidationCycleResponse) => {
      addEvent(
        "success",
        `Consolidation Complete: ${result.orders_dispatched} orders dispatched, ${result.orders_waiting} waiting`,
        `Function: ${result.function_class_used} | Batches: ${result.batches_created}`
      );
      await refreshAll();
    },
    [addEvent, refreshAll]
  );

  // ============================================================================
  // Initialization & Auto-refresh
  // ============================================================================
  useEffect(() => {
    checkHealth();
    const healthCheck = setInterval(checkHealth, 5000);
    return () => clearInterval(healthCheck);
  }, [checkHealth]);

  useEffect(() => {
    if (apiOnline) {
      refreshAll();
    }
  }, [apiOnline, refreshAll]);

  useEffect(() => {
    if (!autoRefresh || !apiOnline) return;

    const interval = setInterval(refreshAll, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, apiOnline, refreshAll]);

  // ============================================================================
  // Render
  // ============================================================================
  if (!apiOnline) {
    return (
      <div className="error-screen">
        <AlertCircle size={64} className="error-icon" />
        <h1>API Offline</h1>
        <p>Cannot connect to Senga SDE backend</p>
        <p className="error-detail">
          Ensure the API is running at http://localhost:8000
        </p>
        <button onClick={checkHealth} className="btn-primary">
          <RefreshCw size={16} /> Retry Connection
        </button>
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
              <h1>Senga Sequential Decision Engine</h1>
              <p className="header-subtitle">
                Real-time Testing & Monitoring Dashboard
              </p>
            </div>
          </div>

          <div className="header-controls">
            <button
              onClick={refreshAll}
              disabled={loading}
              className="btn-icon"
              title="Refresh All Data"
            >
              <RefreshCw size={18} className={loading ? "spinning" : ""} />
            </button>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`btn-icon ${autoRefresh ? "active" : ""}`}
              title={
                autoRefresh ? "Disable Auto-refresh" : "Enable Auto-refresh"
              }
            >
              {autoRefresh ? (
                <PauseCircle size={18} />
              ) : (
                <PlayCircle size={18} />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* System Metrics Bar */}
      <SystemMetrics status={systemStatus} />

      {/* Main Content */}
      <div className="main-content">
        {/* Left Panel - Controls */}
        <div className="left-panel">
          <OrderCreationPanel onOrderCreated={handleOrderCreated} />
          <ConsolidationPanel
            pendingOrdersCount={pendingOrders.length}
            onConsolidationComplete={handleConsolidationComplete}
          />
        </div>

        {/* Center Panel - Data Display */}
        <div className="center-panel">
          {/* Tabs */}
          <div className="tabs">
            <button
              className={`tab ${activeTab === "orders" ? "active" : ""}`}
              onClick={() => setActiveTab("orders")}
            >
              <Package size={16} /> Pending Orders ({pendingOrders.length})
            </button>
            <button
              className={`tab ${activeTab === "routes" ? "active" : ""}`}
              onClick={() => setActiveTab("routes")}
            >
              <MapPin size={16} /> Active Routes ({activeRoutes.length})
            </button>
            <button
              className={`tab ${activeTab === "cycles" ? "active" : ""}`}
              onClick={() => setActiveTab("cycles")}
            >
              <Activity size={16} /> Completed Routes ({completedRoutes.length})
            </button>
          </div>

          {/* Tab Content */}
          <div className="tab-content">
            {activeTab === "orders" && (
              <OrdersTable orders={pendingOrders} loading={loading} />
            )}
            {activeTab === "routes" && (
              <RoutesTable
                routes={activeRoutes}
                title="Active Routes"
                loading={loading}
              />
            )}
            {activeTab === "cycles" && (
              <RoutesTable
                routes={completedRoutes}
                title="Completed Routes"
                loading={loading}
              />
            )}
          </div>
        </div>

        {/* Right Panel - Event Log */}
        <div className="right-panel">
          <EventLog events={eventLog} onClear={() => setEventLog([])} />
        </div>
      </div>
    </div>
  );
}

export default App;
