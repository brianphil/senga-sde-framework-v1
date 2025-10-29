import { useState, useEffect } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Zap,
  Activity,
  TrendingDown,
} from "lucide-react";

const API = "/api";

export default function TestingUI() {
  const [running, setRunning] = useState(false);
  const [pending, setPending] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [completed, setCompleted] = useState([]);
  const [vfa, setVfa] = useState(null);
  const [logs, setLogs] = useState([]);

  const log = (msg, type = "info") => {
    console.log(msg);
    setLogs((prev) => [{ t: Date.now(), msg, type }, ...prev].slice(0, 30));
  };

  const fetchAll = async () => {
    try {
      const [p, r, v] = await Promise.all([
        fetch(`${API}/orders/pending`)
          .then((r) => r.json())
          .catch(() => []),
        fetch(`${API}/routes/active`)
          .then((r) => r.json())
          .then((d) => d.routes || [])
          .catch(() => []),
        fetch(`${API}/learning/vfa/metrics`)
          .then((r) => r.json())
          .catch(() => null),
      ]);
      setPending(p);
      setRoutes(r);
      setVfa(v);
    } catch (e) {
      console.error("Fetch failed:", e);
    }
  };

  useEffect(() => {
    fetchAll();
    const int = setInterval(fetchAll, 2000);
    return () => clearInterval(int);
  }, []);

  const createOrder = async () => {
    const pickups = [
      "Industrial Area",
      "Westlands",
      "CBD Kenyatta Ave",
      "Eastleigh",
    ];
    const dests = ["Thika", "Nakuru", "Mombasa", "Kisumu", "Eldoret"];
    const pickup = pickups[Math.floor(Math.random() * pickups.length)];
    const dest = dests[Math.floor(Math.random() * dests.length)];

    const locs = {
      "Industrial Area": [-1.3167, 36.85],
      Westlands: [-1.2676, 36.807],
      "CBD Kenyatta Ave": [-1.2864, 36.8172],
      Eastleigh: [-1.2833, 36.85],
      Thika: [-1.0332, 37.069],
      Nakuru: [-0.3031, 36.08],
      Mombasa: [-4.0435, 39.6682],
      Kisumu: [-0.0917, 34.768],
      Eldoret: [0.5143, 35.2698],
    };

    const order = {
      customer_name: `Customer${Math.floor(Math.random() * 100)}`,
      customer_phone: `+254700${String(
        Math.floor(Math.random() * 1000000)
      ).padStart(6, "0")}`,
      pickup_location: {
        address: pickup,
        latitude: locs[pickup][0],
        longitude: locs[pickup][1],
      },
      delivery_location: {
        address: dest,
        latitude: locs[dest][0],
        longitude: locs[dest][1],
      },
      package_weight: [100, 250, 500, 1000][Math.floor(Math.random() * 4)],
      volume_m3: parseFloat((Math.random() * 2).toFixed(2)),
      priority: Math.random() > 0.7 ? "urgent" : "standard",
    };

    try {
      await fetch(`${API}/orders`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(order),
      });
      log(`✓ Order: ${pickup} → ${dest}`, "success");
      setTimeout(fetchAll, 300);
    } catch (e) {
      log(`✗ Order failed: ${e.message}`, "error");
    }
  };

  const triggerCycle = async () => {
    try {
      const res = await fetch(`${API}/decisions/consolidation-cycle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      log(
        `Cycle: ${data.function_class_used} - ${data.orders_dispatched} dispatched`,
        "info"
      );
      setTimeout(fetchAll, 500);
    } catch (e) {
      log(`✗ Cycle failed: ${e.message}`, "error");
    }
  };

  const completeRoute = async (route) => {
    const dist = route.estimated_distance_km || 100;
    const delay = Math.random() < 0.3;
    const cost = dist * 45 + (delay ? 800 : 0);
    const duration = dist / 60 + (delay ? 1.5 : 0);

    try {
      await fetch(`${API}/route/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          route_id: route.route_id,
          actual_cost: cost,
          actual_duration_hours: duration,
          shipments_delivered: route.shipments?.length || 1,
          total_shipments: route.shipments?.length || 1,
          sla_compliant: !delay,
          delays: delay ? [{ reason: "traffic", duration_minutes: 60 }] : [],
          issues: delay ? ["Traffic jam"] : [],
        }),
      });
      log(
        `✓ Route completed ${delay ? "(delayed)" : "(on-time)"}`,
        delay ? "warning" : "success"
      );
      setCompleted((prev) => [...prev, { ...route, sla: !delay }]);
      setTimeout(fetchAll, 500);
    } catch (e) {
      log(`✗ Complete failed: ${e.message}`, "error");
    }
  };

  useEffect(() => {
    if (!running) return;
    const int1 = setInterval(createOrder, 20000); // Every 20s
    const int2 = setInterval(triggerCycle, 45000); // Every 45s
    return () => {
      clearInterval(int1);
      clearInterval(int2);
    };
  }, [running]);

  return (
    <div
      style={{
        padding: "24px",
        fontFamily: "system-ui",
        maxWidth: "1800px",
        margin: "0 auto",
        background: "#fafafa",
      }}
    >
      {/* Header */}
      <div
        style={{
          marginBottom: "24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "32px", fontWeight: "700" }}>
            Senga SDE Test Console
          </h1>
          <p style={{ margin: "8px 0 0", color: "#666" }}>
            Interactive Learning Validation
          </p>
        </div>
        <div style={{ display: "flex", gap: "12px" }}>
          <button
            onClick={() => setRunning(!running)}
            style={{
              padding: "12px 24px",
              background: running ? "#dc2626" : "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: "600",
              fontSize: "15px",
            }}
          >
            {running ? "⏸ Stop Auto" : "▶ Start Auto"}
          </button>
          <button
            onClick={createOrder}
            style={{
              padding: "12px 24px",
              background: "#059669",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: "600",
              fontSize: "15px",
            }}
          >
            + Add Order
          </button>
          <button
            onClick={triggerCycle}
            style={{
              padding: "12px 24px",
              background: "#7c3aed",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: "600",
              fontSize: "15px",
            }}
          >
            <Zap size={16} style={{ display: "inline", marginRight: "4px" }} />
            Force Consolidate
          </button>
        </div>
      </div>

      {/* Metrics Bar */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5,1fr)",
          gap: "16px",
          marginBottom: "24px",
        }}
      >
        <MetricCard
          label="Pending Orders"
          value={pending.length}
          color="#3b82f6"
        />
        <MetricCard
          label="Active Routes"
          value={routes.length}
          color="#059669"
        />
        <MetricCard
          label="Completed"
          value={completed.length}
          color="#8b5cf6"
        />
        <MetricCard
          label="VFA Updates"
          value={vfa?.total_updates || 0}
          color="#f59e0b"
        />
        <MetricCard
          label="TD Error"
          value={vfa?.recent_avg_td_error?.toFixed(1) || 0}
          color={vfa?.recent_avg_td_error < 100 ? "#10b981" : "#ef4444"}
        />
      </div>

      {/* Main Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "2fr 3fr 2fr",
          gap: "20px",
        }}
      >
        {/* Pending */}
        <Card title="📦 Pending Orders" count={pending.length}>
          <div style={{ maxHeight: "600px", overflow: "auto" }}>
            {pending.map((o, i) => (
              <OrderCard key={i} order={o} />
            ))}
            {pending.length === 0 && <Empty msg="No pending orders" />}
          </div>
        </Card>

        {/* Routes */}
        <Card title="🚚 Dispatched Routes" count={routes.length}>
          <div style={{ maxHeight: "600px", overflow: "auto" }}>
            {routes.map((r, i) => (
              <RouteCard
                key={i}
                route={r}
                onComplete={() => completeRoute(r)}
              />
            ))}
            {routes.length === 0 && (
              <Empty msg="No active routes. Add orders and consolidate." />
            )}
          </div>
        </Card>

        {/* Completed */}
        <Card title="✅ Completed" count={completed.length}>
          <div style={{ maxHeight: "600px", overflow: "auto" }}>
            {completed.slice(0, 15).map((r, i) => (
              <CompletedCard key={i} route={r} />
            ))}
            {completed.length === 0 && (
              <Empty msg="Complete routes to see learning feedback" />
            )}
          </div>
        </Card>
      </div>

      {/* Log */}
      <div
        style={{
          marginTop: "20px",
          background: "white",
          padding: "20px",
          borderRadius: "12px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
        }}
      >
        <h3 style={{ margin: "0 0 12px", fontSize: "18px", fontWeight: "600" }}>
          <Activity
            size={18}
            style={{ display: "inline", marginRight: "6px" }}
          />
          System Log
        </h3>
        <div
          style={{
            maxHeight: "180px",
            overflow: "auto",
            fontSize: "13px",
            fontFamily: "monospace",
          }}
        >
          {logs.map((l, i) => (
            <div
              key={i}
              style={{
                padding: "6px 10px",
                marginBottom: "2px",
                borderRadius: "4px",
                background:
                  l.type === "error"
                    ? "#fee2e2"
                    : l.type === "success"
                    ? "#d1fae5"
                    : l.type === "warning"
                    ? "#fef3c7"
                    : "#f3f4f6",
              }}
            >
              <span style={{ color: "#9ca3af", marginRight: "8px" }}>
                {new Date(l.t).toLocaleTimeString()}
              </span>
              {l.msg}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }) {
  return (
    <div
      style={{
        background: "white",
        padding: "20px",
        borderRadius: "12px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
      }}
    >
      <div
        style={{
          fontSize: "13px",
          color: "#6b7280",
          marginBottom: "8px",
          fontWeight: "500",
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: "32px", fontWeight: "700", color }}>{value}</div>
    </div>
  );
}

function Card({ title, count, children }) {
  return (
    <div
      style={{
        background: "white",
        padding: "20px",
        borderRadius: "12px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
      }}
    >
      <h3 style={{ margin: "0 0 16px", fontSize: "18px", fontWeight: "600" }}>
        {title} <span style={{ color: "#9ca3af" }}>({count})</span>
      </h3>
      {children}
    </div>
  );
}

function OrderCard({ order }) {
  return (
    <div
      style={{
        padding: "12px",
        marginBottom: "10px",
        background: "#f9fafb",
        borderRadius: "8px",
        borderLeft: `4px solid ${
          order.priority === "urgent" ? "#f59e0b" : "#3b82f6"
        }`,
      }}
    >
      <div style={{ fontWeight: "600", fontSize: "14px", marginBottom: "4px" }}>
        {order.pickup_location?.address || "Unknown"} →{" "}
        {order.delivery_location?.address || "Unknown"}
      </div>
      <div style={{ fontSize: "12px", color: "#6b7280" }}>
        {order.package_weight}kg • {order.volume_m3}m³ • {order.priority}
      </div>
    </div>
  );
}

function RouteCard({ route, onComplete }) {
  const shipments = route.shipments || [];
  return (
    <div
      style={{
        padding: "16px",
        marginBottom: "12px",
        background: "#f0fdf4",
        borderRadius: "8px",
        border: "2px solid #10b981",
      }}
    >
      <div
        style={{
          fontSize: "11px",
          color: "#6b7280",
          marginBottom: "8px",
          fontFamily: "monospace",
        }}
      >
        {route.route_id?.slice(0, 20)}...
      </div>

      <div style={{ marginBottom: "12px" }}>
        <div
          style={{
            fontWeight: "600",
            fontSize: "14px",
            marginBottom: "8px",
            color: "#059669",
          }}
        >
          {shipments.length} Shipment{shipments.length > 1 ? "s" : ""}
        </div>
        {shipments.slice(0, 3).map((s, i) => (
          <div
            key={i}
            style={{
              fontSize: "12px",
              color: "#374151",
              marginBottom: "4px",
              paddingLeft: "12px",
              borderLeft: "2px solid #10b981",
            }}
          >
            📍 {s.pickup_location?.address} → {s.delivery_location?.address}
          </div>
        ))}
        {shipments.length > 3 && (
          <div
            style={{ fontSize: "11px", color: "#9ca3af", paddingLeft: "12px" }}
          >
            +{shipments.length - 3} more
          </div>
        )}
      </div>

      <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px" }}>
        🚗 {route.estimated_distance_km?.toFixed(0)}km • ⏱️{" "}
        {route.estimated_duration_hours?.toFixed(1)}h
      </div>

      <button
        onClick={onComplete}
        style={{
          width: "100%",
          padding: "10px",
          background: "#059669",
          color: "white",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
          fontWeight: "600",
          fontSize: "14px",
        }}
      >
        ✓ Complete Route (Trigger Learning)
      </button>
    </div>
  );
}

function CompletedCard({ route }) {
  return (
    <div
      style={{
        padding: "12px",
        marginBottom: "8px",
        borderRadius: "8px",
        background: route.sla ? "#d1fae5" : "#fee2e2",
        borderLeft: `4px solid ${route.sla ? "#10b981" : "#ef4444"}`,
      }}
    >
      <div style={{ fontWeight: "600", fontSize: "14px" }}>
        {route.sla ? "✓ On-time" : "✗ Delayed"} • {route.shipments?.length || 0}{" "}
        shipments
      </div>
      <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "4px" }}>
        Learning feedback sent
      </div>
    </div>
  );
}

function Empty({ msg }) {
  return (
    <div
      style={{ textAlign: "center", padding: "60px 20px", color: "#9ca3af" }}
    >
      <div style={{ fontSize: "48px", marginBottom: "12px" }}>📭</div>
      <div style={{ fontSize: "14px" }}>{msg}</div>
    </div>
  );
}
