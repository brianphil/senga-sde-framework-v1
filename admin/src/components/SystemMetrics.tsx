// src/components/SystemMetrics.tsx
import { Package, TruckIcon, MapPin, Activity } from "lucide-react";
import { SystemStatusResponse } from "../types";

interface Props {
  status: SystemStatusResponse | null;
}

export default function SystemMetrics({ status }: Props) {
  if (!status) {
    return (
      <div className="metrics-bar">
        <div className="metric-card">
          <div className="metric-value">—</div>
          <div className="metric-label">Loading...</div>
        </div>
      </div>
    );
  }

  const getStatusColor = (statusValue: string) => {
    switch (statusValue.toLowerCase()) {
      case "operational":
        return "var(--success)";
      case "degraded":
        return "var(--warning)";
      case "offline":
        return "var(--error)";
      default:
        return "var(--text-secondary)";
    }
  };

  return (
    <div className="metrics-bar">
      <div className="metric-card">
        <div className="metric-icon">
          <Activity
            size={20}
            style={{ color: getStatusColor(status.status) }}
          />
        </div>
        <div className="metric-content">
          <div className="metric-value">{status.status}</div>
          <div className="metric-label">System Status</div>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <Package size={20} style={{ color: "var(--primary)" }} />
        </div>
        <div className="metric-content">
          <div className="metric-value">{status.pending_orders}</div>
          <div className="metric-label">Pending Orders</div>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <TruckIcon size={20} style={{ color: "var(--success)" }} />
        </div>
        <div className="metric-content">
          <div className="metric-value">{status.available_vehicles}</div>
          <div className="metric-label">Available Vehicles</div>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <MapPin size={20} style={{ color: "var(--warning)" }} />
        </div>
        <div className="metric-content">
          <div className="metric-value">{status.active_routes}</div>
          <div className="metric-label">Active Routes</div>
        </div>
      </div>
    </div>
  );
}
