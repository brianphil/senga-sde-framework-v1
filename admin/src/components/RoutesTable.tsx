// src/components/RoutesTable.tsx
import { MapPin, TruckIcon, Package } from "lucide-react";
import { Route } from "../types";

interface Props {
  routes: Route[];
  title: string;
  loading?: boolean;
}

export default function RoutesTable({ routes, title, loading }: Props) {
  if (loading) {
    return (
      <div className="loading-state">
        <div className="spinner"></div>
        <p>Loading routes...</p>
      </div>
    );
  }

  if (routes.length === 0) {
    return (
      <div className="empty-state">
        <MapPin size={48} className="empty-icon" />
        <h3>No {title}</h3>
        <p>Routes will appear here when consolidation creates them</p>
      </div>
    );
  }

  const getStatusClass = (status: string) => {
    switch (status.toLowerCase()) {
      case "active":
        return "status-active";
      case "completed":
        return "status-completed";
      case "planned":
        return "status-planned";
      default:
        return "status-default";
    }
  };

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Route ID</th>
            <th>Vehicle</th>
            <th>Shipments</th>
            <th>Stops</th>
            <th>Distance</th>
            <th>Duration</th>
            <th>Cost</th>
            <th>Utilization</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {routes.map((route) => (
            <tr key={route.id}>
              <td className="mono">{route.id}</td>
              <td>
                <div className="vehicle-display">
                  <TruckIcon size={14} />
                  <span>{route.vehicle_id}</span>
                </div>
              </td>
              <td>
                <div className="shipment-count">
                  <Package size={14} />
                  <span>{route.shipment_ids.length}</span>
                </div>
              </td>
              <td>{route.stops.length} stops</td>
              <td>{route.estimated_distance_km.toFixed(1)} km</td>
              <td>{route.estimated_duration_hours.toFixed(1)}h</td>
              <td>KES {route.estimated_cost.toFixed(0)}</td>
              <td>
                <div className="utilization-bar">
                  <div
                    className="utilization-fill"
                    style={{
                      width: `${(route.utilization_weight || 0) * 100}%`,
                      backgroundColor:
                        (route.utilization_weight || 0) > 0.8
                          ? "#28a745"
                          : (route.utilization_weight || 0) > 0.5
                          ? "#ffc107"
                          : "#dc3545",
                    }}
                  />
                  <span className="utilization-text">
                    {((route.utilization_weight || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </td>
              <td>
                <span
                  className={`status-badge ${getStatusClass(route.status)}`}
                >
                  {route.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
