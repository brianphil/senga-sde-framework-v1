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

  const getStatus = (route: any) => {
    if (route.completed_at) return "Completed";
    if (route.started_at) return "Active";
    return "Planned";
  };

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

  // Helper function to safely get stops count
  const getStopsCount = (route: any) => {
    return route.sequence?.length || route.stops?.length || 0;
  };

  // Helper function to safely get distance
  const getDistance = (route: any) => {
    return route.estimated_distance_km || route.estimated_distance || 0;
  };

  // Helper function to safely get duration in hours
  const getDurationHours = (route: any) => {
    const durationSeconds =
      route.estimated_duration_hours || route.estimated_duration || 0;
    return durationSeconds / 3600; // Convert seconds to hours
  };

  // Helper function to estimate cost (you might want to adjust this logic)
  const getEstimatedCost = (route: any) => {
    const distance = getDistance(route);
    const duration = getDurationHours(route);
    // Simple cost calculation - adjust based on your business logic
    return distance * 50 + duration * 1000; // Example: KES 50 per km + KES 1000 per hour
  };

  // Helper function to get utilization (you might want to adjust this logic)
  const getUtilization = (route: any) => {
    return route.utilization_weight || route.utilization || 0.7; // Default to 70% if not provided
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
          {routes.map((route) => {
            const status = getStatus(route);
            const stopsCount = getStopsCount(route);
            const distance = getDistance(route);
            const durationHours = getDurationHours(route);
            const cost = getEstimatedCost(route);
            const utilization = getUtilization(route);

            return (
              <tr key={route.id}>
                <td className="mono">{route.id.slice(0, 8)}...</td>
                <td>
                  <div className="vehicle-display">
                    <TruckIcon size={14} />
                    <span>{route.vehicle_id}</span>
                  </div>
                </td>
                <td>
                  <div className="shipment-count">
                    <Package size={14} />
                    <span>{route.shipment_ids?.length || 0}</span>
                  </div>
                </td>
                <td>{stopsCount} stops</td>
                <td>{distance.toFixed(1)} km</td>
                <td>{durationHours.toFixed(1)}h</td>
                <td>KES {cost.toFixed(0)}</td>
                <td>
                  <div className="utilization-bar">
                    <div
                      className="utilization-fill"
                      style={{
                        width: `${utilization * 100}%`,
                        backgroundColor:
                          utilization > 0.8
                            ? "#28a745"
                            : utilization > 0.5
                            ? "#ffc107"
                            : "#dc3545",
                      }}
                    />
                    <span className="utilization-text">
                      {(utilization * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td>
                  <span className={`status-badge ${getStatusClass(status)}`}>
                    {status}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
