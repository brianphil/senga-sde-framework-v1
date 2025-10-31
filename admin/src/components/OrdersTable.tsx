// src/components/OrdersTable.tsx
import { Package, Clock, TrendingUp } from "lucide-react";
import { OrderResponse } from "../types";

interface Props {
  orders: OrderResponse[];
  loading?: boolean;
}

export default function OrdersTable({ orders, loading }: Props) {
  if (loading) {
    return (
      <div className="loading-state">
        <div className="spinner"></div>
        <p>Loading orders...</p>
      </div>
    );
  }

  if (orders.length === 0) {
    return (
      <div className="empty-state">
        <Package size={48} className="empty-icon" />
        <h3>No Pending Orders</h3>
        <p>Create some test orders to see them here</p>
      </div>
    );
  }

  const getPriorityClass = (priority: string) => {
    switch (priority.toLowerCase()) {
      case "emergency":
        return "priority-emergency";
      case "urgent":
        return "priority-urgent";
      default:
        return "priority-standard";
    }
  };

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Order ID</th>
            <th>Customer</th>
            <th>Route</th>
            <th>Weight</th>
            <th>Volume</th>
            <th>Priority</th>
            <th>Time to Deadline</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((order) => (
            <tr key={order.order_id}>
              <td className="mono">{order.order_id}</td>
              <td>{order.customer_name}</td>
              <td className="route-cell">
                <div className="route-display">
                  <span className="location">
                    {order.pickup_location.address}
                  </span>
                  <span className="arrow">→</span>
                  <span className="location">
                    {order.delivery_location.address}
                  </span>
                </div>
              </td>
              <td>{order.package_weight.toFixed(1)} kg</td>
              <td>{order.volume_m3.toFixed(2)} m³</td>
              <td>
                <span
                  className={`priority-badge ${getPriorityClass(
                    order.priority
                  )}`}
                >
                  {order.priority}
                </span>
              </td>
              <td>
                <div className="time-display">
                  <Clock size={14} />
                  <span>{order.time_to_deadline_hours.toFixed(1)}h</span>
                </div>
              </td>
              <td>
                <span className="status-badge status-pending">
                  {order.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
