// src/components/ShipmentsTable.tsx

import React from "react";
import { AlertCircle, Clock, MapPin } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Shipment } from "../types";

interface ShipmentsTableProps {
  shipments: Shipment[];
  loading?: boolean;
}

export default function ShipmentsTable({
  shipments,
  loading,
}: ShipmentsTableProps) {
  const getPriorityBadge = (priority: Shipment["priority"]) => {
    const styles = {
      emergency: "bg-red-100 text-red-800 border-red-300",
      urgent: "bg-yellow-100 text-yellow-800 border-yellow-300",
      standard: "bg-green-100 text-green-800 border-green-300",
    };

    return (
      <span
        className={`px-2 py-1 text-xs font-medium rounded border ${styles[priority]}`}
      >
        {priority.toUpperCase()}
      </span>
    );
  };

  const getDeadlineBadge = (hours?: number) => {
    if (!hours) return null;

    if (hours <= 4) {
      return (
        <span className="text-red-600 flex items-center gap-1 text-sm">
          <AlertCircle size={14} /> Critical
        </span>
      );
    } else if (hours <= 12) {
      return (
        <span className="text-yellow-600 flex items-center gap-1 text-sm">
          <Clock size={14} /> Soon
        </span>
      );
    }
    return (
      <span className="text-gray-600 text-sm">
        {hours.toFixed(0)}h remaining
      </span>
    );
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-semibold mb-4">Pending Shipments</h2>
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-16 bg-gray-100 animate-pulse rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (shipments.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-semibold mb-4">Pending Shipments</h2>
        <div className="text-center py-12 text-gray-500">
          <div className="text-6xl mb-4">📦</div>
          <p>No pending shipments</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Pending Shipments</h2>
          <span className="text-sm text-gray-600">
            {shipments.length} orders
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Order ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Route
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Priority
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Deadline
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Weight/Volume
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Value
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {shipments.map((shipment) => (
              <tr
                key={shipment.id}
                className="hover:bg-gray-50 cursor-pointer transition-colors"
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {shipment.order_id.slice(0, 8)}...
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatDistanceToNow(new Date(shipment.created_at), {
                      addSuffix: true,
                    })}
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-start gap-2">
                    <MapPin
                      size={14}
                      className="text-gray-400 mt-0.5 flex-shrink-0"
                    />
                    <div className="text-sm">
                      <div className="text-gray-900 font-medium truncate max-w-xs">
                        {shipment.pickup_location}
                      </div>
                      <div className="text-gray-500 truncate max-w-xs">
                        → {shipment.delivery_location}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getPriorityBadge(shipment.priority)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getDeadlineBadge(shipment.time_to_deadline_hours)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <div>{shipment.weight.toFixed(1)} kg</div>
                  <div className="text-gray-500">
                    {shipment.volume.toFixed(2)} m³
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  ${shipment.declared_value.toFixed(0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
