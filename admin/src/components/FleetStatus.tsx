// src/components/FleetStatus.tsx

import React from "react";
import { Truck, MapPin } from "lucide-react";
import type { Vehicle } from "../types";

interface FleetStatusProps {
  fleet: Vehicle[];
  loading?: boolean;
}

export default function FleetStatus({ fleet, loading }: FleetStatusProps) {
  const getStatusBadge = (status: Vehicle["status"]) => {
    const styles = {
      idle: "bg-green-100 text-green-800",
      dispatched: "bg-blue-100 text-blue-800",
      in_transit: "bg-yellow-100 text-yellow-800",
      maintenance: "bg-red-100 text-red-800",
    };

    return (
      <span
        className={`px-2 py-0.5 text-xs font-medium rounded ${styles[status]}`}
      >
        {status.toUpperCase()}
      </span>
    );
  };

  const getUtilizationColor = (utilization: number) => {
    if (utilization >= 80) return "bg-red-500";
    if (utilization >= 60) return "bg-yellow-500";
    return "bg-green-500";
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="font-semibold text-gray-900 mb-4">Fleet Status</h3>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 bg-gray-100 animate-pulse rounded" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h3 className="font-semibold text-gray-900">Fleet Status</h3>
        <p className="text-xs text-gray-500 mt-1">{fleet.length} vehicles</p>
      </div>

      <div className="p-4 space-y-3 max-h-80 overflow-y-auto">
        {fleet.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Truck size={32} className="mx-auto mb-2 text-gray-400" />
            <p className="text-sm">No vehicles</p>
          </div>
        ) : (
          fleet.map((vehicle) => (
            <div
              key={vehicle.vehicle_id}
              className="p-3 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Truck size={16} className="text-gray-400" />
                  <div>
                    <div className="text-sm font-medium text-gray-900">
                      {vehicle.vehicle_id.slice(0, 8)}...
                    </div>
                    <div className="text-xs text-gray-500">
                      {vehicle.vehicle_type}
                    </div>
                  </div>
                </div>
                {getStatusBadge(vehicle.status)}
              </div>

              <div className="flex items-start gap-1 text-xs text-gray-600 mb-2">
                <MapPin size={12} className="mt-0.5 flex-shrink-0" />
                <span className="truncate">{vehicle.current_location}</span>
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">Capacity</span>
                  <span className="text-gray-900">
                    {vehicle.capacity_weight_kg}kg /{" "}
                    {vehicle.capacity_volume_m3}m³
                  </span>
                </div>

                {vehicle.current_utilization !== undefined && (
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-gray-600">Utilization</span>
                      <span className="text-gray-900">
                        {vehicle.current_utilization.toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all ${getUtilizationColor(
                          vehicle.current_utilization
                        )}`}
                        style={{ width: `${vehicle.current_utilization}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
