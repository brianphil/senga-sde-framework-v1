// src/pages/Dashboard.tsx

import React from "react";
import { Package, Truck, Clock, TrendingUp, PlayCircle } from "lucide-react";
import {
  useSystemStatus,
  usePendingShipments,
  useFleet,
  useRecentCycles,
  useTriggerCycle,
} from "../hooks/useDataSync";
import { useStore } from "../store/useStore";
import { formatDistanceToNow } from "date-fns";

import MetricsCard from "../components/MetricsCard";
import ShipmentsTable from "../components/OrdersTable";
import CycleHistory from "../components/CycleHistory";
import FleetStatus from "../components/FleetStatus";

export default function Dashboard() {
  const { data: status, isLoading: statusLoading } = useSystemStatus();
  const { data: shipments, isLoading: shipmentsLoading } =
    usePendingShipments();
  const { data: fleet, isLoading: fleetLoading } = useFleet();
  const { data: cycles } = useRecentCycles();
  const { trigger } = useTriggerCycle();

  const autoRefresh = useStore((state) => state.autoRefresh);
  const setAutoRefresh = useStore((state) => state.setAutoRefresh);

  const [isTriggering, setIsTriggering] = React.useState(false);

  const handleTriggerCycle = async () => {
    setIsTriggering(true);
    try {
      await trigger();
    } finally {
      setIsTriggering(false);
    }
  };

  const urgentShipments =
    shipments?.filter(
      (s) => s.priority === "emergency" || s.priority === "urgent"
    ) || [];
  const avgUtilization =
    fleet?.reduce((sum, v) => sum + (v.current_utilization || 0), 0) /
      (fleet?.length || 1) || 0;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Senga Operations
            </h1>
            <p className="text-gray-600 mt-1">
              Real-time freight consolidation & routing
            </p>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              Auto-refresh (5s)
            </label>

            <button
              onClick={handleTriggerCycle}
              disabled={isTriggering}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <PlayCircle size={18} />
              {isTriggering ? "Running..." : "Trigger Cycle"}
            </button>
          </div>
        </div>

        {/* Status Banner */}
        {status && (
          <div
            className={`p-4 rounded-lg ${
              status.engine_status === "running"
                ? "bg-green-50 border border-green-200"
                : "bg-yellow-50 border border-yellow-200"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <span
                  className={`inline-flex h-3 w-3 rounded-full ${
                    status.engine_status === "running"
                      ? "bg-green-500 animate-pulse"
                      : "bg-yellow-500"
                  }`}
                />
                <span className="font-medium capitalize">
                  Engine: {status.engine_status}
                </span>
                {status.last_decision_time && (
                  <span className="text-sm text-gray-600">
                    Last decision:{" "}
                    {formatDistanceToNow(new Date(status.last_decision_time), {
                      addSuffix: true,
                    })}
                  </span>
                )}
              </div>
              <div className="text-sm text-gray-600">
                Cycle #{status.current_cycle}
              </div>
            </div>
          </div>
        )}

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricsCard
            title="Pending Shipments"
            value={status?.pending_shipments || 0}
            subtitle={
              urgentShipments.length > 0
                ? `${urgentShipments.length} urgent`
                : "All standard"
            }
            icon={<Package />}
            trend={urgentShipments.length > 0 ? "warning" : "normal"}
            loading={statusLoading}
          />

          <MetricsCard
            title="Available Vehicles"
            value={status?.available_vehicles || 0}
            subtitle={`${fleet?.length || 0} total`}
            icon={<Truck />}
            loading={fleetLoading}
          />

          <MetricsCard
            title="Fleet Utilization"
            value={`${avgUtilization.toFixed(1)}%`}
            subtitle={
              avgUtilization > 80 ? "High capacity" : "Capacity available"
            }
            icon={<TrendingUp />}
            trend={avgUtilization > 80 ? "warning" : "normal"}
            loading={fleetLoading}
          />

          <MetricsCard
            title="SLA Compliance"
            value={`${((status?.avg_sla_compliance || 0) * 100).toFixed(1)}%`}
            subtitle="On-time deliveries"
            icon={<Clock />}
            trend={(status?.avg_sla_compliance || 0) > 0.9 ? "good" : "warning"}
            loading={statusLoading}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Pending Shipments */}
          <div className="lg:col-span-2">
            <ShipmentsTable
              shipments={shipments || []}
              loading={shipmentsLoading}
            />
          </div>

          {/* Right: Recent Cycles + Fleet Status */}
          <div className="space-y-6">
            <CycleHistory cycles={cycles || []} />
            <FleetStatus fleet={fleet || []} loading={fleetLoading} />
          </div>
        </div>
      </div>
    </div>
  );
}
