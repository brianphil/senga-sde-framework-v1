// src/components/CycleHistory.tsx

import React from "react";
import { formatDistanceToNow } from "date-fns";
import { Brain, Clock, TrendingUp, AlertCircle } from "lucide-react";
import type { ConsolidationCycle } from "../types";

interface CycleHistoryProps {
  cycles: ConsolidationCycle[];
}

export default function CycleHistory({ cycles }: CycleHistoryProps) {
  const getFunctionClassIcon = (fc: string) => {
    return <Brain size={16} />;
  };

  const getActionBadge = (action: string) => {
    const styles = {
      DISPATCH: "bg-blue-100 text-blue-800",
      WAIT: "bg-gray-100 text-gray-800",
      REOPTIMIZE: "bg-purple-100 text-purple-800",
    };
    return (
      <span
        className={`px-2 py-0.5 text-xs font-medium rounded ${
          styles[action as keyof typeof styles] || styles.WAIT
        }`}
      >
        {action}
      </span>
    );
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h3 className="font-semibold text-gray-900">Recent Decisions</h3>
        <p className="text-xs text-gray-500 mt-1">
          Last 10 consolidation cycles
        </p>
      </div>

      <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
        {cycles.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Clock size={32} className="mx-auto mb-2 text-gray-400" />
            <p className="text-sm">No cycles yet</p>
          </div>
        ) : (
          cycles.slice(0, 10).map((cycle) => (
            <div
              key={cycle.cycle_number}
              className="p-3 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="text-gray-400">
                    {getFunctionClassIcon(cycle.function_class)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-900">
                        Cycle #{cycle.cycle_number}
                      </span>
                      {getActionBadge(cycle.action_type)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatDistanceToNow(new Date(cycle.timestamp), {
                        addSuffix: true,
                      })}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div
                    className={`text-sm font-medium ${getConfidenceColor(
                      cycle.confidence
                    )}`}
                  >
                    {(cycle.confidence * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500">confidence</div>
                </div>
              </div>

              <div className="text-xs text-gray-700 mb-2 line-clamp-2">
                {cycle.reasoning}
              </div>

              <div className="flex items-center gap-4 text-xs text-gray-600">
                <span className="flex items-center gap-1">
                  <TrendingUp size={12} />
                  {cycle.function_class}
                </span>
                {cycle.shipments_dispatched > 0 && (
                  <span className="flex items-center gap-1">
                    📦 {cycle.shipments_dispatched} dispatched
                  </span>
                )}
                {cycle.vehicles_utilized > 0 && (
                  <span className="flex items-center gap-1">
                    🚚 {cycle.vehicles_utilized} vehicles
                  </span>
                )}
                <span className="ml-auto text-gray-500">
                  {cycle.execution_time_ms}ms
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
