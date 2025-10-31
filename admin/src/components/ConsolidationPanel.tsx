// src/components/ConsolidationPanel.tsx
import { useState } from "react";
import { Play, Loader, Zap } from "lucide-react";
import { api } from "../services/api.ts";
import { ConsolidationCycleResponse } from "../types/index.ts";

interface Props {
  pendingOrdersCount: number;
  onConsolidationComplete: (result: ConsolidationCycleResponse) => void;
}

export default function ConsolidationPanel({
  pendingOrdersCount,
  onConsolidationComplete,
}: Props) {
  const [running, setRunning] = useState(false);
  const [lastResult, setLastResult] =
    useState<ConsolidationCycleResponse | null>(null);

  const handleTrigger = async (forceDispatch: boolean = false) => {
    if (pendingOrdersCount === 0 && !forceDispatch) {
      alert("No pending orders to consolidate");
      return;
    }

    setRunning(true);
    try {
      const result = await api.triggerConsolidationCycle(forceDispatch);
      setLastResult(result);
      onConsolidationComplete(result);
    } catch (error) {
      alert(`Consolidation failed: ${error}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="panel">
      <h2 className="panel-title">Consolidation Cycle</h2>

      <div className="metric-display">
        <div className="metric-large">
          <span className="metric-value">{pendingOrdersCount}</span>
          <span className="metric-label">Pending Orders</span>
        </div>
      </div>

      <div className="button-group-vertical">
        <button
          onClick={() => handleTrigger(false)}
          disabled={running || pendingOrdersCount === 0}
          className="btn-primary btn-large"
        >
          {running ? (
            <>
              <Loader size={18} className="spinning" /> Running Cycle...
            </>
          ) : (
            <>
              <Play size={18} /> Run Consolidation
            </>
          )}
        </button>

        <button
          onClick={() => handleTrigger(true)}
          disabled={running || pendingOrdersCount === 0}
          className="btn-warning"
        >
          <Zap size={16} /> Force Dispatch
        </button>
      </div>

      {lastResult && (
        <div className="result-summary">
          <h3>Last Cycle Result</h3>
          <div className="result-metrics">
            <div className="result-metric">
              <span className="label">Function Class:</span>
              <span className="value function-class">
                {lastResult.function_class_used}
              </span>
            </div>
            <div className="result-metric">
              <span className="label">Orders Dispatched:</span>
              <span className="value">{lastResult.orders_dispatched}</span>
            </div>
            <div className="result-metric">
              <span className="label">Batches Created:</span>
              <span className="value">{lastResult.batches_created}</span>
            </div>
            <div className="result-metric">
              <span className="label">Orders Waiting:</span>
              <span className="value">{lastResult.orders_waiting}</span>
            </div>
          </div>
          <div className="result-reasoning">
            <strong>Reasoning:</strong>
            <p>{lastResult.reasoning}</p>
          </div>
        </div>
      )}
    </div>
  );
}
