// src/components/EventLog.tsx
import { Trash2, AlertCircle, CheckCircle, Info, XCircle } from "lucide-react";
import { EventLogEntry } from "../types";

interface Props {
  events: EventLogEntry[];
  onClear: () => void;
}

export default function EventLog({ events, onClear }: Props) {
  const getIcon = (type: EventLogEntry["type"]) => {
    switch (type) {
      case "success":
        return <CheckCircle size={16} className="icon-success" />;
      case "error":
        return <XCircle size={16} className="icon-error" />;
      case "warning":
        return <AlertCircle size={16} className="icon-warning" />;
      default:
        return <Info size={16} className="icon-info" />;
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <div className="panel event-log">
      <div className="panel-header">
        <h2 className="panel-title">Event Log</h2>
        <button onClick={onClear} className="btn-icon-small" title="Clear Log">
          <Trash2 size={14} />
        </button>
      </div>

      <div className="event-list">
        {events.length === 0 ? (
          <div className="empty-log">
            <Info size={32} className="empty-icon" />
            <p>No events yet</p>
          </div>
        ) : (
          events.map((event) => (
            <div key={event.id} className={`event-item event-${event.type}`}>
              <div className="event-header">
                {getIcon(event.type)}
                <span className="event-time">
                  {formatTime(event.timestamp)}
                </span>
              </div>
              <div className="event-message">{event.message}</div>
              {event.details && (
                <div className="event-details">{event.details}</div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
