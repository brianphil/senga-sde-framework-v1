// src/types/index.ts

export interface Shipment {
  id: string;
  order_id: string;
  customer_id: string;
  pickup_location: string;
  delivery_location: string;
  pickup_coords: [number, number];
  delivery_coords: [number, number];
  weight: number;
  volume: number;
  declared_value: number;
  deadline: string;
  priority: 'emergency' | 'urgent' | 'standard';
  status: 'pending' | 'assigned' | 'in_transit' | 'delivered' | 'cancelled';
  assigned_route_id?: string;
  created_at: string;
  time_to_deadline_hours?: number;
}

export interface Vehicle {
  vehicle_id: string;
  vehicle_type: string;
  capacity_volume_m3: number;
  capacity_weight_kg: number;
  cost_per_km: number;
  current_location: string;
  current_coords: [number, number];
  status: 'idle' | 'dispatched' | 'in_transit' | 'maintenance';
  assigned_route_id?: string;
  current_utilization?: number;
}

export interface Route {
  route_id: string;
  vehicle_id: string;
  shipment_ids: string[];
  stops: RouteStop[];
  total_distance_km: number;
  estimated_duration_hours: number;
  total_cost: number;
  created_at: string;
  status: 'planned' | 'active' | 'completed' | 'cancelled';
}

export interface RouteStop {
  sequence: number;
  location: string;
  coords: [number, number];
  stop_type: 'pickup' | 'delivery';
  shipment_id: string;
  eta?: string;
  completed_at?: string;
}

export interface ConsolidationCycle {
  cycle_number: number;
  timestamp: string;
  function_class: 'PFA' | 'CFA' | 'VFA' | 'DLA';
  action_type: 'DISPATCH' | 'WAIT' | 'REOPTIMIZE';
  reasoning: string;
  confidence: number;
  shipments_dispatched: number;
  vehicles_utilized: number;
  execution_time_ms: number;
  state_complexity?: string;
  stakes?: string;
  alternatives_considered?: Array<{
    function_class: string;
    score: number;
    reason: string;
  }>;
}

export interface SystemStatus {
  engine_status: 'idle' | 'running' | 'paused' | 'error';
  integrations_online: boolean;
  pending_shipments: number;
  available_vehicles: number;
  current_cycle: number;
  last_decision_time?: string;
  fleet_utilization: number;
  avg_sla_compliance: number;
}

export interface PerformanceMetrics {
  total_cycles: number;
  total_shipments_processed: number;
  total_dispatches: number;
  avg_utilization: number;
  avg_sla_compliance: number;
  avg_cycle_time_ms: number;
  function_class_usage: Record<string, number>;
  hourly_throughput: Array<{ hour: string; shipments: number }>;
  utilization_trend: Array<{ timestamp: string; utilization: number }>;
}

export interface DecisionLog {
  id: string;
  timestamp: string;
  cycle_number: number;
  function_class: string;
  action_type: string;
  shipments_affected: number;
  reasoning: string;
  confidence: number;
  execution_time_ms: number;
}

export interface NewOrderRequest {
  customer_id: string;
  pickup_location: string;
  delivery_location: string;
  weight_kg: number;
  volume_m3: number;
  declared_value: number;
  delivery_deadline: string;
  priority: 'emergency' | 'urgent' | 'standard';
}