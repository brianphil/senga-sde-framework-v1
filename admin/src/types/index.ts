// src/types/index.ts
// Type definitions matching Senga SDE API exactly

export interface Location {
  address: string;
  latitude: number;
  longitude: number;
}

export interface OrderCreateRequest {
  customer_name: string;
  customer_phone?: string;
  pickup_location: Location;
  delivery_location: Location;
  package_weight: number;
  volume_m3?: number;
  priority?: 'standard' | 'urgent' | 'emergency';
  created_at?: string;
  customer_id?: string;
  order_id?: string;
}

export interface OrderResponse {
  order_id: string;
  customer_id: string;
  customer_name: string;
  pickup_location: Location;
  delivery_location: Location;
  package_weight: number;
  volume_m3: number;
  priority: string;
  status: string;
  created_at: string;
  deadline: string;
  time_to_deadline_hours: number;
}

export interface RouteStop {
  sequence: number;
  location: string;
  coords: [number, number];
  stop_type: 'pickup' | 'delivery';
  shipment_id?: string;
  eta?: string;
}

export interface Route {
  id: string;
  vehicle_id: string;
  shipment_ids: string[];
  stops: RouteStop[];
  estimated_distance_km: number;
  estimated_duration_hours: number;
  estimated_cost: number;
  status: 'planned' | 'active' | 'completed';
  created_at: string;
  utilization_weight?: number;
  utilization_volume?: number;
}

export interface DispatchedBatch {
  vehicle_id: string;
  shipments: string[];
  route: Array<{lat: number; lng: number; formatted_address: string}>;
  estimated_distance_km: number;
  estimated_duration_hours: number;
}

export interface ConsolidationCycleResponse {
  timestamp: string;
  total_pending_orders: number;
  orders_dispatched: number;
  orders_waiting: number;
  batches_created: number;
  function_class_used: string;
  reasoning: string;
  dispatched_batches: DispatchedBatch[];
  waiting_orders: OrderResponse[];
}

export interface SystemStatusResponse {
  status: string;
  pending_orders: number;
  available_vehicles: number;
  active_routes: number;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  components?: Record<string, {status: string; message?: string}>;
}

export interface CycleHistory {
  cycle_number: number;
  timestamp: string;
  function_class: string;
  action_type: string;
  shipments_dispatched: number;
  batches_created: number;
  reasoning: string;
  confidence: number;
  execution_time_ms: number;
}

export interface Vehicle {
  id: string;
  type: string;
  capacity_weight_kg: number;
  capacity_volume_m3: number;
  status: 'available' | 'dispatched' | 'in_transit' | 'maintenance';
  current_location?: string;
}

// UI-specific types
export interface EventLogEntry {
  id: string;
  timestamp: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  details?: string;
}

export interface PerformanceSnapshot {
  timestamp: number;
  utilizationRate: number;
  dispatchedOrders: number;
  activeRoutes: number;
}