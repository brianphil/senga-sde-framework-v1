// src/store/useStore.ts

import { create } from 'zustand';
import type { SystemStatus, Shipment, Vehicle, Route, ConsolidationCycle } from '../types';

interface AppState {
  // System state
  systemStatus: SystemStatus | null;
  isLoading: boolean;
  error: string | null;

  // Data
  pendingShipments: Shipment[];
  fleet: Vehicle[];
  activeRoutes: Route[];
  recentCycles: ConsolidationCycle[];

  // UI state
  selectedShipment: Shipment | null;
  selectedRoute: Route | null;
  autoRefresh: boolean;
  refreshInterval: number; // milliseconds

  // Actions
  setSystemStatus: (status: SystemStatus) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  setPendingShipments: (shipments: Shipment[]) => void;
  setFleet: (fleet: Vehicle[]) => void;
  setActiveRoutes: (routes: Route[]) => void;
  setRecentCycles: (cycles: ConsolidationCycle[]) => void;
  addCycle: (cycle: ConsolidationCycle) => void;

  setSelectedShipment: (shipment: Shipment | null) => void;
  setSelectedRoute: (route: Route | null) => void;
  setAutoRefresh: (enabled: boolean) => void;
  setRefreshInterval: (interval: number) => void;

  // Computed getters
  getAvailableVehicles: () => Vehicle[];
  getUrgentShipments: () => Shipment[];
  getFleetUtilization: () => number;
}

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  systemStatus: null,
  isLoading: false,
  error: null,
  pendingShipments: [],
  fleet: [],
  activeRoutes: [],
  recentCycles: [],
  selectedShipment: null,
  selectedRoute: null,
  autoRefresh: true,
  refreshInterval: 5000,

  // Actions
  setSystemStatus: (status) => set({ systemStatus: status }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  
  setPendingShipments: (shipments) => set({ pendingShipments: shipments }),
  setFleet: (fleet) => set({ fleet }),
  setActiveRoutes: (routes) => set({ activeRoutes: routes }),
  setRecentCycles: (cycles) => set({ recentCycles: cycles }),
  
  addCycle: (cycle) => set((state) => ({
    recentCycles: [cycle, ...state.recentCycles].slice(0, 50)
  })),

  setSelectedShipment: (shipment) => set({ selectedShipment: shipment }),
  setSelectedRoute: (route) => set({ selectedRoute: route }),
  setAutoRefresh: (enabled) => set({ autoRefresh: enabled }),
  setRefreshInterval: (interval) => set({ refreshInterval: interval }),

  // Computed getters
  getAvailableVehicles: () => {
    return get().fleet.filter((v) => v.status === 'idle');
  },

  getUrgentShipments: () => {
    return get().pendingShipments.filter((s) => {
      const hours = s.time_to_deadline_hours || 0;
      return hours <= 24 && s.priority !== 'standard';
    });
  },

  getFleetUtilization: () => {
    const fleet = get().fleet;
    if (fleet.length === 0) return 0;
    const busy = fleet.filter((v) => v.status !== 'idle').length;
    return (busy / fleet.length) * 100;
  },
}));