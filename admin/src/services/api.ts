// src/services/api.ts

import axios, { AxiosError } from 'axios';
import type {
  SystemStatus,
  Shipment,
  Vehicle,
  Route,
  ConsolidationCycle,
  PerformanceMetrics,
  DecisionLog,
  NewOrderRequest
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const message = error.response?.data 
      ? (error.response.data as any).detail || 'API Error'
      : error.message;
    console.error('API Error:', message);
    throw new Error(message);
  }
);

export const apiService = {
  // System Status
  async getHealth(): Promise<{ status: string }> {
    const { data } = await api.get('/health');
    return data;
  },

  async getStatus(): Promise<SystemStatus> {
    const { data } = await api.get('/status');
    return data;
  },

  // Orders & Shipments
  async getPendingShipments(): Promise<Shipment[]> {
    const { data } = await api.get('/shipments/pending');
    return data;
  },

  async getShipmentById(id: string): Promise<Shipment> {
    const { data } = await api.get(`/shipments/${id}`);
    return data;
  },

  async createOrder(order: NewOrderRequest): Promise<{ shipment_id: string }> {
    const { data } = await api.post('/orders/ingest', order);
    return data;
  },

  // Fleet Management
  async getFleet(): Promise<Vehicle[]> {
    const { data } = await api.get('/fleet');
    return data;
  },

  async getAvailableVehicles(): Promise<Vehicle[]> {
    const { data } = await api.get('/fleet/available');
    return data;
  },

  // Routes
  async getActiveRoutes(): Promise<Route[]> {
    const { data } = await api.get('/routes/active');
    return data;
  },

  async getRouteById(id: string): Promise<Route> {
    const { data } = await api.get(`/routes/${id}`);
    return data;
  },

  async getCompletedRoutes(limit: number = 50): Promise<Route[]> {
    const { data } = await api.get(`/routes/completed?limit=${limit}`);
    return data;
  },

  // Decision Engine
  async triggerConsolidationCycle(): Promise<ConsolidationCycle> {
    const { data } = await api.post('/decisions/consolidation-cycle');
    return data;
  },

  async getRecentCycles(limit: number = 20): Promise<ConsolidationCycle[]> {
    const { data } = await api.get(`/cycles/recent?n=${limit}`);
    return data;
  },

  async getCycleById(cycleNumber: number): Promise<ConsolidationCycle> {
    const { data } = await api.get(`/cycles/${cycleNumber}`);
    return data;
  },

  // Analytics & Performance
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const { data } = await api.get('/analytics/performance');
    return data;
  },

  async getDecisionHistory(limit: number = 100): Promise<DecisionLog[]> {
    const { data } = await api.get(`/analytics/decisions?limit=${limit}`);
    return data;
  },

  // Configuration (if exposed)
  async getBusinessConfig(): Promise<Record<string, any>> {
    const { data } = await api.get('/config/business');
    return data;
  },

  async updateBusinessConfig(config: Record<string, any>): Promise<void> {
    await api.put('/config/business', config);
  },
};

export default apiService;