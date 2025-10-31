// src/services/api.ts
import axios, { AxiosInstance } from 'axios';
import {
  OrderCreateRequest,
  OrderResponse,
  ConsolidationCycleResponse,
  SystemStatusResponse,
  HealthResponse,
  CycleHistory,
  Route,
  Vehicle
} from '../types';

class SengaAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      response => response,
      error => {
        console.error('API Error:', error.response?.data || error.message);
        throw error;
      }
    );
  }

  // ============= Health & Status =============
  async checkHealth(): Promise<HealthResponse> {
    const { data } = await this.client.get<HealthResponse>('/health');
    return data;
  }

  async getSystemStatus(): Promise<SystemStatusResponse> {
    const { data } = await this.client.get<SystemStatusResponse>('/status');
    return data;
  }

  // ============= Order Management =============
  async createOrder(order: OrderCreateRequest): Promise<OrderResponse> {
    const { data } = await this.client.post<OrderResponse>('/orders', order);
    return data;
  }

  async getPendingOrders(): Promise<OrderResponse[]> {
    const { data } = await this.client.get<OrderResponse[]>('/orders/pending');
    return data;
  }

  async getOrderById(orderId: string): Promise<OrderResponse> {
    const { data } = await this.client.get<OrderResponse>(`/orders/${orderId}`);
    return data;
  }

  // ============= Consolidation & Decisions =============
  async triggerConsolidationCycle(forceDispatch: boolean = false): Promise<ConsolidationCycleResponse> {
    const { data } = await this.client.post<ConsolidationCycleResponse>(
      '/decisions/consolidation-cycle',
      { force_dispatch: forceDispatch, context: null }
    );
    return data;
  }

  async getRecentCycles(limit: number = 20): Promise<CycleHistory[]> {
    const { data } = await this.client.get<CycleHistory[]>(`/cycles/recent?n=${limit}`);
    return data;
  }

  async getCycleById(cycleNumber: number): Promise<CycleHistory> {
    const { data } = await this.client.get<CycleHistory>(`/cycles/${cycleNumber}`);
    return data;
  }

  // ============= Routes =============
  async getActiveRoutes(): Promise<Route[]> {
    const { data } = await this.client.get<Route[]>('/routes/active');
    return data;
  }

  async getCompletedRoutes(limit: number = 50): Promise<Route[]> {
    const { data } = await this.client.get<Route[]>(`/routes/completed?limit=${limit}`);
    return data;
  }

  async completeRoute(routeId: string, outcome: any): Promise<any> {
    const { data } = await this.client.post(`/routes/${routeId}/complete`, outcome);
    return data;
  }

  // ============= Fleet Management =============
  async getFleet(): Promise<Vehicle[]> {
    const { data } = await this.client.get<Vehicle[]>('/fleet');
    return data;
  }

  async getAvailableVehicles(): Promise<Vehicle[]> {
    const { data } = await this.client.get<Vehicle[]>('/fleet/available');
    return data;
  }

  // ============= Demo Data =============
  async initializeDemoData(): Promise<{success: boolean; message: string}> {
    const { data } = await this.client.post('/demo/initialize');
    return data;
  }
}

// Export singleton instance
export const api = new SengaAPI();
export default api;