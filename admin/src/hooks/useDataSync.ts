// src/hooks/useDataSync.ts

import { useEffect, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useStore } from '../store/useStore';
import apiService from '../services/api';
import { toast } from 'sonner';

export function useSystemStatus() {
  const setSystemStatus = useStore((state) => state.setSystemStatus);
  const autoRefresh = useStore((state) => state.autoRefresh);
  const refreshInterval = useStore((state) => state.refreshInterval);

  return useQuery({
    queryKey: ['system-status'],
    queryFn: async () => {
      const status = await apiService.getStatus();
      setSystemStatus(status);
      return status;
    },
    refetchInterval: autoRefresh ? refreshInterval : false,
    onError: (error) => {
      console.error('Failed to fetch system status:', error);
    },
  });
}

export function usePendingShipments() {
  const setPendingShipments = useStore((state) => state.setPendingShipments);
  const autoRefresh = useStore((state) => state.autoRefresh);
  const refreshInterval = useStore((state) => state.refreshInterval);

  return useQuery({
    queryKey: ['pending-shipments'],
    queryFn: async () => {
      const shipments = await apiService.getPendingShipments();
      setPendingShipments(shipments);
      return shipments;
    },
    refetchInterval: autoRefresh ? refreshInterval : false,
  });
}

export function useFleet() {
  const setFleet = useStore((state) => state.setFleet);
  const autoRefresh = useStore((state) => state.autoRefresh);
  const refreshInterval = useStore((state) => state.refreshInterval);

  return useQuery({
    queryKey: ['fleet'],
    queryFn: async () => {
      const fleet = await apiService.getFleet();
      setFleet(fleet);
      return fleet;
    },
    refetchInterval: autoRefresh ? refreshInterval : false,
  });
}

export function useActiveRoutes() {
  const setActiveRoutes = useStore((state) => state.setActiveRoutes);
  const autoRefresh = useStore((state) => state.autoRefresh);
  const refreshInterval = useStore((state) => state.refreshInterval);

  return useQuery({
    queryKey: ['active-routes'],
    queryFn: async () => {
      const routes = await apiService.getActiveRoutes();
      setActiveRoutes(routes);
      return routes;
    },
    refetchInterval: autoRefresh ? refreshInterval : false,
  });
}

export function useRecentCycles() {
  const setRecentCycles = useStore((state) => state.setRecentCycles);
  const autoRefresh = useStore((state) => state.autoRefresh);
  const refreshInterval = useStore((state) => state.refreshInterval);

  return useQuery({
    queryKey: ['recent-cycles'],
    queryFn: async () => {
      const cycles = await apiService.getRecentCycles(50);
      setRecentCycles(cycles);
      return cycles;
    },
    refetchInterval: autoRefresh ? refreshInterval : false,
  });
}

export function useTriggerCycle() {
  const queryClient = useQueryClient();
  const addCycle = useStore((state) => state.addCycle);

  const trigger = useCallback(async () => {
    try {
      toast.loading('Running consolidation cycle...', { id: 'cycle' });
      
      const cycle = await apiService.triggerConsolidationCycle();
      
      addCycle(cycle);
      
      // Invalidate all data queries to refresh
      await queryClient.invalidateQueries(['pending-shipments']);
      await queryClient.invalidateQueries(['fleet']);
      await queryClient.invalidateQueries(['active-routes']);
      await queryClient.invalidateQueries(['system-status']);
      await queryClient.invalidateQueries(['recent-cycles']);
      
      toast.success(
        `Cycle ${cycle.cycle_number}: ${cycle.action_type} - ${cycle.function_class}`,
        { id: 'cycle', duration: 3000 }
      );
      
      return cycle;
    } catch (error) {
      toast.error('Failed to run consolidation cycle', { id: 'cycle' });
      throw error;
    }
  }, [queryClient, addCycle]);

  return { trigger };
}

export function useAutoRefresh() {
  const autoRefresh = useStore((state) => state.autoRefresh);
  const refreshInterval = useStore((state) => state.refreshInterval);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Silent refresh - queries will auto-update
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);
}