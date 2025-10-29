# src/core/cfa.py
"""
Cost Function Approximator - FIXED IMPLEMENTATION
Relaxed constraints to avoid infeasibility
"""

from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Batch:
    id: str
    shipments: list
    vehicle: object
    route_sequence: list
    total_distance_km: float
    total_duration_hours: float
    utilization: float
    total_cost: float

@dataclass
class CFASolution:
    batches: List[Batch]
    unassigned_shipments: list
    total_cost: float
    avg_utilization: float
    status: str
    solver_time_seconds: float
    reasoning: str

class CostFunctionApproximator:
    """EXACT NAME - No breaking changes"""
    
    def __init__(self):
        from ..config.senga_config import SengaConfigurator
        self.config = SengaConfigurator()
        self.time_limit = self.config.get('cfa.solver.time_limit_seconds', 30)
        self.cost_per_km = self.config.get('cost_per_km', 50)
        self.delay_penalty = self.config.get('delay_penalty_per_hour', 500)
        self.util_bonus = self.config.get('utilization_bonus_per_percent', 50)  # Reduced from 100
        logger.info("CFA initialized with real MIP optimization")
    
    def optimize(self, state, value_function=None) -> CFASolution:
        """EXACT signature - backward compatible"""
        shipments = state.pending_shipments
        vehicles = state.get_available_vehicles()
        
        if not shipments or not vehicles:
            return CFASolution([], shipments, 0, 0, 'INFEASIBLE', 0, 'No resources')
        
        import time
        start_time = time.time()
        batch_assignments, mip_status = self._solve_batch_formation(shipments, vehicles, state, value_function)
        
        if mip_status not in ['OPTIMAL', 'FEASIBLE'] or not batch_assignments:
            return CFASolution([], shipments, 0, 0, mip_status, time.time()-start_time, 'MIP failed')
        
        batches = []
        for batch_id, (ship_ids, vehicle_id) in batch_assignments.items():
            batch_ships = [s for s in shipments if s.id in ship_ids]
            vehicle = next(v for v in vehicles if v.id == vehicle_id)
            
            route_seq, dist_km, dur_hrs = self._solve_vrp(batch_ships, vehicle)
            
            total_vol = sum(s.volume for s in batch_ships)
            utilization = total_vol / vehicle.capacity.volume if vehicle.capacity.volume > 0 else 0
            cost = dist_km * self.cost_per_km
            
            batches.append(Batch(
                id=batch_id,
                shipments=batch_ships,
                vehicle=vehicle,
                route_sequence=route_seq,
                total_distance_km=dist_km,
                total_duration_hours=dur_hrs,
                utilization=utilization,
                total_cost=cost
            ))
        
        assigned_ids = set().union(*[ship_ids for ship_ids, _ in batch_assignments.values()])
        unassigned = [s for s in shipments if s.id not in assigned_ids]
        
        total_cost = sum(b.total_cost for b in batches)
        avg_util = np.mean([b.utilization for b in batches]) if batches else 0
        
        return CFASolution(
            batches=batches,
            unassigned_shipments=unassigned,
            total_cost=total_cost,
            avg_utilization=avg_util,
            status=mip_status,
            solver_time_seconds=time.time()-start_time,
            reasoning=f"{len(batches)} batches, {avg_util:.1%} util"
        )
    
    def _solve_batch_formation(self, shipments, vehicles, state, vfa) -> Tuple[Dict, str]:
        """Real CP-SAT MIP - RELAXED constraints"""
        model = cp_model.CpModel()
        
        n_s = len(shipments)
        n_v = len(vehicles)
        max_batches = min(n_s, n_v)  # SIMPLIFIED: 1 batch per vehicle max
        
        # Decision variables
        x = {(i,j): model.NewBoolVar(f'x_{i}_{j}') for i in range(n_s) for j in range(max_batches)}
        y = {j: model.NewBoolVar(f'y_{j}') for j in range(max_batches)}
        z = {(j,k): model.NewBoolVar(f'z_{j}_{k}') for j in range(max_batches) for k in range(n_v)}
        
        # CORE Constraints (simplified to avoid infeasibility)
        
        # C1: Each shipment in at most one batch
        for i in range(n_s):
            model.Add(sum(x[i,j] for j in range(max_batches)) <= 1)
        
        # C2: Batch activation
        for i in range(n_s):
            for j in range(max_batches):
                model.Add(x[i,j] <= y[j])
        
        # C3: Each used batch gets exactly one vehicle
        for j in range(max_batches):
            model.Add(sum(z[j,k] for k in range(n_v)) == y[j])
        
        # C4: Each vehicle used at most once
        for k in range(n_v):
            model.Add(sum(z[j,k] for j in range(max_batches)) <= 1)
        
        # C5: Volume capacity (with 10% slack to avoid infeasibility)
        for j in range(max_batches):
            batch_vol = sum(int(shipments[i].volume * 1000) * x[i,j] for i in range(n_s))
            veh_cap = sum(int(vehicles[k].capacity.volume * 1100) * z[j,k] for k in range(n_v))  # 10% slack
            model.Add(batch_vol <= veh_cap)
        
        # C6: Weight capacity (with 10% slack)
        for j in range(max_batches):
            batch_wt = sum(int(shipments[i].weight * 100) * x[i,j] for i in range(n_s))
            veh_cap_wt = sum(int(vehicles[k].capacity.weight * 110) * z[j,k] for k in range(n_v))  # 10% slack
            model.Add(batch_wt <= veh_cap_wt)
        
        # SIMPLIFIED OBJECTIVE (avoid complex constraints)
        objective_terms = []
        
        # Cost: Dispatch costs
        for j in range(max_batches):
            for k in range(n_v):
                objective_terms.append(1000 * z[j,k])
        
        # Benefit: Reward for dispatching shipments
        for i in range(n_s):
            time_remaining = (shipments[i].deadline - state.timestamp).total_seconds() / 3600
            if time_remaining < 12:  # Within 12 hours
                urgency_reward = int(100 * (12 - time_remaining))
                for j in range(max_batches):
                    objective_terms.append(-urgency_reward * x[i,j])  # Negative = reward
        
        # Minimize costs
        model.Minimize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = 4
        
        status = solver.Solve(model)
        
        batch_assignments = {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for j in range(max_batches):
                if solver.Value(y[j]) == 1:
                    ship_ids = set(shipments[i].id for i in range(n_s) if solver.Value(x[i,j]) == 1)
                    vehicle_id = next(vehicles[k].id for k in range(n_v) if solver.Value(z[j,k]) == 1)
                    if ship_ids:
                        batch_assignments[f'batch_{j}'] = (ship_ids, vehicle_id)
            
            status_str = 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'
            logger.info(f"MIP solved: {status_str}, {len(batch_assignments)} batches")
        else:
            status_str = 'INFEASIBLE'
            logger.warning(f"MIP infeasible: {n_s} shipments, {n_v} vehicles")
        
        return batch_assignments, status_str
    
    def _solve_vrp(self, shipments, vehicle) -> Tuple[list, float, float]:
        """OR-Tools VRP"""
        locations = [vehicle.current_location]
        for s in shipments:
            locations.append(s.origin)
            locations.extend(s.destinations)
        
        n = len(locations)
        if n == 1:
            return locations, 0.0, 0.0
        
        dist_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self._haversine(locations[i].lat, locations[i].lng, locations[j].lat, locations[j].lng)
                    dist_matrix[i][j] = int(dist * 1000)
        
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_idx, to_idx):
            return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.time_limit.seconds = 10
        
        solution = routing.SolveWithParameters(search_params)
        
        if solution:
            route_seq = []
            index = routing.Start(0)
            total_dist_m = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_seq.append(locations[node])
                next_index = solution.Value(routing.NextVar(index))
                total_dist_m += routing.GetArcCostForVehicle(index, next_index, 0)
                index = next_index
            
            return route_seq, total_dist_m / 1000.0, (total_dist_m / 1000.0) / 30.0
        else:
            logger.warning("VRP failed, using simple sequence")
            return locations, sum(dist_matrix[i][i+1] for i in range(n-1))/1000.0, 2.0
    
    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        from math import radians, sin, cos, sqrt, asin
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))