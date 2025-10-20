# src/core/cfa.py

"""
Cost Function Approximation (CFA): Consolidation optimization via MIP
Uses Google OR-Tools to solve batch formation and route sequencing
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from ortools.sat.python import cp_model

from .state_manager import (
    StateManager, SystemState, Shipment, VehicleState, Route, Location
)
from ..config.senga_config import SengaConfigurator

@dataclass
class Batch:
    """A batch of shipments assigned to a vehicle"""
    id: str
    shipments: List[Shipment]
    vehicle: VehicleState
    sequence: List[Location]
    estimated_distance: float
    estimated_duration: timedelta
    utilization: float
    total_cost: float

@dataclass
class CFASolution:
    """Solution from CFA optimization"""
    batches: List[Batch]
    total_cost: float
    avg_utilization: float
    solver_time: float
    status: str  # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE'
    reasoning: str

class CostFunctionApproximator:
    """
    CFA: Consolidation optimization using Mixed Integer Programming
    
    Objective: Minimize cost while achieving utilization targets
    Constraints: Vehicle capacity, time windows, fleet availability
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
    
    def optimize(self, state: SystemState, 
                 value_function=None) -> CFASolution:
        """
        Main optimization: solve batch formation and routing
        
        Args:
            state: Current system state
            value_function: Optional VFA for guidance
            
        Returns:
            CFASolution with optimal batches and routes
        """
        
        pending = state.pending_shipments
        vehicles = state.get_available_vehicles()
        
        if not pending:
            return CFASolution(
                batches=[],
                total_cost=0.0,
                avg_utilization=0.0,
                solver_time=0.0,
                status='NO_PENDING',
                reasoning="No pending shipments to optimize"
            )
        
        if not vehicles:
            return CFASolution(
                batches=[],
                total_cost=0.0,
                avg_utilization=0.0,
                solver_time=0.0,
                status='NO_VEHICLES',
                reasoning="No available vehicles for dispatch"
            )
        
        # Build and solve MIP
        start_time = datetime.now()
        
        try:
            solution = self._solve_mip(state, pending, vehicles, value_function)
            solver_time = (datetime.now() - start_time).total_seconds()
            solution.solver_time = solver_time
            return solution
            
        except Exception as e:
            return CFASolution(
                batches=[],
                total_cost=0.0,
                avg_utilization=0.0,
                solver_time=(datetime.now() - start_time).total_seconds(),
                status='ERROR',
                reasoning=f"Optimization failed: {str(e)}"
            )
    
    def _solve_mip(self, state: SystemState, 
                   shipments: List[Shipment],
                   vehicles: List[VehicleState],
                   value_function) -> CFASolution:
        """
        Solve the batch formation and routing MIP
        
        Decision Variables:
        - x[i,j]: shipment i in batch j
        - y[j]: batch j is dispatched
        - z[j,k]: batch j assigned to vehicle k
        
        Objective:
        - Minimize: fixed costs + variable costs + delay penalties
        - Bonus: high utilization
        
        Constraints:
        - Each shipment in exactly one batch (if dispatched)
        - Capacity constraints
        - Time window constraints
        - Vehicle availability
        """
        
        model = cp_model.CpModel()
        
        n_shipments = len(shipments)
        n_vehicles = len(vehicles)
        max_batches = min(n_shipments, n_vehicles * 2)  # Upper bound on batches
        
        # Decision variables
        # x[i, j] = 1 if shipment i assigned to batch j
        x = {}
        for i in range(n_shipments):
            for j in range(max_batches):
                x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
        
        # y[j] = 1 if batch j is used
        y = {}
        for j in range(max_batches):
            y[j] = model.NewBoolVar(f'y_{j}')
        
        # z[j, k] = 1 if batch j assigned to vehicle k
        z = {}
        for j in range(max_batches):
            for k in range(n_vehicles):
                z[j, k] = model.NewBoolVar(f'z_{j}_{k}')
        
        # Constraint 1: Each shipment in at most one batch
        for i in range(n_shipments):
            model.Add(sum(x[i, j] for j in range(max_batches)) <= 1)
        
        # Constraint 2: Batch activation
        for j in range(max_batches):
            for i in range(n_shipments):
                model.Add(x[i, j] <= y[j])
        
        # Constraint 3: Each batch assigned to at most one vehicle
        for j in range(max_batches):
            model.Add(sum(z[j, k] for k in range(n_vehicles)) <= 1)
            # If batch is used, must be assigned to a vehicle
            model.Add(sum(z[j, k] for k in range(n_vehicles)) >= y[j])
        
        # Constraint 4: Each vehicle used at most once
        for k in range(n_vehicles):
            model.Add(sum(z[j, k] for j in range(max_batches)) <= 1)
        
        # Constraint 5: Volume capacity
        for j in range(max_batches):
            for k in range(n_vehicles):
                total_volume = sum(
                    int(shipments[i].volume * 1000) * x[i, j]  # Scale to integer
                    for i in range(n_shipments)
                )
                vehicle_capacity = int(vehicles[k].capacity.volume * 1000)
                
                # If batch j assigned to vehicle k, volume must fit
                model.Add(total_volume <= vehicle_capacity).OnlyEnforceIf(z[j, k])
        
        # Constraint 6: Weight capacity
        for j in range(max_batches):
            for k in range(n_vehicles):
                total_weight = sum(
                    int(shipments[i].weight) * x[i, j]
                    for i in range(n_shipments)
                )
                vehicle_capacity = int(vehicles[k].capacity.weight)
                
                model.Add(total_weight <= vehicle_capacity).OnlyEnforceIf(z[j, k])
        
        # Objective function
        # Cost = fixed costs + variable costs - utilization bonus
        
        # Fixed costs
        fixed_costs = []
        for j in range(max_batches):
            for k in range(n_vehicles):
                fixed_cost = int(vehicles[k].fixed_cost_per_trip)
                fixed_costs.append(fixed_cost * z[j, k])
        
        # Variable costs (estimated)
        variable_costs = []
        for j in range(max_batches):
            for k in range(n_vehicles):
                # Estimate route distance for this batch
                batch_shipments = [shipments[i] for i in range(n_shipments)]
                estimated_distance = self._estimate_batch_distance(
                    vehicles[k], batch_shipments
                )
                var_cost = int(vehicles[k].cost_per_km * estimated_distance)
                variable_costs.append(var_cost * z[j, k])
        
        # Utilization bonus
        utilization_bonus = []
        min_util = self.config.min_utilization
        bonus_per_percent = self.config.get('utilization_bonus_per_percent', 100)
        
        for j in range(max_batches):
            for k in range(n_vehicles):
                total_volume = sum(
                    int(shipments[i].volume * 1000) * x[i, j]
                    for i in range(n_shipments)
                )
                vehicle_capacity = int(vehicles[k].capacity.volume * 1000)
                
                # Bonus if utilization > threshold
                # Simplified: bonus proportional to volume used
                bonus = int((total_volume / 1000) * bonus_per_percent)
                utilization_bonus.append(bonus * z[j, k])
        
        # Total objective
        total_cost = (
            sum(fixed_costs) +
            sum(variable_costs) -
            sum(utilization_bonus)
        )
        
        model.Minimize(total_cost)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.cfa_solver_time_limit
        solver.parameters.num_search_workers = self.config.get('cfa.solver.num_workers', 4)
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._extract_solution(
                solver, x, y, z, shipments, vehicles,
                status == cp_model.OPTIMAL
            )
        else:
            return CFASolution(
                batches=[],
                total_cost=0.0,
                avg_utilization=0.0,
                solver_time=0.0,
                status='INFEASIBLE',
                reasoning="No feasible solution found within time limit"
            )
    
    def _extract_solution(self, solver, x, y, z,
                         shipments: List[Shipment],
                         vehicles: List[VehicleState],
                         is_optimal: bool) -> CFASolution:
        """Extract solution from solved model"""
        
        batches = []
        n_shipments = len(shipments)
        n_vehicles = len(vehicles)
        max_batches = len(y)
        
        for j in range(max_batches):
            if solver.Value(y[j]) == 0:
                continue  # Batch not used
            
            # Find shipments in this batch
            batch_shipments = []
            for i in range(n_shipments):
                if solver.Value(x[i, j]) == 1:
                    batch_shipments.append(shipments[i])
            
            if not batch_shipments:
                continue
            
            # Find assigned vehicle
            assigned_vehicle = None
            for k in range(n_vehicles):
                if solver.Value(z[j, k]) == 1:
                    assigned_vehicle = vehicles[k]
                    break
            
            if not assigned_vehicle:
                continue
            
            # Optimize route sequence for this batch
            sequence, distance, duration = self._optimize_route_sequence(
                assigned_vehicle, batch_shipments
            )
            
            # Calculate utilization
            total_volume = sum(s.volume for s in batch_shipments)
            utilization = total_volume / assigned_vehicle.capacity.volume
            
            # Calculate cost
            total_cost = (
                assigned_vehicle.fixed_cost_per_trip +
                assigned_vehicle.cost_per_km * distance
            )
            
            batch = Batch(
                id=f"batch_{j}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                shipments=batch_shipments,
                vehicle=assigned_vehicle,
                sequence=sequence,
                estimated_distance=distance,
                estimated_duration=duration,
                utilization=utilization,
                total_cost=total_cost
            )
            
            batches.append(batch)
        
        if not batches:
            return CFASolution(
                batches=[],
                total_cost=0.0,
                avg_utilization=0.0,
                solver_time=0.0,
                status='NO_SOLUTION',
                reasoning="Solution found but no valid batches extracted"
            )
        
        avg_utilization = np.mean([b.utilization for b in batches])
        total_cost = sum(b.total_cost for b in batches)
        
        return CFASolution(
            batches=batches,
            total_cost=total_cost,
            avg_utilization=avg_utilization,
            solver_time=0.0,  # Will be set by caller
            status='OPTIMAL' if is_optimal else 'FEASIBLE',
            reasoning=f"Found {len(batches)} batches with avg utilization {avg_utilization:.2%}"
        )
    
    def _optimize_route_sequence(self, vehicle: VehicleState,
                                 shipments: List[Shipment]) -> Tuple[List[Location], float, timedelta]:
        """
        Solve TSP for route sequencing
        
        Uses Christofides algorithm (1.5-approximation)
        For simplicity, using nearest neighbor heuristic here
        
        Returns: (sequence, total_distance_km, total_duration)
        """
        
        # Collect all locations
        locations = [vehicle.current_location]
        for shipment in shipments:
            locations.append(shipment.origin)
            locations.extend(shipment.destinations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for loc in locations:
            if loc.place_id not in seen:
                seen.add(loc.place_id)
                unique_locations.append(loc)
        
        if len(unique_locations) <= 2:
            # Trivial case
            return unique_locations, self._total_distance(unique_locations), timedelta(hours=1)
        
        # Nearest neighbor TSP
        unvisited = set(range(1, len(unique_locations)))  # Skip vehicle location
        current = 0
        sequence = [unique_locations[0]]
        total_distance = 0.0
        
        while unvisited:
            nearest = min(
                unvisited,
                key=lambda i: self._distance(
                    unique_locations[current], unique_locations[i]
                )
            )
            
            distance = self._distance(unique_locations[current], unique_locations[nearest])
            total_distance += distance
            sequence.append(unique_locations[nearest])
            unvisited.remove(nearest)
            current = nearest
        
        # Estimate duration (assume average speed of 40 km/h in traffic)
        avg_speed_kmh = 40.0
        travel_time_hours = total_distance / avg_speed_kmh
        
        # Add stop time (15 minutes per stop)
        stop_time_hours = (len(sequence) - 1) * 0.25
        
        total_duration = timedelta(hours=travel_time_hours + stop_time_hours)
        
        return sequence, total_distance, total_duration
    
    def _distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance between two locations (km)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = radians(loc1.lat), radians(loc1.lng)
        lat2, lon2 = radians(loc2.lat), radians(loc2.lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _total_distance(self, locations: List[Location]) -> float:
        """Calculate total distance for a sequence of locations"""
        total = 0.0
        for i in range(len(locations) - 1):
            total += self._distance(locations[i], locations[i+1])
        return total
    
    def _estimate_batch_distance(self, vehicle: VehicleState,
                                 shipments: List[Shipment]) -> float:
        """
        Quick distance estimate for batch (used in MIP objective)
        Uses centroid approximation for speed
        """
        if not shipments:
            return 0.0
        
        # Calculate centroid of all destinations
        all_lats = []
        all_lngs = []
        
        for shipment in shipments:
            all_lats.append(shipment.origin.lat)
            all_lngs.append(shipment.origin.lng)
            for dest in shipment.destinations:
                all_lats.append(dest.lat)
                all_lngs.append(dest.lng)
        
        centroid_lat = np.mean(all_lats)
        centroid_lng = np.mean(all_lngs)
        
        # Approximate as: distance to centroid + spread around centroid
        centroid_loc = Location(
            place_id="centroid",
            lat=centroid_lat,
            lng=centroid_lng,
            formatted_address="Centroid"
        )
        
        to_centroid = self._distance(vehicle.current_location, centroid_loc)
        spread = np.std(all_lats) + np.std(all_lngs)  # Rough measure of spread
        
        # Estimate: 2 * (distance to center + spread)
        return 2 * (to_centroid + spread * 111)  # 111 km per degree roughly
    
    def evaluate_wait_value(self, state: SystemState,
                            value_function) -> float:
        """
        Estimate value of waiting vs dispatching now
        Used by DLA for lookahead decisions
        
        Returns: Expected value gain from waiting
        """
        if value_function is None:
            # Without VFA, use simple heuristic
            return self._heuristic_wait_value(state)
        
        # Use VFA to estimate
        current_value = value_function.evaluate(state)
        
        # Simulate waiting (assume some new shipments arrive)
        future_state = self._simulate_wait(state)
        future_value = value_function.evaluate(future_state)
        
        return future_value - current_value
    
    def _heuristic_wait_value(self, state: SystemState) -> float:
        """
        Simple heuristic for wait value without VFA
        
        Based on:
        - Current utilization potential
        - Time until deadlines
        - Expected arrival rate
        """
        pending = state.pending_shipments
        
        if not pending:
            return 0.0
        
        # If utilization already high, waiting has low value
        vehicles = state.get_available_vehicles()
        if vehicles:
            avg_vehicle_capacity = np.mean([v.capacity.volume for v in vehicles])
            total_pending_volume = sum(s.volume for s in pending)
            current_util = total_pending_volume / avg_vehicle_capacity
            
            if current_util >= self.config.min_utilization:
                return -10.0  # Negative value, should dispatch now
        
        # If deadlines approaching, waiting has negative value
        avg_time_pressure = np.mean([s.time_pressure(state.timestamp) for s in pending])
        if avg_time_pressure > 0.7:
            return -20.0  # High time pressure, dispatch now
        
        # Otherwise, waiting might improve consolidation
        # Value proportional to underutilization
        max_wait_hours = self.config.get('max_consolidation_wait_hours', 24)
        hours_remaining = min(
            s.time_to_deadline(state.timestamp).total_seconds() / 3600
            for s in pending
        )
        
        if hours_remaining > max_wait_hours:
            return 5.0  # Positive value for waiting
        
        return 0.0
    
    def _simulate_wait(self, state: SystemState,
                      wait_hours: float = 1.0) -> SystemState:
        """
        Simulate waiting for specified hours
        Assumes some new shipments will arrive
        """
        # Simple simulation: add expected number of new shipments
        # In production, this would use learned arrival rates
        
        from copy import deepcopy
        future_state = deepcopy(state)
        future_state.timestamp = state.timestamp + timedelta(hours=wait_hours)
        
        # Assume 1-2 new shipments per hour (placeholder)
        # In reality, use learned arrival distribution
        expected_new = int(wait_hours * 1.5)
        
        # Don't actually add shipments (just for value estimation)
        # Real implementation would sample from arrival distribution
        
        return future_state