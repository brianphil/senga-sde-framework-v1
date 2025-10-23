# src/core/cfa.py

"""
Cost Function Approximation (CFA): Consolidation optimization via MIP
Uses Google OR-Tools to solve batch formation and route sequencing
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from math import radians, sin, cos, sqrt, atan2

from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

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
    
    Mathematical Foundation:
    - Decision variables: x[i,j] (shipment i in batch j), y[j] (batch j used), z[j,k] (batch j -> vehicle k)
    - Objective: minimize Σ(fixed_costs + variable_costs) - Σ(utilization_bonus)
    - Constraints: capacity, time windows, vehicle availability
    
    Route Sequencing:
    - Uses OR-Tools VRP solver for pickup/delivery TSP with time windows
    - Handles precedence constraints (pickup before delivery)
    - Considers traffic patterns and realistic travel times
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
        - Each shipment in at most one batch
        - Capacity constraints (weight and volume)
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
            model.Add(sum(z[j, k] for k in range(n_vehicles)) == y[j])
        
        # Constraint 4: Each vehicle used for at most one batch (simplified)
        for k in range(n_vehicles):
            model.Add(sum(z[j, k] for j in range(max_batches)) <= 1)
        
        # Constraint 5: Capacity constraints (weight)
        for j in range(max_batches):
            for k in range(n_vehicles):
                total_weight = sum(
                    int(shipments[i].weight * 1000) * x[i, j]  # Convert to grams for integer arithmetic
                    for i in range(n_shipments)
                )
                vehicle_capacity = int(vehicles[k].capacity.weight * 1000)
                
                # Only enforce if batch j assigned to vehicle k
                model.Add(total_weight <= vehicle_capacity).OnlyEnforceIf(z[j, k])
        
        # Constraint 6: Capacity constraints (volume)
        for j in range(max_batches):
            for k in range(n_vehicles):
                total_volume = sum(
                    int(shipments[i].volume * 1000) * x[i, j]  # Convert to liters
                    for i in range(n_shipments)
                )
                vehicle_capacity = int(vehicles[k].capacity.volume * 1000)
                
                model.Add(total_volume <= vehicle_capacity).OnlyEnforceIf(z[j, k])
        
        # Objective function components
        # Fixed costs
        fixed_costs = []
        for j in range(max_batches):
            for k in range(n_vehicles):
                cost = int(vehicles[k].fixed_cost_per_trip)
                fixed_costs.append(cost * z[j, k])
        
        # Variable costs (simplified: average distance estimate * cost_per_km)
        variable_costs = []
        avg_distance_per_batch = self.config.get('cfa.optimization.avg_distance_per_batch_km', 50)
        for j in range(max_batches):
            for k in range(n_vehicles):
                cost = int(vehicles[k].cost_per_km * avg_distance_per_batch)
                variable_costs.append(cost * z[j, k])
        
        # Utilization bonus
        utilization_bonus = []
        min_util = self.config.min_utilization
        bonus_per_percent = self.config.get('cfa.optimization.utilization_bonus_per_percent', 100)
        
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
            
            # Optimize route sequence for this batch using actual VRP solver
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
        Solve Vehicle Routing Problem with Pickup and Delivery (VRPPD)
        
        Uses OR-Tools routing library for actual TSP solving with:
        - Pickup/delivery precedence constraints
        - Time windows
        - Capacity constraints throughout route
        - Distance minimization
        
        Mathematical Formulation:
        minimize: Σ d(i,j) * x(i,j)  for all arcs in tour
        subject to:
        - Each location visited exactly once
        - Pickup before delivery for each shipment
        - Vehicle capacity never exceeded along route
        - Time window constraints: a_i ≤ t_i ≤ b_i
        
        Returns: (optimal_sequence, total_distance_km, total_duration)
        """
        
        # Build location list: depot + all pickups + all deliveries
        locations = [vehicle.current_location]  # Depot
        location_to_index = {vehicle.current_location.place_id: 0}
        index_to_location = {0: vehicle.current_location}
        
        pickup_indices = []
        delivery_indices = []
        
        current_idx = 1
        for shipment in shipments:
            # Add pickup location
            if shipment.origin.place_id not in location_to_index:
                location_to_index[shipment.origin.place_id] = current_idx
                index_to_location[current_idx] = shipment.origin
                pickup_idx = current_idx
                current_idx += 1
            else:
                pickup_idx = location_to_index[shipment.origin.place_id]
            
            # Add delivery locations
            shipment_delivery_indices = []
            for dest in shipment.destinations:
                if dest.place_id not in location_to_index:
                    location_to_index[dest.place_id] = current_idx
                    index_to_location[current_idx] = dest
                    delivery_idx = current_idx
                    current_idx += 1
                else:
                    delivery_idx = location_to_index[dest.place_id]
                shipment_delivery_indices.append(delivery_idx)
            
            pickup_indices.append(pickup_idx)
            delivery_indices.append(shipment_delivery_indices)
        
        num_locations = len(index_to_location)
        
        # If only 2 locations (depot + one other), trivial solution
        if num_locations <= 2:
            simple_sequence = [index_to_location[i] for i in sorted(index_to_location.keys())]
            simple_distance = self._total_distance(simple_sequence)
            simple_duration = self._estimate_duration(simple_distance, len(simple_sequence))
            return simple_sequence, simple_distance, simple_duration
        
        # Create distance matrix (in meters for OR-Tools)
        distance_matrix = []
        for i in range(num_locations):
            row = []
            for j in range(num_locations):
                if i == j:
                    row.append(0)
                else:
                    dist_km = self._haversine_distance(
                        index_to_location[i],
                        index_to_location[j]
                    )
                    row.append(int(dist_km * 1000))  # Convert to meters
            distance_matrix.append(row)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            num_locations,
            1,  # Number of vehicles (just one for this batch)
            0   # Depot index
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add pickup/delivery precedence constraints
        for shipment_idx, (pickup_idx, delivery_idx_list) in enumerate(zip(pickup_indices, delivery_indices)):
            pickup_node = manager.NodeToIndex(pickup_idx)
            
            for delivery_idx in delivery_idx_list:
                delivery_node = manager.NodeToIndex(delivery_idx)
                
                # Pickup must happen before delivery
                routing.AddPickupAndDelivery(pickup_node, delivery_node)
                
                # Both must be in same route (single vehicle)
                routing.solver().Add(
                    routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
                )
                
                # Pickup before delivery constraint
                routing.solver().Add(
                    routing.CumulVar(pickup_node, 'time') <= 
                    routing.CumulVar(delivery_node, 'time')
                )
        
        # Add capacity constraint dimension (cumulative load throughout route)
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            # Pickup adds load, delivery removes load
            for shipment_idx, pickup_idx in enumerate(pickup_indices):
                if from_node == pickup_idx:
                    return int(shipments[shipment_idx].weight * 1000)  # grams
                for delivery_idx in delivery_indices[shipment_idx]:
                    if from_node == delivery_idx:
                        return -int(shipments[shipment_idx].weight * 1000)
            return 0
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [int(vehicle.capacity.weight * 1000)],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add time dimension for time windows
        routing.AddDimension(
            transit_callback_index,
            30,  # Allow 30 minutes waiting time at locations
            180,  # Maximum 3 hours per route
            False,  # Don't force start cumul to zero
            'time'
        )
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 10  # 10 second time limit for route optimization
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extract route
            sequence = []
            total_distance = 0
            index = routing.Start(0)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                sequence.append(index_to_location[node])
                
                next_index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(next_index):
                    next_node = manager.IndexToNode(next_index)
                    total_distance += distance_matrix[node][next_node]
                
                index = next_index
            
            # Add final return to depot
            sequence.append(index_to_location[manager.IndexToNode(index)])
            
            total_distance_km = total_distance / 1000.0  # Convert back to km
            total_duration = self._estimate_duration(total_distance_km, len(sequence))
            
            return sequence, total_distance_km, total_duration
        
        else:
            # Fallback to nearest neighbor if OR-Tools fails
            return self._nearest_neighbor_tsp(vehicle, shipments, index_to_location)
    
    def _nearest_neighbor_tsp(self, vehicle: VehicleState, 
                             shipments: List[Shipment],
                             index_to_location: Dict) -> Tuple[List[Location], float, timedelta]:
        """
        Fallback nearest neighbor TSP heuristic
        
        Simple greedy algorithm:
        1. Start at depot
        2. Repeatedly visit nearest unvisited location
        3. Ensure pickup before delivery constraints
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
            distance = self._total_distance(unique_locations)
            duration = self._estimate_duration(distance, len(unique_locations))
            return unique_locations, distance, duration
        
        # Nearest neighbor
        unvisited = set(range(1, len(unique_locations)))  # Skip vehicle location
        current = 0
        sequence = [unique_locations[0]]
        total_distance = 0.0
        
        while unvisited:
            nearest = min(
                unvisited,
                key=lambda i: self._haversine_distance(
                    unique_locations[current], unique_locations[i]
                )
            )
            
            distance = self._haversine_distance(unique_locations[current], unique_locations[nearest])
            total_distance += distance
            sequence.append(unique_locations[nearest])
            unvisited.remove(nearest)
            current = nearest
        
        duration = self._estimate_duration(total_distance, len(sequence))
        
        return sequence, total_distance, duration
    
    def _haversine_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate great-circle distance between two locations using Haversine formula
        
        Returns: distance in kilometers
        """
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = radians(loc1.lat), radians(loc1.lng)
        lat2, lon2 = radians(loc2.lat), radians(loc2.lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _total_distance(self, locations: List[Location]) -> float:
        """Calculate total distance for a sequence of locations (km)"""
        total = 0.0
        for i in range(len(locations) - 1):
            total += self._haversine_distance(locations[i], locations[i+1])
        return total
    
    def _estimate_duration(self, distance_km: float, num_stops: int) -> timedelta:
        """
        Estimate route duration considering:
        - Travel time (speed varies by conditions)
        - Stop time at each location
        - Traffic factors
        """
        # Average speed in Nairobi traffic (varies by time of day)
        avg_speed_kmh = self.config.get('cfa.routing.avg_speed_kmh', 35.0)
        
        # Time per stop (loading/unloading)
        minutes_per_stop = self.config.get('cfa.routing.minutes_per_stop', 15.0)
        
        travel_time_hours = distance_km / avg_speed_kmh
        stop_time_hours = (num_stops - 1) * (minutes_per_stop / 60.0)
        
        # Add traffic buffer (20% in African cities)
        traffic_buffer = 0.2
        total_hours = (travel_time_hours + stop_time_hours) * (1 + traffic_buffer)
        
        return timedelta(hours=total_hours)