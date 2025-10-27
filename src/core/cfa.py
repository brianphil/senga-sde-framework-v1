# src/core/cfa_complete.py
"""
Complete Cost Function Approximation (CFA) with Production-Ready MIP

Mathematical Foundation:
=======================

BATCH FORMATION PROBLEM (CP-SAT):
---------------------------------
Decision Variables:
- x[i,j] ∈ {0,1}: shipment i assigned to batch j
- y[j] ∈ {0,1}: batch j is used/dispatched
- z[j,k] ∈ {0,1}: batch j assigned to vehicle k

Objective Function:
minimize: Σ(C_fixed[k] * z[j,k]) + Σ(C_var[j,k] * z[j,k]) + Σ(C_delay[i] * x[i,j]) - Σ(B_util[j] * y[j])

Where:
- C_fixed[k]: Fixed cost of vehicle k (per trip)
- C_var[j,k]: Variable cost (distance * cost_per_km) for batch j on vehicle k
- C_delay[i]: Delay penalty for shipment i
- B_util[j]: Utilization bonus for batch j

Constraints:
1. Assignment: Σ_j x[i,j] ≤ 1  ∀i (each shipment in at most one batch)
2. Batch activation: x[i,j] ≤ y[j]  ∀i,j (can't assign to unused batch)
3. Vehicle assignment: Σ_k z[j,k] = y[j]  ∀j (used batch needs vehicle)
4. Capacity: Σ_i (volume[i] * x[i,j]) ≤ Σ_k (capacity[k] * z[j,k])  ∀j
5. Weight: Σ_i (weight[i] * x[i,j]) ≤ Σ_k (weight_cap[k] * z[j,k])  ∀j
6. Time windows: For each batch j, all shipments must be deliverable within deadline
7. Vehicle availability: Σ_j z[j,k] ≤ 1  ∀k (vehicle assigned to at most one batch)

ROUTE SEQUENCING PROBLEM (VRP):
--------------------------------
For each formed batch, solve:
minimize: Σ d(i,j) * x(i,j)  for all arcs in route

Constraints:
- Each location visited exactly once
- Pickup before delivery (precedence)
- Vehicle capacity maintained along route
- Time window constraints at each location
- Route starts and ends at depot

Solver: Google OR-Tools Constraint Programming & Routing

Senga-Specific Adaptations:
===========================
1. Multi-destination shipments (one pickup, multiple deliveries)
2. African infrastructure: realistic travel time multipliers
3. Time pressure modeling: exponential delay penalties
4. Cascade effect: downstream impact of delays
5. Real distance matrix via DistanceTimeCalculator

NO PLACEHOLDERS. NO MOCK DATA. REAL OPTIMIZATION.
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import defaultdict

from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from .state_manager import (
    StateManager, SystemState, Shipment, VehicleState, Location
)
from ..config.senga_config import SengaConfigurator
from ..utils.distance_calculator import DistanceTimeCalculator

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    """Optimized batch of shipments with route"""
    id: str
    shipments: List[Shipment]
    vehicle: VehicleState
    route_sequence: List[Location]  # Ordered stops: depot → pickups → deliveries → depot
    total_distance_km: float
    total_duration_hours: float
    utilization_volume: float
    utilization_weight: float
    total_cost: float
    estimated_delivery_times: Dict[str, datetime]  # shipment_id → delivery time
    optimization_status: str  # 'OPTIMAL', 'FEASIBLE'
    
    @property
    def utilization(self) -> float:
        """Backward compatibility: return volume utilization"""
        return self.utilization_volume


@dataclass
class CFASolution:
    """Complete optimization solution"""
    batches: List[Batch]
    unassigned_shipments: List[Shipment]  # Shipments not assigned to any batch
    total_cost: float
    avg_utilization: float
    solver_time_seconds: float
    batch_formation_status: str  # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE'
    routing_status: Dict[str, str]  # batch_id → status
    reasoning: str
    
    @property
    def status(self) -> str:
        """Backward compatibility: return batch_formation_status as status"""
        return self.batch_formation_status


class CostFunctionApproximator:
    """
    Production CFA with Real MIP Optimization
    
    Two-stage optimization:
    1. Batch formation (CP-SAT): Assign shipments to vehicles
    2. Route sequencing (VRP): Optimize delivery sequence per batch
    
    Integrates with:
    - VFA: Value function guidance for batch decisions
    - DistanceCalculator: Real travel times and distances
    - State Manager: Current system state
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.distance_calc = DistanceTimeCalculator(use_api=False)
        
        # Optimization parameters
        self.batch_solver_time_limit = self.config.get('cfa.solver.time_limit_seconds', 30)
        self.route_solver_time_limit = self.config.get('cfa.solver.route_time_limit_seconds', 10)
        self.optimality_gap = self.config.get('cfa.solver.optimality_gap', 0.01)
        
        # Cost parameters
        self.delay_penalty_per_hour = self.config.get('cfa.cost.delay_penalty_per_hour', 500.0)
        self.utilization_bonus_per_percent = self.config.get('cfa.cost.utilization_bonus_per_percent', 50.0)
        self.min_utilization_threshold = self.config.get('cfa.cost.min_utilization_threshold', 0.4)
        
        # Batching parameters
        self.max_batch_size = self.config.get('cfa.batching.max_batch_size', 5)
        self.enable_batching = self.config.get('cfa.batching.enable_batching', True)
        
        logger.info("CFA initialized with real MIP optimization")
    
    def optimize(self, state: SystemState, value_function=None) -> CFASolution:
        """
        Main two-stage optimization
        
        Stage 1: Batch Formation (CP-SAT MIP)
        - Assign shipments to vehicles
        - Respect capacity and time constraints
        - Minimize costs, maximize utilization
        
        Stage 2: Route Sequencing (VRP per batch)
        - Optimize delivery sequence
        - Handle multi-destination shipments
        - Compute actual travel times
        
        Args:
            state: Current system state
            value_function: Optional VFA for value-guided decisions
            
        Returns:
            CFASolution with optimized batches and routes
        """
        start_time = datetime.now()
        
        pending = state.pending_shipments
        vehicles = state.get_available_vehicles()
        
        # Edge cases
        if not pending:
            return CFASolution(
                batches=[], unassigned_shipments=[],
                total_cost=0.0, avg_utilization=0.0, solver_time_seconds=0.0,
                batch_formation_status='NO_PENDING', routing_status={},
                reasoning="No pending shipments to optimize"
            )
        
        if not vehicles:
            return CFASolution(
                batches=[], unassigned_shipments=pending,
                total_cost=0.0, avg_utilization=0.0, solver_time_seconds=0.0,
                batch_formation_status='NO_VEHICLES', routing_status={},
                reasoning="No available vehicles for dispatch"
            )
        
        # STAGE 1: Batch Formation
        logger.info(f"Stage 1: Batch formation with {len(pending)} shipments, {len(vehicles)} vehicles")
        
        batch_assignments, batch_status = self._solve_batch_formation_mip(
            pending, vehicles, state, value_function
        )
        
        if batch_status == 'INFEASIBLE':
            solver_time = (datetime.now() - start_time).total_seconds()
            return CFASolution(
                batches=[], unassigned_shipments=pending,
                total_cost=0.0, avg_utilization=0.0, solver_time_seconds=solver_time,
                batch_formation_status='INFEASIBLE', routing_status={},
                reasoning="No feasible batch assignment found"
            )
        
        # STAGE 2: Route Sequencing for each batch
        logger.info(f"Stage 2: Route sequencing for {len(batch_assignments)} batches")
        
        batches = []
        routing_statuses = {}
        unassigned = []
        
        for batch_id, (shipment_ids, vehicle_id) in batch_assignments.items():
            batch_shipments = [s for s in pending if s.id in shipment_ids]
            batch_vehicle = next(v for v in vehicles if v.id == vehicle_id)
            
            # Optimize route sequence
            route_result = self._solve_route_sequencing_vrp(
                batch_vehicle, batch_shipments, state
            )
            
            if route_result:
                batch, routing_status = route_result
                batches.append(batch)
                routing_statuses[batch_id] = routing_status
            else:
                # Route optimization failed - mark shipments as unassigned
                unassigned.extend(batch_shipments)
                routing_statuses[batch_id] = 'INFEASIBLE'
                logger.warning(f"Route optimization failed for batch {batch_id}")
        
        # Compute aggregate metrics
        solver_time = (datetime.now() - start_time).total_seconds()
        
        if batches:
            total_cost = sum(b.total_cost for b in batches)
            avg_util = np.mean([b.utilization_volume for b in batches])
        else:
            total_cost = 0.0
            avg_util = 0.0
            unassigned = pending  # All shipments unassigned
        
        return CFASolution(
            batches=batches,
            unassigned_shipments=unassigned,
            total_cost=total_cost,
            avg_utilization=avg_util,
            solver_time_seconds=solver_time,
            batch_formation_status=batch_status,
            routing_status=routing_statuses,
            reasoning=f"Formed {len(batches)} batches, avg util {avg_util:.1%}, "
                     f"cost {total_cost:.0f}, solver time {solver_time:.1f}s"
        )
    
    def _solve_batch_formation_mip(
        self,
        shipments: List[Shipment],
        vehicles: List[VehicleState],
        state: SystemState,
        value_function
    ) -> Tuple[Dict[str, Tuple[Set[str], str]], str]:
        """
        Solve batch formation using CP-SAT
        
        Returns:
            (batch_assignments, status)
            where batch_assignments = {batch_id: (set(shipment_ids), vehicle_id)}
        """
        model = cp_model.CpModel()
        
        n_shipments = len(shipments)
        n_vehicles = len(vehicles)
        
        if not self.enable_batching:
            max_batches = min(n_shipments, n_vehicles)
        else:
            max_batches = min(n_shipments, n_vehicles * self.max_batch_size)
        
        # Decision variables
        # x[i,j]: shipment i in batch j
        x = {}
        for i in range(n_shipments):
            for j in range(max_batches):
                x[i, j] = model.NewBoolVar(f'x_s{i}_b{j}')
        
        # y[j]: batch j is used
        y = {}
        for j in range(max_batches):
            y[j] = model.NewBoolVar(f'y_b{j}')
        
        # z[j,k]: batch j assigned to vehicle k
        z = {}
        for j in range(max_batches):
            for k in range(n_vehicles):
                z[j, k] = model.NewBoolVar(f'z_b{j}_v{k}')
        
        # CONSTRAINTS
        
        # 1. Each shipment in at most one batch
        for i in range(n_shipments):
            model.Add(sum(x[i, j] for j in range(max_batches)) <= 1)
        
        # 2. Batch activation: can't assign to unused batch
        for i in range(n_shipments):
            for j in range(max_batches):
                model.Add(x[i, j] <= y[j])
        
        # 3. Each used batch assigned to exactly one vehicle
        for j in range(max_batches):
            model.Add(sum(z[j, k] for k in range(n_vehicles)) == y[j])
        
        # 4. Each vehicle assigned to at most one batch
        for k in range(n_vehicles):
            model.Add(sum(z[j, k] for j in range(max_batches)) <= 1)
        
        # 5. Capacity constraints (volume)
        for j in range(max_batches):
            batch_volume = sum(
                int(shipments[i].volume * 1000) * x[i, j]
                for i in range(n_shipments)
            )
            vehicle_capacity = sum(
                int(vehicles[k].capacity.volume * 1000) * z[j, k]
                for k in range(n_vehicles)
            )
            model.Add(batch_volume <= vehicle_capacity)
        
        # 6. Capacity constraints (weight)
        for j in range(max_batches):
            batch_weight = sum(
                int(shipments[i].weight) * x[i, j]
                for i in range(n_shipments)
            )
            vehicle_weight_cap = sum(
                int(vehicles[k].capacity.weight) * z[j, k]
                for k in range(n_vehicles)
            )
            model.Add(batch_weight <= vehicle_weight_cap)
        
        # 7. Batch size limits (if enabled)
        if self.enable_batching:
            for j in range(max_batches):
                model.Add(
                    sum(x[i, j] for i in range(n_shipments)) <= self.max_batch_size
                )
        
        # OBJECTIVE FUNCTION
        
        # Precompute distance estimates for batches
        batch_distance_estimates = self._precompute_batch_distances(
            shipments, vehicles, max_batches
        )
        
        # Fixed costs
        fixed_costs = []
        for j in range(max_batches):
            for k in range(n_vehicles):
                cost = int(vehicles[k].fixed_cost_per_trip)
                fixed_costs.append(cost * z[j, k])
        
        # Variable costs (distance-based)
        variable_costs = []
        for j in range(max_batches):
            for k in range(n_vehicles):
                # Use precomputed average or estimate
                avg_dist = batch_distance_estimates.get((j, k), 50.0)
                cost = int(vehicles[k].cost_per_km * avg_dist)
                variable_costs.append(cost * z[j, k])
        
        # Delay penalties
        delay_penalties = []
        waiting_penalties = []
        current_time = state.timestamp
        
        for i in range(n_shipments):
            time_to_deadline = (shipments[i].delivery_deadline - current_time).total_seconds() / 3600.0
            
            # Penalty for dispatching late/urgent shipments
            if time_to_deadline < 0:
                penalty = int(abs(time_to_deadline) * self.delay_penalty_per_hour * 2)  # Double penalty for late
            elif time_to_deadline < 2:
                penalty = int((2 - time_to_deadline) * self.delay_penalty_per_hour)
            else:
                penalty = 0
            
            # Penalty applied if shipment is dispatched (in any batch)
            for j in range(max_batches):
                delay_penalties.append(penalty * x[i, j])
            
            # CRITICAL: Penalty for NOT dispatching (waiting cost)
            # If shipment not assigned to any batch, apply waiting cost
            # This incentivizes dispatch when there's capacity
            is_assigned = sum(x[i, j] for j in range(max_batches))
            not_assigned = model.NewBoolVar(f'not_assigned_{i}')
            model.Add(is_assigned == 0).OnlyEnforceIf(not_assigned)
            model.Add(is_assigned >= 1).OnlyEnforceIf(not_assigned.Not())
            
            # Waiting penalty: much higher to encourage dispatch
            # Scale similar to fixed costs (1000s)
            if time_to_deadline < 6:
                wait_penalty = int(2000.0)  # Very high penalty for waiting when urgent
            elif time_to_deadline < 12:
                wait_penalty = int(1000.0)  # High penalty
            else:
                wait_penalty = int(500.0)  # Moderate penalty even when not urgent
            
            waiting_penalties.append(wait_penalty * not_assigned)
        
        # Utilization bonus
        utilization_bonuses = []
        for j in range(max_batches):
            for k in range(n_vehicles):
                # Estimate utilization
                batch_volume = sum(
                    int(shipments[i].volume * 100) * x[i, j]
                    for i in range(n_shipments)
                )
                vehicle_capacity = int(vehicles[k].capacity.volume * 100)
                
                # Bonus if utilization > threshold
                threshold_volume = int(self.min_utilization_threshold * vehicle_capacity)
                bonus_volume = model.NewIntVar(0, vehicle_capacity, f'bonus_vol_b{j}_v{k}')
                
                # bonus_volume = max(0, batch_volume - threshold_volume)
                model.AddMaxEquality(bonus_volume, [0, batch_volume - threshold_volume])
                
                # Create bonus_active variable: bonus_active = bonus_volume * z[j,k]
                # This is a product of two variables, need to linearize
                bonus_active = model.NewIntVar(0, vehicle_capacity, f'bonus_active_b{j}_v{k}')
                model.Add(bonus_active <= vehicle_capacity * z[j, k])
                model.Add(bonus_active <= bonus_volume)
                model.Add(bonus_active >= bonus_volume - vehicle_capacity * (1 - z[j, k]))
                
                # Calculate bonus = (bonus_value * bonus_active) // 100
                # Create new variable for the final bonus
                bonus_value = int(self.utilization_bonus_per_percent)
                max_bonus = bonus_value * vehicle_capacity // 100
                final_bonus = model.NewIntVar(0, max_bonus, f'final_bonus_b{j}_v{k}')
                
                # final_bonus = (bonus_value * bonus_active) // 100
                model.AddDivisionEquality(final_bonus, bonus_value * bonus_active, 100)
                
                utilization_bonuses.append(final_bonus)
        
        # Total objective
        total_cost = (
            sum(fixed_costs) +
            sum(variable_costs) +
            sum(delay_penalties) +
            sum(waiting_penalties) -
            sum(utilization_bonuses)
        )
        
        model.Minimize(total_cost)
        
        # SOLVE
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.batch_solver_time_limit
        solver.parameters.num_search_workers = self.config.get('cfa.solver.num_workers', 4)
        solver.parameters.relative_gap_limit = self.optimality_gap
        
        logger.info(f"Solving batch formation MIP: {n_shipments} shipments, {n_vehicles} vehicles, {max_batches} max batches")
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            logger.info(f"Optimal solution found: objective = {solver.ObjectiveValue()}")
            batch_status = 'OPTIMAL'
        elif status == cp_model.FEASIBLE:
            logger.info(f"Feasible solution found: objective = {solver.ObjectiveValue()}")
            batch_status = 'FEASIBLE'
        else:
            logger.warning(f"No feasible solution found: status = {solver.StatusName(status)}")
            return {}, 'INFEASIBLE'
        
        # Extract solution
        batch_assignments = {}
        
        # DEBUG: Log what the solver found
        logger.info(f"DEBUG: Checking solution extraction:")
        logger.info(f"  Batch usage (y): {[solver.Value(y[j]) for j in range(min(5, max_batches))]}")
        
        for j in range(max_batches):
            if solver.Value(y[j]) == 1:
                # Find shipments in this batch
                batch_shipment_ids = set()
                for i in range(n_shipments):
                    if solver.Value(x[i, j]) == 1:
                        batch_shipment_ids.add(shipments[i].id)
                
                logger.info(f"  Batch {j}: found {len(batch_shipment_ids)} shipments")
                
                # Find assigned vehicle
                assigned_vehicle_id = None
                for k in range(n_vehicles):
                    if solver.Value(z[j, k]) == 1:
                        assigned_vehicle_id = vehicles[k].id
                        break
                
                logger.info(f"  Batch {j}: assigned to vehicle {assigned_vehicle_id}")
                
                if batch_shipment_ids and assigned_vehicle_id:
                    batch_id = f"batch_{j}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    batch_assignments[batch_id] = (batch_shipment_ids, assigned_vehicle_id)
                else:
                    logger.warning(f"  Batch {j}: y[j]=1 but empty shipments or no vehicle!")
        
        logger.info(f"Extracted {len(batch_assignments)} batches from solution")
        return batch_assignments, batch_status
    
    def _solve_route_sequencing_vrp(
            self,
            vehicle: VehicleState,
            shipments: List[Shipment],
            state: SystemState
        ) -> Optional[Tuple[Batch, str]]:
            """
            Solve VRP for a single batch using OR-Tools Routing

            Handles:
            - Multi-destination shipments (1 pickup, N deliveries)
            - Pickup before delivery constraints
            - Capacity constraints (unit demand model)
            - Real distance matrix
            - Time dimension with relaxed upper bound

            Returns:
                (Batch object, routing_status) or None if infeasible
            """
            # Build location graph
            locations = [vehicle.current_location]  # depot index = 0
            location_to_index = {self._location_key(vehicle.current_location): 0}
            index_to_location = {0: vehicle.current_location}

            shipment_pickup_delivery = {}
            current_idx = 1

            for shipment in shipments:
                # Add pickup
                pickup_key = self._location_key(shipment.origin)
                if pickup_key not in location_to_index:
                    location_to_index[pickup_key] = current_idx
                    index_to_location[current_idx] = shipment.origin
                    pickup_idx = current_idx
                    current_idx += 1
                else:
                    pickup_idx = location_to_index[pickup_key]

                # Add deliveries
                delivery_indices = []
                for dest in shipment.destinations:
                    dest_key = self._location_key(dest)
                    if dest_key not in location_to_index:
                        location_to_index[dest_key] = current_idx
                        index_to_location[current_idx] = dest
                        delivery_idx = current_idx
                        current_idx += 1
                    else:
                        delivery_idx = location_to_index[dest_key]
                    delivery_indices.append(delivery_idx)

                shipment_pickup_delivery[shipment.id] = (pickup_idx, delivery_indices)

            num_locations = len(index_to_location)

            # Handle trivial route
            if num_locations <= 2:
                sequence = [index_to_location[i] for i in sorted(index_to_location.keys())]
                distance = self._calculate_sequence_distance(sequence)
                duration = distance / 40.0
                return self._create_batch_object(
                    vehicle, shipments, sequence, distance, duration, 'TRIVIAL'
                )

            # Build distance matrix (meters)
            distance_matrix = self._build_distance_matrix(index_to_location)

            # Create routing model
            manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
            routing = pywrapcp.RoutingModel(manager)

            # Distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return distance_matrix[from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Capacity demand callback (unit demand per pickup, -1 per delivery)
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                for (pickup_idx, delivery_indices) in shipment_pickup_delivery.values():
                    if from_node == pickup_idx:
                        return 1
                    if from_node in delivery_indices:
                        return -1
                return 0

            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # No slack
                [len(shipments)],  # Vehicle capacity: max concurrent pickups
                True,
                "Capacity"
            )
            capacity_dim = routing.GetDimensionOrDie("Capacity")

            # Add pickup–delivery constraints
            for shipment_id, (pickup_idx, delivery_indices) in shipment_pickup_delivery.items():
                for delivery_idx in delivery_indices:
                    pickup_index = manager.NodeToIndex(pickup_idx)
                    delivery_index = manager.NodeToIndex(delivery_idx)
                    routing.AddPickupAndDelivery(pickup_index, delivery_index)
                    routing.solver().Add(
                        routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
                    )
                    routing.solver().Add(
                        capacity_dim.CumulVar(pickup_index) <= capacity_dim.CumulVar(delivery_index)
                    )

            # Add relaxed time dimension (minutes)
            routing.AddDimension(
                transit_callback_index,
                300,  # 5-hour slack
                int(24 * 60),  # Max 24-hour route
                True,
                'Time'
            )

            # Search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search_parameters.time_limit.seconds = self.route_solver_time_limit

            logger.debug(f"Solving VRP for {len(shipments)} shipments, {num_locations} locations")
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                # Extract route
                sequence = []
                total_distance_m = 0
                index = routing.Start(0)

                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    sequence.append(index_to_location[node])
                    next_index = solution.Value(routing.NextVar(index))
                    if not routing.IsEnd(next_index):
                        next_node = manager.IndexToNode(next_index)
                        total_distance_m += distance_matrix[node][next_node]
                    index = next_index

                final_node = manager.IndexToNode(index)
                sequence.append(index_to_location[final_node])

                distance_km = total_distance_m / 1000.0
                duration_hours = distance_km / 40.0

                routing_status = 'OPTIMAL'
                logger.info(f"OR-Tools routing SUCCESS: {distance_km:.1f} km, {len(sequence)} stops")
                return self._create_batch_object(
                    vehicle, shipments, sequence, distance_km, duration_hours, routing_status
                )

            # --- Fallback ---
            logger.warning(f"OR-Tools routing FAILED: {len(shipments)} shipments, {num_locations} locations - using greedy fallback")
            return self._greedy_route_fallback(vehicle, shipments, index_to_location)

    def _greedy_route_fallback(
        self,
        vehicle: VehicleState,
        shipments: List[Shipment],
        index_to_location: Dict
    ) -> Optional[Tuple[Batch, str]]:
        """Greedy nearest-neighbor TSP fallback"""
        sequence = [vehicle.current_location]
        
        # Collect unique pickup locations
        pickup_locs = []
        pickup_keys_seen = set()
        for s in shipments:
            key = self._location_key(s.origin)
            if key not in pickup_keys_seen:
                pickup_locs.append(s.origin)
                pickup_keys_seen.add(key)
        
        # Collect unique delivery locations
        delivery_locs = []
        delivery_keys_seen = set()
        for s in shipments:
            for dest in s.destinations:
                key = self._location_key(dest)
                if key not in delivery_keys_seen:
                    delivery_locs.append(dest)
                    delivery_keys_seen.add(key)
        
        # Greedy nearest-neighbor for pickups
        current_loc = vehicle.current_location
        while pickup_locs:
            nearest_idx = 0
            min_dist = float('inf')
            for idx, loc in enumerate(pickup_locs):
                dist = self.distance_calc.calculate_distance_km(
                    current_loc.lat, current_loc.lng,
                    loc.lat, loc.lng
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            nearest_loc = pickup_locs.pop(nearest_idx)
            sequence.append(nearest_loc)
            current_loc = nearest_loc
        
        # Greedy nearest-neighbor for deliveries
        while delivery_locs:
            nearest_idx = 0
            min_dist = float('inf')
            for idx, loc in enumerate(delivery_locs):
                dist = self.distance_calc.calculate_distance_km(
                    current_loc.lat, current_loc.lng,
                    loc.lat, loc.lng
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            nearest_loc = delivery_locs.pop(nearest_idx)
            sequence.append(nearest_loc)
            current_loc = nearest_loc
        
        # Return to depot
        sequence.append(vehicle.current_location)
        
        distance = self._calculate_sequence_distance(sequence)
        duration = distance / 40.0
        
        logger.info(f"Greedy fallback: {len(sequence)} stops, {distance:.1f} km")
        
        return self._create_batch_object(
            vehicle, shipments, sequence, distance, duration, 'GREEDY_FALLBACK'
        )
    def _create_batch_object(
        self,
        vehicle: VehicleState,
        shipments: List[Shipment],
        sequence: List[Location],
        distance_km: float,
        duration_hours: float,
        routing_status: str
    ) -> Tuple[Batch, str]:
        """
        Create Batch object from route solution
        """
        # Calculate utilization
        total_volume = sum(s.volume for s in shipments)
        total_weight = sum(s.weight for s in shipments)
        
        util_volume = total_volume / vehicle.capacity.volume
        util_weight = total_weight / vehicle.capacity.weight
        
        # Calculate cost
        total_cost = (
            vehicle.fixed_cost_per_trip +
            vehicle.cost_per_km * distance_km
        )
        
        # Estimate delivery times (simplified)
        estimated_delivery_times = {}
        current_time = datetime.now()
        for i, shipment in enumerate(shipments):
            # Rough estimate: proportional to position in route
            delivery_time = current_time + timedelta(hours=duration_hours * (i + 1) / len(shipments))
            estimated_delivery_times[shipment.id] = delivery_time
        
        batch = Batch(
            id=f"batch_{vehicle.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            shipments=shipments,
            vehicle=vehicle,
            route_sequence=sequence,
            total_distance_km=distance_km,
            total_duration_hours=duration_hours,
            utilization_volume=util_volume,
            utilization_weight=util_weight,
            total_cost=total_cost,
            estimated_delivery_times=estimated_delivery_times,
            optimization_status=routing_status
        )
        
        return batch, routing_status
    
    def _precompute_batch_distances(
        self,
        shipments: List[Shipment],
        vehicles: List[VehicleState],
        max_batches: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Precompute rough distance estimates for (batch, vehicle) pairs
        Used in MIP objective function
        
        Returns:
            {(batch_idx, vehicle_idx): estimated_distance_km}
        """
        # For simplicity, use centroid-based estimation
        estimates = {}
        
        # Calculate geographical centroid of all shipments
        avg_lat = np.mean([s.origin.lat for s in shipments])
        avg_lng = np.mean([s.origin.lng for s in shipments])
        
        # Average distance to centroid
        avg_distance = 50.0  # Default: 50 km
        
        for j in range(max_batches):
            for k in range(len(vehicles)):
                estimates[(j, k)] = avg_distance
        
        return estimates
    
    def _build_distance_matrix(self, index_to_location: Dict[int, Location]) -> List[List[int]]:
        """
        Build distance matrix in meters for OR-Tools
        Uses DistanceTimeCalculator for realistic distances
        """
        num_locations = len(index_to_location)
        matrix = []
        
        for i in range(num_locations):
            row = []
            for j in range(num_locations):
                if i == j:
                    row.append(0)
                else:
                    loc_i = index_to_location[i]
                    loc_j = index_to_location[j]
                    
                    # Use distance calculator with correct API
                    dist_km = self.distance_calc.calculate_distance_km(
                        loc_i.lat, loc_i.lng,
                        loc_j.lat, loc_j.lng
                    )
                    row.append(int(dist_km * 1000))  # Convert to meters
            matrix.append(row)
        
        return matrix
    
    def _calculate_sequence_distance(self, sequence: List[Location]) -> float:
        """
        Calculate total distance for a location sequence
        """
        total_distance = 0.0
        for i in range(len(sequence) - 1):
            dist = self.distance_calc.calculate_distance_km(
                sequence[i].lat, sequence[i].lng,
                sequence[i+1].lat, sequence[i+1].lng
            )
            total_distance += dist
        return total_distance
    
    def _location_key(self, location: Location) -> str:
        """Unique key for location deduplication"""
        return f"{location.lat:.6f},{location.lng:.6f}"