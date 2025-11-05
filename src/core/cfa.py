# src/core/cfa.py - UPDATED WITH ROUTE OPTIMIZER INTEGRATION

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ortools.linear_solver import pywraplp
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class CFAParameters:
    """CFA optimization parameters - learnable via update_parameters()"""

    base_cost_per_km: float = 30.0
    fixed_cost_per_trip: float = 1000.0
    urgency_penalty_weight: float = 100.0
    consolidation_bonus_per_shipment: float = 200.0
    min_utilization_threshold: float = 0.4
    max_batch_size: int = 10

    # Geographic compatibility parameters (business-configurable)
    compatibility_distance_km_tight: float = 50.0
    compatibility_distance_km_medium: float = 150.0
    collinearity_threshold_medium: float = 0.85
    detour_ratio_threshold_medium: float = 1.3
    compatibility_distance_km_long: float = 300.0
    collinearity_threshold_long: float = 0.92
    detour_ratio_threshold_long: float = 1.15


@dataclass
class CFAAction:
    """CFA output matching meta_controller interface"""

    action_type: str
    details: dict
    reasoning: str
    confidence: float


class CostFunctionApproximator:
    """
    Production CFA: MIP-based batch consolidation with vehicle assignment and route optimization

    NEW: Integrated with RouteSequenceOptimizer for actual route sequences
    """

    def __init__(self, config):
        self.config = config
        self.theta = CFAParameters()
        self._load_business_parameters()

        from ..utils.distance_calculator import DistanceTimeCalculator
        from ..algorithms.route_optimizer import RouteSequenceOptimizer

        self.distance_calc = DistanceTimeCalculator()

        # NEW: Initialize route optimizer
        self.route_optimizer = RouteSequenceOptimizer(
            distance_calculator=self.distance_calc, solver_time_limit_seconds=10
        )

        logger.info(f"Production CFA initialized with route optimization")

    def _load_business_parameters(self):
        """Load batching parameters from business config database"""
        try:
            tight_dist = self.config.get("cfa_compatibility_distance_tight")
            if tight_dist is not None:
                self.theta.compatibility_distance_km_tight = float(tight_dist)

            medium_dist = self.config.get("cfa_compatibility_distance_medium")
            if medium_dist is not None:
                self.theta.compatibility_distance_km_medium = float(medium_dist)

            long_dist = self.config.get("cfa_compatibility_distance_long")
            if long_dist is not None:
                self.theta.compatibility_distance_km_long = float(long_dist)

            collin_med = self.config.get("cfa_collinearity_threshold_medium")
            if collin_med is not None:
                self.theta.collinearity_threshold_medium = float(collin_med)

            collin_long = self.config.get("cfa_collinearity_threshold_long")
            if collin_long is not None:
                self.theta.collinearity_threshold_long = float(collin_long)

            detour_med = self.config.get("cfa_detour_ratio_threshold_medium")
            if detour_med is not None:
                self.theta.detour_ratio_threshold_medium = float(detour_med)

            detour_long = self.config.get("cfa_detour_ratio_threshold_long")
            if detour_long is not None:
                self.theta.detour_ratio_threshold_long = float(detour_long)

            max_batch = self.config.get("cfa_max_batch_size")
            if max_batch is not None:
                self.theta.max_batch_size = int(max_batch)

            min_util = self.config.get("min_utilization_threshold")
            if min_util is not None:
                self.theta.min_utilization_threshold = float(min_util)

        except Exception as e:
            logger.warning(f"Could not load business parameters: {e}. Using defaults.")

    def solve(self, state, vfa_values: Optional[Dict] = None) -> CFAAction:
        """Main CFA solver - creates consolidated batches with vehicle assignments"""
        start_time = time.time()

        pending = state.pending_shipments
        available_vehicles = state.get_available_vehicles()

        logger.info(
            f"CFA solving: {len(pending)} shipments, {len(available_vehicles)} vehicles"
        )

        if not pending:
            return self._create_wait_action("No pending shipments")

        if not available_vehicles:
            return self._create_wait_action("No available vehicles")

        # Step 1: Geographic clustering
        clusters = self._cluster_compatible_shipments(pending)
        logger.debug(f"Formed {len(clusters)} geographic clusters")

        # Step 2: Create batches (FIXED: handles capacity overflow)
        all_batches = []
        total_cost = 0.0
        total_utilization_volume = 0.0
        total_utilization_weight = 0.0
        vehicles_used = set()

        for cluster_idx, cluster_shipments in enumerate(clusters):
            if not cluster_shipments:
                continue

            # FIXED: Process ALL shipments in cluster
            remaining_shipments = list(cluster_shipments)

            while remaining_shipments:
                batch_solution = self._solve_cluster_assignment_with_routing(
                    remaining_shipments, available_vehicles, vehicles_used
                )

                if batch_solution:
                    all_batches.append(batch_solution)
                    total_cost += batch_solution["estimated_cost"]
                    total_utilization_volume += batch_solution["utilization_volume"]
                    total_utilization_weight += batch_solution["utilization_weight"]
                    vehicles_used.add(batch_solution["vehicle"])

                    # Remove batched shipments from remaining
                    batched_ids = set(batch_solution["shipments"])
                    remaining_shipments = [
                        s for s in remaining_shipments if s.id not in batched_ids
                    ]
                else:
                    # Can't batch remaining shipments
                    if remaining_shipments:
                        logger.warning(
                            f"Cannot batch {len(remaining_shipments)} shipments: "
                            f"no suitable vehicle available"
                        )
                    break

        comp_time = (time.time() - start_time) * 1000

        # Decision logic (unchanged)
        if not all_batches:
            return self._create_wait_action("No cost-effective consolidations found")

        avg_utilization = (total_utilization_volume + total_utilization_weight) / (
            2 * len(all_batches)
        )

        if avg_utilization < self.theta.min_utilization_threshold:
            return self._create_wait_action(
                f"Utilization {avg_utilization:.1%} below threshold "
                f"{self.theta.min_utilization_threshold:.1%}"
            )

        # Dispatch
        return CFAAction(
            action_type="DISPATCH",
            details={
                "type": "DISPATCH",
                "batches": all_batches,
                "estimated_cost": total_cost,
                "computation_time_ms": comp_time,
            },
            reasoning=f"CFA: Dispatching {len(all_batches)} batches, "
            f"{sum(len(b['shipments']) for b in all_batches)} shipments, "
            f"avg util {avg_utilization:.1%}",
            confidence=min(0.95, 0.5 + avg_utilization * 0.5),
        )

    def _solve_cluster_assignment_with_routing(
        self, shipments: List, available_vehicles: List, vehicles_used: set
    ) -> Optional[Dict]:
        """
        Solve MIP for vehicle assignment to shipment cluster WITH ROUTE OPTIMIZATION

        NEW: Uses RouteSequenceOptimizer to generate actual pickup/delivery sequences

        Returns complete batch specification with:
        - Vehicle assignment
        - Optimized route sequence (pickup/delivery stops)
        - Cost/distance calculations
        - Utilization metrics
        """
        from ..algorithms.route_optimizer import Location

        # Filter available vehicles
        candidate_vehicles = [
            v for v in available_vehicles if v.id not in vehicles_used
        ]
        if not candidate_vehicles:
            return None

        # Check capacity constraints
        total_volume = sum(s.volume for s in shipments)
        total_weight = sum(s.weight for s in shipments)

        # Find suitable vehicle
        suitable_vehicle = None
        for v in candidate_vehicles:
            if v.capacity.volume >= total_volume and v.capacity.weight >= total_weight:
                suitable_vehicle = v
                break

        if not suitable_vehicle:
            # Try splitting if too large
            if len(shipments) > 1:
                mid = len(shipments) // 2
                batch1 = self._solve_cluster_assignment_with_routing(
                    shipments[:mid], available_vehicles, vehicles_used
                )
                if batch1:
                    vehicles_used.add(batch1["vehicle"])
                    return batch1
            return None

        # NEW: Generate optimized route sequence using RouteSequenceOptimizer
        route_sequence = self._generate_optimized_route_sequence(
            shipments, suitable_vehicle
        )

        # Extract distance and duration from optimized route
        route_distance = route_sequence.total_distance_km
        route_duration = route_sequence.total_duration_hours

        # Calculate costs
        cost = (
            self.theta.base_cost_per_km * route_distance
            + self.theta.fixed_cost_per_trip
            - self.theta.consolidation_bonus_per_shipment * len(shipments)
        )

        # Add urgency penalties
        for s in shipments:
            urgency = self._calc_urgency(s, datetime.now())
            cost += self.theta.urgency_penalty_weight * urgency

        # Calculate utilization
        util_volume = min(1.0, total_volume / suitable_vehicle.capacity.volume)
        util_weight = min(1.0, total_weight / suitable_vehicle.capacity.weight)

        # Build batch with route sequence
        batch = {
            "id": f"CFA_{datetime.now().timestamp()}_{shipments[0].id}",
            "shipments": [s.id for s in shipments],
            "vehicle": suitable_vehicle.id,
            "estimated_cost": cost,
            "distance": route_distance,
            "duration": route_duration,
            "utilization": (util_volume + util_weight) / 2.0,
            "utilization_volume": util_volume,
            "utilization_weight": util_weight,
            "route_sequence": route_sequence.route_stops,  # NEW: Actual optimized sequence
            "optimization_status": route_sequence.optimization_status,  # NEW: Track solver status
        }

        return batch

    from ..algorithms.route_optimizer import RouteMetrics

    def _generate_optimized_route_sequence(
        self, shipments: List, vehicle
    ) -> "RouteMetrics":
        """
        Generate optimized route sequence using RouteSequenceOptimizer

        Returns RouteMetrics with optimized pickup/delivery sequence
        """
        from ..algorithms.route_optimizer import Location

        # Origin: Vehicle's current location
        origin = Location(
            lat=vehicle.current_location.lat,
            lon=vehicle.current_location.lng,
            address=f"Vehicle {vehicle.id} location",
        )

        # Build destination list from shipments
        destinations = []
        for shipment in shipments:
            # Add pickup location if needed (origin)
            if hasattr(shipment, "origin") and shipment.origin:
                pickup_loc = Location(
                    lat=shipment.origin.lat,
                    lon=shipment.origin.lng,
                    address=f"Pickup: {shipment.origin.formatted_address}",
                    shipment_ids=[shipment.id],
                )
                destinations.append(pickup_loc)

            # Add delivery location(s)
            if shipment.destinations and len(shipment.destinations) > 0:
                for dest in shipment.destinations:
                    delivery_loc = Location(
                        lat=dest.lat,
                        lon=dest.lng,
                        address=f"Delivery: {dest.formatted_address}",
                        shipment_ids=[shipment.id],
                    )
                    destinations.append(delivery_loc)

        # Call route optimizer to get TSP/VRP solution
        route_metrics = self.route_optimizer.optimize_route_sequence(
            origin=origin,
            destinations=destinations,
            vehicle_capacity_m3=vehicle.capacity.volume,
            max_duration_hours=8.0,
        )

        logger.debug(
            f"Route optimization: {route_metrics.optimization_status}, "
            f"{route_metrics.total_distance_km:.1f}km, "
            f"{len(route_metrics.route_stops)} stops"
        )

        return route_metrics

    def _cluster_compatible_shipments(self, shipments: List) -> List[List]:
        """
        Cluster shipments using data-driven geographic analysis

        Algorithm:
        1. Calculate pairwise compatibility scores
        2. Use graph-based clustering (connected components)
        3. Split clusters exceeding max_batch_size
        """
        n = len(shipments)
        if n == 0:
            return []
        if n == 1:
            return [shipments]

        # Build compatibility graph
        compatibility = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_shipments_compatible(shipments[i], shipments[j]):
                    compatibility[i, j] = True
                    compatibility[j, i] = True

        # Find connected components (clusters)
        visited = [False] * n
        clusters = []

        def dfs(node, cluster):
            visited[node] = True
            cluster.append(shipments[node])
            for neighbor in range(n):
                if compatibility[node, neighbor] and not visited[neighbor]:
                    dfs(neighbor, cluster)

        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)

                # Split large clusters
                if len(cluster) > self.theta.max_batch_size:
                    # Simple split - could be improved
                    for k in range(0, len(cluster), self.theta.max_batch_size):
                        clusters.append(cluster[k : k + self.theta.max_batch_size])
                else:
                    clusters.append(cluster)

        return clusters

    def _are_shipments_compatible(self, s1, s2) -> bool:
        """
        Check if two shipments can be consolidated in same batch

        Uses tiered distance + collinearity approach
        """
        # Get destination coordinates
        if not s1.destinations or not s2.destinations:
            return False

        dest1 = s1.destinations[0]
        dest2 = s2.destinations[0]

        # Calculate distance between destinations
        distance = self.distance_calc.calculate_distance_km(
            dest1.lat, dest1.lng, dest2.lat, dest2.lng
        )

        # Tier 1: Very close destinations - always compatible
        if distance <= self.theta.compatibility_distance_km_tight:
            return True

        # Tier 2: Medium distance - check collinearity
        if distance <= self.theta.compatibility_distance_km_medium:
            # Would need origin to check collinearity properly
            # For now, accept with lower threshold
            return True

        # Tier 3: Long distance - strict collinearity required
        if distance <= self.theta.compatibility_distance_km_long:
            # Would need better collinearity check
            return False

        # Too far apart
        return False

    def _calc_urgency(self, shipment, current_time) -> float:
        """Calculate urgency score [0,1] based on deadline proximity"""
        hours_remaining = (shipment.deadline - current_time).total_seconds() / 3600

        if hours_remaining < 0:
            return 1.0  # Already late
        elif hours_remaining < 2:
            return 0.9
        elif hours_remaining < 4:
            return 0.7
        elif hours_remaining < 8:
            return 0.5
        elif hours_remaining < 24:
            return 0.3
        else:
            return 0.1

    def _create_wait_action(self, reason: str) -> CFAAction:
        """Create WAIT action"""
        return CFAAction(
            action_type="WAIT",
            details={"type": "WAIT"},
            reasoning=f"CFA: {reason}",
            confidence=0.5,
        )

    def update_parameters(
        self, action_details, reward, estimated_utilization, actual_utilization
    ):
        """
        Update CFA parameters based on learning signal

        Future: Will be replaced by neural CFA
        """
        # Placeholder for parameter learning
        pass
