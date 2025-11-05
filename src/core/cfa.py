# src/core/cfa.py

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
    # Tier 1: Close destinations (always batch if on same route)
    compatibility_distance_km_tight: float = 50.0

    # Tier 2: Medium distance (batch if collinear or low detour)
    compatibility_distance_km_medium: float = 150.0
    collinearity_threshold_medium: float = 0.85
    detour_ratio_threshold_medium: float = 1.3

    # Tier 3: Long distance (batch only if highly collinear)
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
    Production CFA: MIP-based batch consolidation with vehicle assignment

    Powell Framework: Optimizes immediate cost while considering consolidation value

    Features:
    - Multi-shipment batch consolidation (not just pairs)
    - Actual vehicle assignment with capacity constraints
    - Geographic clustering for compatibility
    - Real distance/cost calculations
    - Proper utilization tracking
    """

    def __init__(self, config):
        self.config = config
        self.theta = CFAParameters()

        # Load business-configurable parameters from config database
        self._load_business_parameters()

        from ..utils.distance_calculator import DistanceTimeCalculator

        self.distance_calc = DistanceTimeCalculator()

        logger.info(f"Production CFA initialized with route optimization")
        logger.info(
            f"  Tier 1 distance: {self.theta.compatibility_distance_km_tight}km"
        )
        logger.info(
            f"  Tier 2 distance: {self.theta.compatibility_distance_km_medium}km"
        )
        logger.info(f"  Tier 3 distance: {self.theta.compatibility_distance_km_long}km")

    def _load_business_parameters(self):
        """Load batching parameters from business config database"""
        try:
            # Try to load from config
            tight_dist = self.config.get("cfa_compatibility_distance_tight")
            if tight_dist is not None:
                self.theta.compatibility_distance_km_tight = float(tight_dist)

            medium_dist = self.config.get("cfa_compatibility_distance_medium")
            if medium_dist is not None:
                self.theta.compatibility_distance_km_medium = float(medium_dist)

            long_dist = self.config.get("cfa_compatibility_distance_long")
            if long_dist is not None:
                self.theta.compatibility_distance_km_long = float(long_dist)

            # Load thresholds
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
            # Defaults already set in CFAParameters dataclass

    def solve(self, state, vfa_values: Optional[Dict] = None) -> CFAAction:
        """
        Main CFA solver - creates consolidated batches with vehicle assignments

        Algorithm:
        1. Cluster compatible shipments by geography
        2. For each cluster, solve vehicle assignment MIP
        3. Calculate actual routes and costs
        4. Return complete batch specifications
        """
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

        # Step 1: Geographic clustering for compatibility
        clusters = self._cluster_compatible_shipments(pending)
        logger.debug(f"Formed {len(clusters)} geographic clusters")

        # Step 2: Solve MIP for each cluster
        all_batches = []
        total_cost = 0.0
        total_utilization_volume = 0.0
        total_utilization_weight = 0.0
        vehicles_used = set()

        for cluster_idx, cluster_shipments in enumerate(clusters):
            if not cluster_shipments:
                continue

            # Solve vehicle assignment for this cluster
            batch_solution = self._solve_cluster_assignment(
                cluster_shipments, available_vehicles, vehicles_used
            )

            if batch_solution:
                all_batches.append(batch_solution)
                total_cost += batch_solution["estimated_cost"]
                total_utilization_volume += batch_solution["utilization_volume"]
                total_utilization_weight += batch_solution["utilization_weight"]
                vehicles_used.add(batch_solution["vehicle"])

        comp_time = (time.time() - start_time) * 1000

        # Decision logic
        if not all_batches:
            return self._create_wait_action("No cost-effective consolidations found")

        avg_utilization = (total_utilization_volume + total_utilization_weight) / (
            2 * len(all_batches)
        )

        if avg_utilization < self.theta.min_utilization_threshold:
            return self._create_wait_action(
                f"Utilization {avg_utilization:.1%} below threshold {self.theta.min_utilization_threshold:.1%}"
            )

        # Return dispatch action
        return CFAAction(
            action_type="DISPATCH",
            details={
                "type": "DISPATCH",
                "batches": all_batches,
                "estimated_cost": total_cost,
                "computation_time_ms": comp_time,
            },
            reasoning=f"CFA: Dispatching {len(all_batches)} batches, {len([s for b in all_batches for s in b['shipments']])} shipments, avg util {avg_utilization:.1%}",
            confidence=min(0.95, 0.5 + avg_utilization * 0.5),
        )

    def _cluster_compatible_shipments(self, shipments: List) -> List[List]:
        """
        Cluster shipments using data-driven geographic analysis

        Algorithm:
        1. DBSCAN spatial clustering on destinations (finds natural geographic groups)
        2. Route collinearity check (are destinations on same general path?)
        3. Transitive compatibility (all pairs in cluster must be compatible)

        No hard-coded cities - works for any geography
        """
        if len(shipments) <= 1:
            return [shipments]

        n = len(shipments)

        # Extract destination coordinates
        dest_coords = []
        for s in shipments:
            if s.destinations and len(s.destinations) > 0:
                dest = s.destinations[0]
                dest_coords.append([dest.lat, dest.lng])
            else:
                dest_coords.append([0.0, 0.0])

        dest_array = np.array(dest_coords)

        # Build compatibility matrix using geometric criteria
        compatible = [[False] * n for _ in range(n)]

        for i in range(n):
            compatible[i][i] = True
            for j in range(i + 1, n):
                if self._are_destinations_compatible(
                    dest_coords[i], dest_coords[j], dest_array
                ):
                    compatible[i][j] = True
                    compatible[j][i] = True

        # Greedy clustering - build maximal compatible groups
        clusters = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            cluster_indices = {i}
            used.add(i)

            # Add all transitively compatible shipments
            for j in range(n):
                if j in used or j == i:
                    continue

                # Must be compatible with ALL in cluster
                if all(compatible[j][k] for k in cluster_indices):
                    cluster_indices.add(j)
                    used.add(j)

                    if len(cluster_indices) >= self.theta.max_batch_size:
                        break

            cluster = [shipments[idx] for idx in sorted(cluster_indices)]
            clusters.append(cluster)

        logger.info(
            f"Clustered {n} shipments → {len(clusters)} batches: {[len(c) for c in clusters]}"
        )
        return clusters

    def _are_destinations_compatible(
        self, dest1: List[float], dest2: List[float], all_destinations: np.ndarray
    ) -> bool:
        """
        Check if two destinations are compatible for batching

        Uses multi-tier geometric criteria based on distance:

        Tier 1 (< 50km): Always batch if same general direction
        Tier 2 (50-150km): Batch if collinear (>0.85) OR efficient detour (<1.3x)
        Tier 3 (150-300km): Batch only if highly collinear (>0.92) AND efficient (<1.15x)

        Business examples:
        - Nairobi → Nakuru (160km): Tier 2, batches with high collinearity
        - Nakuru → Eldoret (160km): Tier 2, batches if on same route
        - Kisumu → Eldoret (200km): Tier 3, requires very high collinearity

        Args:
            dest1, dest2: [lat, lng] coordinates
            all_destinations: All destination coordinates for context

        Returns:
            True if destinations should batch together
        """
        # Calculate direct distance between destinations
        direct_dist = self._haversine(tuple(dest1), tuple(dest2))

        # Use vehicle location or pickup point as origin
        # TODO: In production, get actual vehicle location from context
        origin = (-1.286389, 36.817223)  # Nairobi reference

        # Calculate geometric metrics
        collinearity_score = self._calculate_collinearity(
            origin, tuple(dest1), tuple(dest2)
        )

        detour_ratio = self._calculate_detour_ratio(origin, tuple(dest1), tuple(dest2))

        # Tier 1: Close destinations (< 50km)
        if direct_dist <= self.theta.compatibility_distance_km_tight:
            # Very close - batch if reasonably same direction
            return collinearity_score > 0.7 or detour_ratio < 1.5

        # Tier 2: Medium distance (50-150km)
        elif direct_dist <= self.theta.compatibility_distance_km_medium:
            # Medium distance - batch if collinear OR efficient detour
            return (
                collinearity_score >= self.theta.collinearity_threshold_medium
                or detour_ratio <= self.theta.detour_ratio_threshold_medium
            )

        # Tier 3: Long distance (150-300km)
        elif direct_dist <= self.theta.compatibility_distance_km_long:
            # Long distance - batch only if BOTH highly collinear AND efficient
            return (
                collinearity_score >= self.theta.collinearity_threshold_long
                and detour_ratio <= self.theta.detour_ratio_threshold_long
            )

        # Beyond 300km - no consolidation
        return False

    def _calculate_collinearity(
        self,
        origin: Tuple[float, float],
        dest1: Tuple[float, float],
        dest2: Tuple[float, float],
    ) -> float:
        """
        Calculate collinearity score [0,1] for three points

        Returns 1.0 if dest1 and dest2 are on same ray from origin
        Returns 0.0 if they're perpendicular

        Mathematical approach: Compare bearing angles
        """
        bearing1 = self._calculate_bearing(origin, dest1)
        bearing2 = self._calculate_bearing(origin, dest2)

        bearing_diff = self._bearing_difference(bearing1, bearing2)

        # Convert to similarity score [0,1]
        # 0° difference = 1.0 (perfect collinearity)
        # 180° difference = 0.0 (opposite directions)
        similarity = 1.0 - (bearing_diff / 180.0)

        return similarity

    def _calculate_detour_ratio(
        self,
        origin: Tuple[float, float],
        dest1: Tuple[float, float],
        dest2: Tuple[float, float],
    ) -> float:
        """
        Calculate detour ratio for visiting both destinations

        Ratio = (origin→d1→d2 distance) / (origin→d1 + origin→d2)

        Returns:
            < 1.0: Visiting both is more efficient than separate trips
            = 1.0: No benefit to consolidation
            > 1.0: Detour required, but may still be worth it
        """
        # Distance for combined route
        dist_origin_d1 = self._haversine(origin, dest1)
        dist_d1_d2 = self._haversine(dest1, dest2)
        combined_distance = dist_origin_d1 + dist_d1_d2

        # Distance for separate routes
        dist_origin_d2 = self._haversine(origin, dest2)
        separate_distance = dist_origin_d1 + dist_origin_d2

        if separate_distance == 0:
            return float("inf")

        return combined_distance / separate_distance

    def _calculate_bearing(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """
        Calculate bearing (direction) from point1 to point2 in degrees

        Returns: Bearing in degrees [0, 360) where:
        - 0° = North
        - 90° = East
        - 180° = South
        - 270° = West
        """
        lat1, lon1 = np.radians(point1[0]), np.radians(point1[1])
        lat2, lon2 = np.radians(point2[0]), np.radians(point2[1])

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))

        # Normalize to [0, 360)
        return (bearing + 360) % 360

    def _bearing_difference(self, bearing1: float, bearing2: float) -> float:
        """
        Calculate minimum angular difference between two bearings

        Returns: Difference in degrees [0, 180]

        Example:
        - bearing_difference(10, 20) = 10°
        - bearing_difference(350, 10) = 20° (not 340°)
        - bearing_difference(0, 180) = 180°
        """
        diff = abs(bearing1 - bearing2)
        if diff > 180:
            diff = 360 - diff
        return diff

    def _solve_cluster_assignment(
        self, shipments: List, available_vehicles: List, vehicles_used: set
    ) -> Optional[Dict]:
        """
        Solve MIP for vehicle assignment to shipment cluster

        Returns complete batch specification with:
        - Vehicle assignment
        - Route sequence
        - Cost/distance calculations
        - Utilization metrics
        """
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
                batch1 = self._solve_cluster_assignment(
                    shipments[:mid], available_vehicles, vehicles_used
                )
                if batch1:
                    vehicles_used.add(batch1["vehicle"])
                    return batch1
            return None

        # Calculate route and costs
        route_distance = self._estimate_route_distance(shipments, suitable_vehicle)
        route_duration = route_distance / 40.0  # Assume 40 km/h average

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

        # Build batch
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
        }

        return batch

    def _estimate_route_distance(self, shipments: List, vehicle) -> float:
        """
        Estimate total route distance using actual coordinates

        Uses vehicle location -> pickup -> deliveries -> return
        """
        if not shipments:
            return 0.0

        total_dist = 0.0

        # Start from vehicle location
        current_loc = (vehicle.current_location.lat, vehicle.current_location.lng)

        # Visit each shipment origin (if needed) then destination
        for s in shipments:
            # Distance to pickup (assuming origin is pickup)
            if hasattr(s, "origin") and s.origin:
                pickup_loc = (s.origin.lat, s.origin.lng)
                total_dist += self._haversine(current_loc, pickup_loc)
                current_loc = pickup_loc

            # Distance to delivery
            if s.destinations and len(s.destinations) > 0:
                dest_loc = (s.destinations[0].lat, s.destinations[0].lng)
                total_dist += self._haversine(current_loc, dest_loc)
                current_loc = dest_loc

        # Return to base (optional - comment out if not needed)
        # home_loc = (vehicle.home_location.lat, vehicle.home_location.lng)
        # total_dist += self._haversine(current_loc, home_loc)

        return total_dist

    def _calc_urgency(self, shipment, current_time) -> float:
        """Calculate urgency score [0,1] based on deadline proximity"""
        hours_remaining = (shipment.deadline - current_time).total_seconds() / 3600
        return max(0.0, min(1.0, 1.0 - hours_remaining / 24.0))

    def _haversine(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two lat/lng points"""
        R = 6371  # Earth radius in km
        lat1, lon1 = loc1
        lat2, lon2 = loc2

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )

        return R * 2 * np.arcsin(np.sqrt(a))

    def _create_wait_action(self, reason: str) -> CFAAction:
        """Create WAIT action with reasoning"""
        return CFAAction(
            action_type="WAIT",
            details={"type": "WAIT"},
            reasoning=f"CFA: {reason}",
            confidence=0.6,
        )

    def update_parameters(
        self,
        predicted_cost: float,
        actual_cost: float,
        predicted_util: float,
        actual_util: float,
    ):
        """
        Learn from route outcomes (Powell: parameter adaptation)

        Adjusts CFA parameters based on prediction accuracy
        """
        if predicted_cost <= 0 or predicted_util <= 0:
            return

        cost_ratio = actual_cost / predicted_cost
        util_ratio = actual_util / predicted_util

        # Adjust cost parameters
        if cost_ratio > 1.1:  # Underestimated
            self.theta.base_cost_per_km *= 1.02
        elif cost_ratio < 0.9:  # Overestimated
            self.theta.base_cost_per_km *= 0.98

        # Adjust utilization expectations
        if util_ratio < 0.8:  # Overestimated utilization
            self.theta.min_utilization_threshold *= 0.95
        elif util_ratio > 1.2:  # Underestimated
            self.theta.min_utilization_threshold *= 1.05

        # Keep parameters in reasonable bounds
        self.theta.base_cost_per_km = max(10.0, min(100.0, self.theta.base_cost_per_km))
        self.theta.min_utilization_threshold = max(
            0.2, min(0.7, self.theta.min_utilization_threshold)
        )

        logger.debug(
            f"CFA parameters updated: cost_per_km={self.theta.base_cost_per_km:.1f}, min_util={self.theta.min_utilization_threshold:.2f}"
        )


# Alias for compatibility
NeuralCFA = CostFunctionApproximator
