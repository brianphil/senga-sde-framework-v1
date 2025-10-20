# src/core/pfa.py

"""
Policy Function Approximation (PFA): Simple rule-based decisions
Fast, deterministic, transparent - for simple states and emergencies
"""

from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .state_manager import (
    StateManager, SystemState, Shipment, VehicleState,
    ShipmentStatus, VehicleStatus
)
from ..config.senga_config import SengaConfigurator

@dataclass
class PFAAction:
    """Action output from PFA"""
    action_type: str  # 'DISPATCH_IMMEDIATE', 'DEFER_TO_CFA', 'WAIT', 'EMERGENCY_*'
    shipments: List[Shipment]
    vehicle: Optional[VehicleState]
    reasoning: str
    confidence: float  # How confident PFA is (0-1, 1.0 = very confident)

class PolicyFunctionApproximator:
    """
    PFA: Fast rule-based decisions for simple states
    
    Rule Priority Order:
    1. Emergency dispatch (deadline imminent)
    2. Emergency no vehicle available (alert)
    3. Emergency capacity exceeded (alert)
    4. Simple single-shipment single-vehicle match
    5. Defer to CFA for complex scenarios
    6. Wait for consolidation
    
    Mathematical Foundation:
    π_PFA: S → A is a deterministic policy based on if-then rules
    Each rule has clear triggering conditions and outputs
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
    
    def decide(self, state: SystemState) -> PFAAction:
        """
        Main decision logic - evaluates rules in priority order
        
        Args:
            state: Current system state S_t
            
        Returns:
            PFAAction with decision or defers to CFA
        """
        
        # Rule 1: Emergency dispatch check (highest priority)
        emergency_action = self._check_emergency_dispatch(state)
        if emergency_action:
            return emergency_action
        
        # Rule 2: Simple dispatch check
        simple_action = self._check_simple_dispatch(state)
        if simple_action:
            return simple_action
        
        # Rule 3: Check if state is too complex for PFA
        if self._is_complex_state(state):
            return PFAAction(
                action_type='DEFER_TO_CFA',
                shipments=[],
                vehicle=None,
                reasoning=f"Complex state: {len(state.pending_shipments)} pending shipments, "
                         f"{len(state.get_available_vehicles())} available vehicles - requires CFA optimization",
                confidence=1.0
            )
        
        # Rule 4: Check if waiting is beneficial
        wait_decision = self._evaluate_wait(state)
        return wait_decision
    
    def _check_emergency_dispatch(self, state: SystemState) -> Optional[PFAAction]:
        """
        Rule 1: Emergency dispatch for shipments approaching deadline
        
        Mathematical Condition:
        IF ∃ shipment s: time_to_deadline(s) < threshold
        AND shipment not in active route
        THEN dispatch immediately with available vehicle
        
        Triggers when:
        - Shipment has < emergency_threshold_hours until deadline
        - No active route covers this shipment
        - Available vehicle exists
        """
        emergency_threshold = self.config.emergency_threshold_hours
        urgent_shipments = state.get_urgent_shipments(emergency_threshold)
        
        if not urgent_shipments:
            return None
        
        # Sort by urgency (most urgent first)
        urgent_shipments.sort(
            key=lambda s: s.time_to_deadline(state.timestamp)
        )
        
        most_urgent = urgent_shipments[0]
        time_remaining = most_urgent.time_to_deadline(state.timestamp)
        
        # Check if already being handled by an active route
        if self._shipment_covered_by_active_route(most_urgent, state):
            return None
        
        # Find available vehicles
        available_vehicles = state.get_available_vehicles()
        
        if not available_vehicles:
            # CRITICAL: No vehicle available for emergency
            return PFAAction(
                action_type='EMERGENCY_NO_VEHICLE',
                shipments=[most_urgent],
                vehicle=None,
                reasoning=f"CRITICAL EMERGENCY: Shipment {most_urgent.id} has only "
                         f"{time_remaining.total_seconds()/3600:.1f} hours until deadline but "
                         f"NO VEHICLE AVAILABLE! Customer: {most_urgent.customer_id}",
                confidence=1.0
            )
        
        # Find best vehicle (closest with capacity)
        best_vehicle = self._select_best_vehicle(most_urgent, available_vehicles)
        
        if not best_vehicle:
            # No vehicle has capacity
            return PFAAction(
                action_type='EMERGENCY_NO_CAPACITY',
                shipments=[most_urgent],
                vehicle=None,
                reasoning=f"CRITICAL EMERGENCY: Shipment {most_urgent.id} "
                         f"(volume={most_urgent.volume}m³, weight={most_urgent.weight}kg) "
                         f"exceeds ALL available vehicle capacities!",
                confidence=1.0
            )
        
        # Can dispatch - emergency action
        return PFAAction(
            action_type='DISPATCH_IMMEDIATE',
            shipments=[most_urgent],
            vehicle=best_vehicle,
            reasoning=f"EMERGENCY DISPATCH: Shipment {most_urgent.id} has "
                     f"{time_remaining.total_seconds()/3600:.1f}h until deadline. "
                     f"Dispatching with vehicle {best_vehicle.id}",
            confidence=1.0
        )
    
    def _check_simple_dispatch(self, state: SystemState) -> Optional[PFAAction]:
        """
        Rule 2: Simple dispatch when obvious match exists
        
        Mathematical Condition:
        IF |pending_shipments| = 1 AND |available_vehicles| = 1
        AND vehicle_can_handle(vehicle, shipment)
        AND (utilization(shipment, vehicle) ≥ threshold OR time_pressure > 0.5)
        THEN dispatch
        
        Triggers when:
        - Exactly one shipment waiting
        - Exactly one vehicle idle
        - Vehicle has capacity
        - Either utilization is good OR time pressure is building
        """
        pending = state.pending_shipments
        available_vehicles = state.get_available_vehicles()
        
        # Not a simple case if multiple shipments or vehicles
        if len(pending) != 1 or len(available_vehicles) != 1:
            return None
        
        shipment = pending[0]
        vehicle = available_vehicles[0]
        
        # Check capacity
        if not self._vehicle_can_handle(vehicle, [shipment]):
            return PFAAction(
                action_type='DEFER_TO_CFA',
                shipments=[],
                vehicle=None,
                reasoning=f"Single shipment exceeds single vehicle capacity - need CFA",
                confidence=0.8
            )
        
        # Calculate utilization
        volume_util = shipment.volume / vehicle.capacity.volume
        weight_util = shipment.weight / vehicle.capacity.weight
        utilization = max(volume_util, weight_util)  # Limiting factor
        
        min_utilization = self.config.min_utilization
        time_pressure = shipment.time_pressure(state.timestamp)
        
        # Decision logic
        if utilization >= min_utilization:
            # Good utilization - dispatch
            return PFAAction(
                action_type='DISPATCH_IMMEDIATE',
                shipments=[shipment],
                vehicle=vehicle,
                reasoning=f"Simple dispatch: utilization={utilization:.2%} "
                         f"≥ threshold={min_utilization:.2%}",
                confidence=0.95
            )
        
        elif time_pressure > 0.5:
            # Moderate time pressure - dispatch even with lower utilization
            return PFAAction(
                action_type='DISPATCH_IMMEDIATE',
                shipments=[shipment],
                vehicle=vehicle,
                reasoning=f"Simple dispatch: time_pressure={time_pressure:.2f} > 0.5, "
                         f"utilization={utilization:.2%}",
                confidence=0.85
            )
        
        # Low utilization and low time pressure - should wait
        return None
    
    def _is_complex_state(self, state: SystemState) -> bool:
        """
        Determine if state is too complex for PFA
        
        Mathematical Condition:
        complexity(S) > threshold where:
        complexity = f(|pending|, |vehicles|, spatial_clustering, time_variance)
        
        Complex when:
        - Multiple shipments with consolidation opportunities
        - Multiple vehicles requiring optimization
        - Spatial clustering suggests batch potential
        - Varying time pressures across shipments
        """
        num_pending = len(state.pending_shipments)
        num_available = len(state.get_available_vehicles())
        
        # Thresholds from config
        max_simple_pending = self.config.get(
            'dla.complexity_assessment.low_complexity_threshold.max_pending_shipments', 
            5
        )
        
        # Too many shipments for simple rules
        if num_pending > max_simple_pending:
            return True
        
        # Too many vehicles for simple matching
        if num_available > 3:
            return True
        
        # Check for consolidation potential
        if num_pending >= 2:
            if self._has_consolidation_potential(state.pending_shipments):
                return True
        
        # Check for time pressure variance (some urgent, some not)
        if num_pending >= 2:
            time_pressures = [s.time_pressure(state.timestamp) for s in state.pending_shipments]
            pressure_variance = max(time_pressures) - min(time_pressures)
            if pressure_variance > 0.3:  # High variance
                return True
        
        return False
    
    def _evaluate_wait(self, state: SystemState) -> PFAAction:
        """
        Rule 4: Evaluate if waiting is the best action
        
        Default fallback when no other rule triggers
        """
        pending = state.pending_shipments
        
        if not pending:
            return PFAAction(
                action_type='WAIT',
                shipments=[],
                vehicle=None,
                reasoning="No pending shipments",
                confidence=1.0
            )
        
        # Calculate consolidation window
        max_wait_hours = self.config.get('max_consolidation_wait_hours', 24)
        
        if pending:
            min_time_to_deadline = min(
                s.time_to_deadline(state.timestamp).total_seconds() / 3600
                for s in pending
            )
            
            if min_time_to_deadline > max_wait_hours:
                # Plenty of time - wait for more shipments
                return PFAAction(
                    action_type='WAIT',
                    shipments=[],
                    vehicle=None,
                    reasoning=f"Waiting for consolidation: {min_time_to_deadline:.1f}h "
                             f"until next deadline (max wait: {max_wait_hours}h)",
                    confidence=0.7
                )
            else:
                # Time getting tight but not emergency - defer to CFA
                return PFAAction(
                    action_type='DEFER_TO_CFA',
                    shipments=[],
                    vehicle=None,
                    reasoning=f"Time constraint approaching: {min_time_to_deadline:.1f}h "
                             f"until deadline - need CFA optimization",
                    confidence=0.8
                )
        
        return PFAAction(
            action_type='WAIT',
            shipments=[],
            vehicle=None,
            reasoning="Waiting for better consolidation opportunities",
            confidence=0.6
        )
    
    def _shipment_covered_by_active_route(self, shipment: Shipment, 
                                          state: SystemState) -> bool:
        """
        Check if shipment is already in an active route
        
        Args:
            shipment: Shipment to check
            state: Current system state
            
        Returns:
            True if shipment is already being handled
        """
        for route in state.active_routes:
            if shipment.id in route.shipment_ids:
                return True
        return False
    
    def _select_best_vehicle(self, shipment: Shipment,
                            vehicles: List[VehicleState]) -> Optional[VehicleState]:
        """
        Select best vehicle for a shipment
        
        Criteria (in order):
        1. Has sufficient capacity
        2. Closest to shipment origin
        3. Most cost-efficient
        
        Returns:
            Best vehicle or None if no suitable vehicle
        """
        candidates = []
        
        for vehicle in vehicles:
            # Must have capacity
            if not self._vehicle_can_handle(vehicle, [shipment]):
                continue
            
            # Calculate distance to origin
            distance = self._estimate_distance(
                vehicle.current_location, 
                shipment.origin
            )
            
            # Calculate cost score (lower is better)
            # Cost = distance_cost + utilization_penalty
            distance_cost = distance * vehicle.cost_per_km
            
            # Penalty for low utilization
            utilization = shipment.volume / vehicle.capacity.volume
            min_util = self.config.min_utilization
            if utilization < min_util:
                utilization_penalty = (min_util - utilization) * 1000
            else:
                utilization_penalty = 0
            
            total_score = distance_cost + utilization_penalty
            
            candidates.append((vehicle, total_score, distance))
        
        if not candidates:
            return None
        
        # Sort by score (lowest first)
        candidates.sort(key=lambda x: x[1])
        
        return candidates[0][0]
    
    def _vehicle_can_handle(self, vehicle: VehicleState,
                           shipments: List[Shipment]) -> bool:
        """
        Check if vehicle has capacity for shipments
        
        Args:
            vehicle: Vehicle to check
            shipments: List of shipments
            
        Returns:
            True if vehicle can handle all shipments
        """
        total_volume = sum(s.volume for s in shipments)
        total_weight = sum(s.weight for s in shipments)
        
        return (total_volume <= vehicle.capacity.volume and
                total_weight <= vehicle.capacity.weight)
    
    def _vehicle_near_location(self, vehicle: VehicleState, 
                              location, threshold_km: float = 5.0) -> bool:
        """
        Check if vehicle is near a location
        
        Args:
            vehicle: Vehicle to check
            location: Target location
            threshold_km: Distance threshold in km
            
        Returns:
            True if vehicle is within threshold distance
        """
        distance = self._estimate_distance(vehicle.current_location, location)
        return distance <= threshold_km
    
    def _estimate_distance(self, loc1, loc2) -> float:
        """
        Estimate distance between two locations using Haversine formula
        
        Args:
            loc1: First location (Location object)
            loc2: Second location (Location object)
            
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = radians(loc1.lat), radians(loc1.lng)
        lat2, lon2 = radians(loc2.lat), radians(loc2.lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _has_consolidation_potential(self, shipments: List[Shipment]) -> bool:
        """
        Check if shipments have consolidation potential
        
        Mathematical Criterion:
        Potential exists when:
        - Multiple shipments share origin zones
        - Destinations are in similar geographic areas
        - Time windows overlap sufficiently
        
        Args:
            shipments: List of shipments to check
            
        Returns:
            True if consolidation potential exists
        """
        if len(shipments) < 2:
            return False
        
        # Check 1: Origin clustering
        origin_zones = {}
        for s in shipments:
            zone = s.origin.zone_id or 'unknown'
            origin_zones[zone] = origin_zones.get(zone, 0) + 1
        
        # If multiple shipments share origins, good consolidation potential
        max_same_origin = max(origin_zones.values())
        if max_same_origin >= 2:
            return True
        
        # Check 2: Destination clustering
        all_destinations = []
        for s in shipments:
            all_destinations.extend(s.destinations)
        
        if len(all_destinations) >= 2:
            dest_zones = {}
            for dest in all_destinations:
                zone = dest.zone_id or 'unknown'
                dest_zones[zone] = dest_zones.get(zone, 0) + 1
            
            # If destinations cluster, good for consolidation
            max_same_dest = max(dest_zones.values())
            if max_same_dest >= 2:
                return True
        
        # Check 3: Geographic proximity
        # Calculate centroid and check if shipments are clustered
        if len(shipments) >= 2:
            lats = [s.origin.lat for s in shipments]
            lngs = [s.origin.lng for s in shipments]
            
            # Standard deviation of locations (measure of spread)
            import numpy as np
            lat_std = np.std(lats)
            lng_std = np.std(lngs)
            
            # If spread is small (< 0.1 degrees ≈ 11km), good consolidation
            if lat_std < 0.1 and lng_std < 0.1:
                return True
        
        return False
    
    def can_handle_state(self, state: SystemState) -> Tuple[bool, float]:
        """
        Check if PFA can confidently handle this state
        
        Used by meta-controller to decide function class
        
        Args:
            state: Current system state
            
        Returns:
            Tuple of (can_handle: bool, confidence: float)
        """
        # PFA can always try, but confidence varies
        
        urgent_shipments = state.get_urgent_shipments(
            self.config.emergency_threshold_hours
        )
        
        if urgent_shipments:
            # Always handle emergencies with high confidence
            return True, 1.0
        
        num_pending = len(state.pending_shipments)
        num_available = len(state.get_available_vehicles())
        
        if num_pending == 0:
            # Trivial case
            return True, 1.0
        
        if num_pending == 1 and num_available == 1:
            # Simple case - high confidence
            return True, 0.9
        
        if self._is_complex_state(state):
            # Too complex for PFA
            return False, 0.0
        
        # Can try but moderate confidence
        return True, 0.6
    
    def get_emergency_status(self, state: SystemState) -> Dict:
        """
        Get emergency status summary for monitoring
        
        Returns:
            Dictionary with emergency metrics
        """
        emergency_threshold = self.config.emergency_threshold_hours
        urgent = state.get_urgent_shipments(emergency_threshold)
        
        critical = []
        for s in urgent:
            time_remaining = s.time_to_deadline(state.timestamp)
            if time_remaining.total_seconds() < 3600:  # < 1 hour
                critical.append(s)
        
        return {
            'num_urgent': len(urgent),
            'num_critical': len(critical),
            'urgent_shipments': [
                {
                    'id': s.id,
                    'time_remaining_hours': s.time_to_deadline(state.timestamp).total_seconds() / 3600,
                    'customer_id': s.customer_id
                }
                for s in urgent
            ],
            'available_vehicles': len(state.get_available_vehicles())
        }