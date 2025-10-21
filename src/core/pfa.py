# src/core/pfa.py

"""
Policy Function Approximation (PFA): Simple rule-based decisions
Fast, deterministic, transparent - for simple states and emergencies
UPDATED: Fixed to prevent premature dispatch of standard orders
"""

from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from .state_manager import (
    StateManager, SystemState, Shipment, VehicleState,
    ShipmentStatus, VehicleStatus, Location
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
    1. Emergency dispatch (deadline imminent < 2 hours)
    2. Simple high-utilization dispatch (single order, ≥75% utilization)
    3. Defer to CFA for consolidation opportunities
    4. Wait for better consolidation
    
    Mathematical Foundation:
    π_PFA: S → A is a deterministic policy based on if-then rules
    Each rule has clear triggering conditions and outputs
    
    CRITICAL FIX: Standard orders (priority=1.0) should NOT dispatch immediately
    Only emergency/urgent (priority≥2.0) OR high utilization (≥75%) trigger dispatch
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
        
        # Rule 2: Simple high-utilization dispatch check
        simple_action = self._check_simple_high_util_dispatch(state)
        if simple_action:
            return simple_action
        
        # Rule 3: Check if state needs CFA optimization
        if self._should_use_cfa(state):
            return PFAAction(
                action_type='DEFER_TO_CFA',
                shipments=[],
                vehicle=None,
                reasoning=self._get_cfa_deferral_reason(state),
                confidence=1.0
            )
        
        # Rule 4: Default to waiting for consolidation
        return self._evaluate_wait(state)
    
    def _check_emergency_dispatch(self, state: SystemState) -> Optional[PFAAction]:
        """
        Rule 1: Emergency dispatch ONLY for genuinely urgent shipments
        
        Mathematical Condition:
        IF ∃ shipment s WHERE:
           - priority ≥ 2.0 (urgent or emergency) AND
           - time_to_deadline(s) < emergency_threshold_hours (default 2h)
        THEN dispatch immediately
        
        CRITICAL: Standard orders (priority=1.0) NEVER trigger this rule
        """
        emergency_threshold = self.config.get('emergency_threshold_hours', 2.0)
        
        # Get shipments approaching deadline
        urgent_shipments = state.get_urgent_shipments(emergency_threshold)
        
        # Filter: ONLY dispatch if priority indicates emergency/urgent
        genuine_emergencies = [
            s for s in urgent_shipments 
            if s.priority >= 2.0  # Must be urgent (2.0) or emergency (3.0)
        ]
        
        if not genuine_emergencies:
            return None
        
        # Sort by urgency (most urgent first)
        genuine_emergencies.sort(
            key=lambda s: s.time_to_deadline(state.timestamp)
        )
        
        most_urgent = genuine_emergencies[0]
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
                reasoning=f"CRITICAL EMERGENCY: {most_urgent.priority_name()} shipment {most_urgent.id} "
                         f"has only {time_remaining.total_seconds()/3600:.1f}h until deadline but "
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
                reasoning=f"CRITICAL EMERGENCY: {most_urgent.priority_name()} shipment {most_urgent.id} "
                         f"(volume={most_urgent.volume:.2f}m³, weight={most_urgent.weight:.1f}kg) "
                         f"exceeds ALL available vehicle capacities!",
                confidence=1.0
            )
        
        # Can dispatch - emergency action
        return PFAAction(
            action_type='DISPATCH_IMMEDIATE',
            shipments=[most_urgent],
            vehicle=best_vehicle,
            reasoning=f"EMERGENCY DISPATCH: {most_urgent.priority_name()} shipment {most_urgent.id} "
                     f"has only {time_remaining.total_seconds()/3600:.1f}h until deadline. "
                     f"Dispatching immediately with vehicle {best_vehicle.id}",
            confidence=1.0
        )
    
    def _check_simple_high_util_dispatch(self, state: SystemState) -> Optional[PFAAction]:
        """
        Rule 2: Simple dispatch ONLY when high utilization achieved
        
        Mathematical Condition:
        IF |pending_shipments| = 1 AND |available_vehicles| ≥ 1
        AND vehicle_can_handle(vehicle, shipment)
        AND utilization(shipment, vehicle) ≥ min_utilization_threshold (75%)
        THEN dispatch
        
        CRITICAL CHANGE: Removed time_pressure bypass
        Standard orders must achieve ≥75% utilization OR wait for consolidation
        """
        pending = state.pending_shipments
        available_vehicles = state.get_available_vehicles()
        
        # Only consider if exactly one shipment
        if len(pending) != 1:
            return None
        
        if not available_vehicles:
            return None
        
        shipment = pending[0]
        
        # Find best vehicle for this shipment
        best_vehicle = self._select_best_vehicle(shipment, available_vehicles)
        
        if not best_vehicle:
            return PFAAction(
                action_type='DEFER_TO_CFA',
                shipments=[],
                vehicle=None,
                reasoning=f"Single shipment exceeds available vehicle capacities - need CFA",
                confidence=0.8
            )
        
        # Calculate utilization
        volume_util = shipment.volume / best_vehicle.capacity.volume
        weight_util = shipment.weight / best_vehicle.capacity.weight
        utilization = max(volume_util, weight_util)  # Limiting factor
        
        min_utilization = self.config.get('min_utilization_threshold', 0.75)
        
        # CRITICAL: Only dispatch if utilization meets threshold
        if utilization >= min_utilization:
            return PFAAction(
                action_type='DISPATCH_IMMEDIATE',
                shipments=[shipment],
                vehicle=best_vehicle,
                reasoning=f"High utilization dispatch: {utilization:.1%} ≥ {min_utilization:.1%} threshold. "
                         f"Single order achieves good vehicle utilization.",
                confidence=0.95
            )
        
        # Low utilization - should wait for consolidation
        # Do NOT dispatch - return None to trigger CFA/WAIT evaluation
        return None
    
    def _should_use_cfa(self, state: SystemState) -> bool:
        """
        Determine if CFA optimization is needed
        
        CFA should be used when:
        1. Multiple shipments with potential geographical clustering
        2. Multiple vehicles requiring assignment optimization
        3. Consolidation opportunities exist
        4. Time constraints allow for optimization
        
        Returns True if state requires CFA optimization
        """
        num_pending = len(state.pending_shipments)
        num_available = len(state.get_available_vehicles())
        
        # No pending = no need for CFA
        if num_pending == 0:
            return False
        
        # Multiple shipments = potential consolidation → use CFA
        if num_pending >= 2:
            return True
        
        # Single shipment with low utilization = might wait for consolidation
        if num_pending == 1 and num_available > 0:
            shipment = state.pending_shipments[0]
            vehicles = state.get_available_vehicles()
            
            # Check if any vehicle achieves good utilization
            for vehicle in vehicles:
                if self._vehicle_can_handle(vehicle, [shipment]):
                    volume_util = shipment.volume / vehicle.capacity.volume
                    weight_util = shipment.weight / vehicle.capacity.weight
                    utilization = max(volume_util, weight_util)
                    
                    min_util = self.config.get('min_utilization_threshold', 0.75)
                    if utilization >= min_util:
                        # High utilization possible - not complex
                        return False
            
            # Low utilization - should consider waiting → use CFA to evaluate
            return True
        
        return False
    
    def _get_cfa_deferral_reason(self, state: SystemState) -> str:
        """Generate reasoning for why we're deferring to CFA"""
        num_pending = len(state.pending_shipments)
        num_available = len(state.get_available_vehicles())
        
        reasons = []
        
        if num_pending >= 2:
            reasons.append(f"{num_pending} pending orders with potential consolidation")
            
            # Check geographical clustering
            if self._has_geographical_clustering(state.pending_shipments):
                reasons.append("geographical clustering detected")
        
        if num_pending == 1:
            reasons.append("single order with low utilization - evaluating wait vs dispatch")
        
        if num_available > 1:
            reasons.append(f"{num_available} vehicles requiring optimal assignment")
        
        return "CFA optimization needed: " + ", ".join(reasons)
    
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
                reasoning="No pending shipments - system idle",
                confidence=1.0
            )
        
        # Calculate time until next deadline
        min_time_to_deadline = min(
            s.time_to_deadline(state.timestamp).total_seconds() / 3600
            for s in pending
        )
        
        max_wait_hours = self.config.get('max_consolidation_wait_hours', 24.0)
        
        if min_time_to_deadline > max_wait_hours:
            # Plenty of time - wait for more shipments
            return PFAAction(
                action_type='WAIT',
                shipments=[],
                vehicle=None,
                reasoning=f"Waiting for consolidation: {min_time_to_deadline:.1f}h until next deadline "
                         f"(max wait: {max_wait_hours:.1f}h). Low utilization orders should wait for batching.",
                confidence=0.7
            )
        else:
            # Time getting tighter - defer to CFA for optimization
            return PFAAction(
                action_type='DEFER_TO_CFA',
                shipments=[],
                vehicle=None,
                reasoning=f"Time constraint approaching: {min_time_to_deadline:.1f}h until deadline. "
                         f"Need CFA to optimize dispatch vs wait decision.",
                confidence=0.8
            )
    
    def _has_geographical_clustering(self, shipments: List[Shipment]) -> bool:
        """
        Check if shipments have geographical clustering potential
        
        Clustering indicators:
        1. Shared origin zones
        2. Shared destination zones
        3. Geographic proximity (origins within ~20km)
        4. Route overlap potential
        
        Returns True if consolidation makes geographical sense
        """
        if len(shipments) < 2:
            return False
        
        # Check 1: Origin zone clustering
        origin_zones = {}
        for s in shipments:
            zone = s.origin.zone_id or 'unknown'
            origin_zones[zone] = origin_zones.get(zone, 0) + 1
        
        # If 2+ shipments share origin zone → good clustering
        if any(count >= 2 for count in origin_zones.values()):
            return True
        
        # Check 2: Destination zone clustering
        dest_zones = {}
        for s in shipments:
            for dest in s.destinations:
                zone = dest.zone_id or 'unknown'
                dest_zones[zone] = dest_zones.get(zone, 0) + 1
        
        # If 2+ destinations in same zone → good clustering
        if any(count >= 2 for count in dest_zones.values()):
            return True
        
        # Check 3: Geographic proximity of origins
        origins = [s.origin for s in shipments]
        if self._locations_are_clustered(origins, radius_km=20):
            return True
        
        # Check 4: Geographic proximity of destinations
        all_destinations = []
        for s in shipments:
            all_destinations.extend(s.destinations)
        
        if len(all_destinations) >= 2:
            if self._locations_are_clustered(all_destinations, radius_km=50):
                return True
        
        return False
    
    def _locations_are_clustered(self, locations: List[Location], radius_km: float) -> bool:
        """
        Check if locations are geographically clustered within radius
        
        Args:
            locations: List of Location objects
            radius_km: Maximum radius for clustering (km)
            
        Returns:
            True if locations cluster within radius
        """
        if len(locations) < 2:
            return False
        
        # Calculate centroid
        lats = [loc.lat for loc in locations]
        lngs = [loc.lng for loc in locations]
        
        centroid_lat = np.mean(lats)
        centroid_lng = np.mean(lngs)
        
        # Check if all locations within radius of centroid
        for loc in locations:
            distance = self._haversine_distance(
                loc.lat, loc.lng,
                centroid_lat, centroid_lng
            )
            if distance > radius_km:
                return False  # At least one location too far
        
        return True  # All locations within radius
    
    def _haversine_distance(self, lat1: float, lng1: float, 
                           lat2: float, lng2: float) -> float:
        """Calculate haversine distance between two points in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lng1 = radians(lat1), radians(lng1)
        lat2, lng2 = radians(lat2), radians(lng2)
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _shipment_covered_by_active_route(self, shipment: Shipment, 
                                          state: SystemState) -> bool:
        """Check if shipment is already in an active route"""
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
        3. Best utilization
        
        Returns:
            Best vehicle or None if no suitable vehicle
        """
        candidates = []
        
        for vehicle in vehicles:
            # Must have capacity
            if not self._vehicle_can_handle(vehicle, [shipment]):
                continue
            
            # Calculate distance to origin
            distance = self._haversine_distance(
                vehicle.current_location.lat,
                vehicle.current_location.lng,
                shipment.origin.lat,
                shipment.origin.lng
            )
            
            # Calculate utilization
            volume_util = shipment.volume / vehicle.capacity.volume
            weight_util = shipment.weight / vehicle.capacity.weight
            utilization = max(volume_util, weight_util)
            
            # Score: prioritize high utilization and low distance
            # Higher score = better vehicle
            utilization_score = utilization * 100
            distance_penalty = distance * 0.1
            total_score = utilization_score - distance_penalty
            
            candidates.append((vehicle, total_score, utilization, distance))
        
        if not candidates:
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
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
    
    def _estimate_distance(self, loc1: Location, loc2: Location) -> float:
        """Estimate distance between two locations in km"""
        return self._haversine_distance(
            loc1.lat, loc1.lng,
            loc2.lat, loc2.lng
        )
    
    def can_handle_state(self, state: SystemState) -> Tuple[bool, float]:
        """
        Check if PFA can confidently handle this state
        
        Used by meta-controller to decide function class
        
        Args:
            state: Current system state
            
        Returns:
            Tuple of (can_handle: bool, confidence: float)
        """
        decision = self.decide(state)
        
        # PFA can handle if it made a confident decision
        if decision.action_type.startswith('EMERGENCY'):
            return (True, 1.0)  # Always handle emergencies
        
        if decision.action_type == 'DISPATCH_IMMEDIATE':
            return (True, decision.confidence)
        
        # Defers to CFA or waits - PFA cannot handle
        return (False, 0.0)


# Helper method to add to Shipment class (if not already present)
def _add_shipment_priority_name_method():
    """Add priority_name() method to Shipment class if needed"""
    def priority_name(self) -> str:
        """Convert priority value to readable name"""
        if self.priority >= 3.0:
            return "EMERGENCY"
        elif self.priority >= 2.0:
            return "URGENT"
        else:
            return "STANDARD"
    
    # This would be added to Shipment dataclass
    # For now, it's defined here as a reference
    pass