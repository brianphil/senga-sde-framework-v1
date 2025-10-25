# src/core/pfa_adapter.py
"""
Adapter for PFA to produce StandardBatch outputs.

This bridges the existing PFA class with the new standardized types
WITHOUT modifying the existing pfa.py file.

Usage in meta_controller.py:
    from .pfa_adapter import PFABatchAdapter
    
    adapter = PFABatchAdapter(distance_calc, route_optimizer)
    pfa_action = self.pfa.decide(state)
    standard_batches = adapter.convert_to_standard_batches(pfa_action, state)
"""

from typing import List, Optional
from datetime import datetime
import logging

from .standard_types import (
    StandardBatch, StandardAction, ActionType, RouteStop,
    create_standard_batch_from_pfa, create_wait_action
)
from ..utils.distance_calculator import DistanceTimeCalculator
from ..algorithms.route_optimizer import RouteSequenceOptimizer, Location
from .state_manager import Shipment, VehicleState, SystemState

logger = logging.getLogger(__name__)


class PFABatchAdapter:
    """
    Converts PFAAction to StandardBatch format.
    
    This adapter bridges existing PFA with new standardized types,
    enabling:
    1. Real distance/time calculations
    2. Route sequence optimization
    3. Proper batch validation
    4. Consistent format for executor
    """
    
    def __init__(self, 
                 distance_calculator: DistanceTimeCalculator,
                 route_optimizer: RouteSequenceOptimizer):
        """
        Initialize adapter
        
        Args:
            distance_calculator: For real distance/time calculations
            route_optimizer: For route sequence optimization
        """
        self.distance_calc = distance_calculator
        self.route_optimizer = route_optimizer
    
    def convert_to_standard_action(self, pfa_action, state: SystemState) -> StandardAction:
        """
        Convert PFAAction to StandardAction with StandardBatch.
        
        Args:
            pfa_action: PFAAction from existing PFA
            state: Current system state
            
        Returns:
            StandardAction with validated batches
        """
        # Map PFA action type to StandardAction ActionType
        action_type = self._map_action_type(pfa_action.action_type)
        
        # Handle different action types
        if action_type == ActionType.WAIT:
            return create_wait_action(
                function_class='pfa',
                reasoning=pfa_action.reasoning,
                confidence=pfa_action.confidence
            )
        
        if action_type in [ActionType.EMERGENCY_NO_VEHICLE, ActionType.EMERGENCY_NO_CAPACITY]:
            # Emergency but can't dispatch - no batches
            return StandardAction(
                action_type=action_type,
                batches=[],
                function_class='pfa',
                reasoning=pfa_action.reasoning,
                confidence=pfa_action.confidence,
                total_estimated_cost=0,
                timestamp=datetime.now().timestamp()
            )
        
        # Dispatch actions - create batches
        if action_type in [ActionType.DISPATCH, ActionType.DISPATCH_IMMEDIATE]:
            batches = self._create_batches_from_pfa(pfa_action, state)
            
            if not batches:
                logger.warning("Failed to create batches from PFA action. Returning WAIT.")
                return create_wait_action('pfa', 'Failed to create valid batch', 0.5)
            
            total_cost = sum(b.estimated_cost for b in batches)
            
            return StandardAction(
                action_type=action_type,
                batches=batches,
                function_class='pfa',
                reasoning=pfa_action.reasoning,
                confidence=pfa_action.confidence,
                total_estimated_cost=total_cost,
                timestamp=datetime.now().timestamp()
            )
        
        # Unknown action type
        logger.error(f"Unknown PFA action type: {pfa_action.action_type}")
        return create_wait_action('pfa', f'Unknown action type: {pfa_action.action_type}', 0.3)
    
    def _create_batches_from_pfa(self, pfa_action, state: SystemState) -> List[StandardBatch]:
        """
        Create StandardBatch from PFAAction with real calculations.
        
        Args:
            pfa_action: PFAAction with shipments and vehicle
            state: Current system state
            
        Returns:
            List of StandardBatch (typically one batch for PFA)
        """
        if not pfa_action.shipments or not pfa_action.vehicle:
            return []
        
        shipments = pfa_action.shipments
        vehicle = pfa_action.vehicle
        
        # Get shipment details
        shipment_ids = [s.id for s in shipments]
        
        # Build origin and destinations
        # Assuming vehicle has current_location or we use first shipment origin
        if hasattr(vehicle, 'current_location') and vehicle.current_location:
            origin = Location(
                lat=vehicle.current_location.lat,
                lon=vehicle.current_location.lng,
                address=getattr(vehicle.current_location, 'formatted_address', 'Vehicle Location')
            )
        else:
            # Use first shipment origin as starting point
            first_shipment = shipments[0]
            origin = Location(
                lat=first_shipment.origin.lat,
                lon=first_shipment.origin.lng,
                address=first_shipment.origin.formatted_address
            )
        
        # Build destination locations
        destinations = []
        for shipment in shipments:
            dest = Location(
                lat=shipment.destination.lat,
                lon=shipment.destination.lng,
                address=shipment.destination.formatted_address,
                shipment_ids=[shipment.id]
            )
            destinations.append(dest)
        
        # Optimize route sequence
        try:
            route_metrics = self.route_optimizer.optimize_route_sequence(
                origin=origin,
                destinations=destinations,
                vehicle_capacity_m3=vehicle.capacity.volume if hasattr(vehicle.capacity, 'volume') else None,
                max_duration_hours=8
            )
        except Exception as e:
            logger.error(f"Route optimization failed: {e}. Using simple calculation.")
            # Fallback to simple direct route
            if len(destinations) == 1:
                distance = self.distance_calc.calculate_distance_km(
                    origin.lat, origin.lon,
                    destinations[0].lat, destinations[0].lon
                )
                duration = self.distance_calc.estimate_travel_time_hours(distance)
                
                route_stops = [
                    RouteStop(origin.lat, origin.lon, origin.address, []),
                    RouteStop(destinations[0].lat, destinations[0].lon, 
                             destinations[0].address, destinations[0].shipment_ids,
                             duration * 60, 15)
                ]
                
                total_distance = distance
                total_duration = duration
            else:
                # Multiple destinations but optimization failed - can't proceed
                return []
        else:
            route_stops = route_metrics.route_stops
            total_distance = route_metrics.total_distance_km
            total_duration = route_metrics.total_duration_hours
        
        # Calculate cost
        # Assuming vehicle has cost_per_km and fixed_cost attributes
        if hasattr(vehicle, 'cost_per_km'):
            cost_per_km = vehicle.cost_per_km
        else:
            cost_per_km = 30  # Default KES per km
        
        if hasattr(vehicle, 'fixed_cost'):
            fixed_cost = vehicle.fixed_cost
        elif hasattr(vehicle, 'fixed_cost_per_trip'):
            fixed_cost = vehicle.fixed_cost_per_trip
        else:
            fixed_cost = 1000  # Default fixed cost
        
        total_cost = cost_per_km * total_distance + fixed_cost
        
        # Calculate utilization
        total_volume = sum(s.volume for s in shipments)
        total_weight = sum(s.weight for s in shipments)
        
        vehicle_volume = vehicle.capacity.volume if hasattr(vehicle.capacity, 'volume') else 15
        vehicle_weight = vehicle.capacity.weight if hasattr(vehicle.capacity, 'weight') else 2000
        
        utilization_volume = total_volume / vehicle_volume
        utilization_weight = total_weight / vehicle_weight
        
        # Create StandardBatch
        batch = StandardBatch(
            batch_id=f"pfa_{datetime.now().strftime('%Y%m%d%H%M%S')}_{shipment_ids[0]}",
            shipment_ids=shipment_ids,
            vehicle_id=vehicle.id,
            route_stops=route_stops,
            estimated_distance_km=total_distance,
            estimated_duration_hours=total_duration,
            estimated_cost=total_cost,
            utilization_volume=min(utilization_volume, 1.0),  # Cap at 1.0
            utilization_weight=min(utilization_weight, 1.0),
            created_at=datetime.now().timestamp()
        )
        
        # Validate batch
        if not batch.validate():
            logger.error(f"Created invalid batch: {batch.batch_id}")
            return []
        
        return [batch]
    
    def _map_action_type(self, pfa_action_type: str) -> ActionType:
        """
        Map PFA action type string to StandardAction ActionType enum.
        
        PFA action types:
        - DISPATCH_IMMEDIATE
        - EMERGENCY_NO_VEHICLE
        - EMERGENCY_NO_CAPACITY
        - DEFER_TO_CFA (becomes WAIT)
        - WAIT
        """
        action_type_upper = pfa_action_type.upper()
        
        if 'DISPATCH_IMMEDIATE' in action_type_upper or 'DISPATCH' in action_type_upper:
            return ActionType.DISPATCH_IMMEDIATE
        elif 'EMERGENCY_NO_VEHICLE' in action_type_upper:
            return ActionType.EMERGENCY_NO_VEHICLE
        elif 'EMERGENCY_NO_CAPACITY' in action_type_upper:
            return ActionType.EMERGENCY_NO_CAPACITY
        elif 'DEFER' in action_type_upper or 'WAIT' in action_type_upper:
            return ActionType.WAIT
        else:
            logger.warning(f"Unknown PFA action type: {pfa_action_type}. Defaulting to WAIT.")
            return ActionType.WAIT


def create_pfa_adapter(config: dict = None) -> PFABatchAdapter:
    """
    Factory function to create PFA adapter with dependencies.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Configured PFABatchAdapter
    """
    if config is None:
        config = {}
    
    # Initialize dependencies
    distance_calc = DistanceTimeCalculator(
        use_api=config.get('use_maps_api', False),
        api_key=config.get('maps_api_key')
    )
    
    route_optimizer = RouteSequenceOptimizer(
        distance_calculator=distance_calc,
        solver_time_limit_seconds=config.get('route_optimizer_time_limit', 10)
    )
    
    return PFABatchAdapter(distance_calc, route_optimizer)