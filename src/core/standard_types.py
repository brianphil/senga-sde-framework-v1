# src/core/standard_types.py
"""
Standardized data types for consistent batch/action representation
across all function classes (PFA, CFA, DLA).

These types provide a unified interface while preserving backward compatibility
with existing state_manager types.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum


class ActionType(Enum):
    """Standardized action types"""

    DISPATCH = "DISPATCH"
    DISPATCH_IMMEDIATE = "DISPATCH_IMMEDIATE"  # For PFA emergency
    WAIT = "WAIT"
    EMERGENCY_NO_VEHICLE = "EMERGENCY_NO_VEHICLE"
    EMERGENCY_NO_CAPACITY = "EMERGENCY_NO_CAPACITY"


@dataclass(frozen=True)
class RouteStop:
    """
    Single stop in a route sequence

    Represents a location where shipments are picked up or delivered.
    Immutable to ensure route integrity.
    """

    location_lat: float
    location_lon: float
    location_address: str
    shipment_ids: List[str] = field(default_factory=list)
    arrival_time_minutes: Optional[float] = None  # Minutes from route start
    service_time_minutes: float = 5.0  # Time for pickup/delivery

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict for JSON storage"""
        return {
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "location_address": self.location_address,
            "shipment_ids": self.shipment_ids,
            "arrival_time_minutes": self.arrival_time_minutes,
            "service_time_minutes": self.service_time_minutes,
        }

    def to_location_dict(self) -> dict:
        """Convert to state_manager Location format"""
        return {
            "lat": self.location_lat,
            "lng": self.location_lon,
            "formatted_address": self.location_address,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteStop":
        """Create from serialized dict"""
        return cls(
            location_lat=data["location_lat"],
            location_lon=data["location_lon"],
            location_address=data["location_address"],
            shipment_ids=data.get("shipment_ids", []),
            arrival_time_minutes=data.get("arrival_time_minutes"),
            service_time_minutes=data.get("service_time_minutes", 5.0),
        )


@dataclass(frozen=True)
class StandardBatch:
    """
    Standardized batch representation - produced by ALL function classes.

    This is the unified format that replaces inconsistent dict structures.
    All function classes (PFA, CFA, DLA) must return batches in this format.

    Mathematical representation:
    - Batch B = (S, V, R, d, t, c, u_v, u_w)
    - S: set of shipment IDs
    - V: vehicle ID
    - R: ordered route sequence
    - d: estimated distance (km)
    - t: estimated duration (hours)
    - c: estimated cost
    - u_v: volume utilization [0,1]
    - u_w: weight utilization [0,1]
    """

    batch_id: str
    shipment_ids: List[str]  # ALWAYS IDs, never objects
    vehicle_id: str  # ALWAYS ID, never object
    route_stops: List[RouteStop]
    estimated_distance_km: float
    estimated_duration_hours: float
    estimated_cost: float
    utilization_volume: float  # 0-1
    utilization_weight: float  # 0-1

    # Optional metadata
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def validate(self) -> bool:
        """
        Validate batch integrity

        Returns:
            True if batch is valid and ready for execution
        """
        if len(self.shipment_ids) == 0:
            return False

        if len(self.route_stops) < 2:  # Need at least origin + one destination
            return False

        if self.estimated_distance_km <= 0:
            return False

        if not (0 <= self.utilization_volume <= 1):
            return False

        if not (0 <= self.utilization_weight <= 1):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for backward compatibility with existing code
        """
        return {
            "id": self.batch_id,
            "shipments": self.shipment_ids,
            "vehicle": self.vehicle_id,
            "route_stops": [
                stop.to_dict() for stop in self.route_stops
            ],  # Use serializable dict
            "route": [
                stop.to_location_dict() for stop in self.route_stops
            ],  # Legacy format
            "estimated_distance": self.estimated_distance_km,
            "estimated_duration": self.estimated_duration_hours,
            "estimated_cost": self.estimated_cost,
            "utilization_volume": self.utilization_volume,
            "utilization_weight": self.utilization_weight,
            "utilization": max(
                self.utilization_volume, self.utilization_weight
            ),  # Legacy
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StandardBatch":
        """
        Create from legacy dict format

        Handles both PFA and CFA dict formats
        """
        # Extract route stops - handle both new and legacy formats
        route_stops = []

        # Try new format first
        if "route_stops" in data:
            for stop_data in data["route_stops"]:
                route_stops.append(RouteStop.from_dict(stop_data))
        # Fall back to legacy format
        elif "route" in data:
            for stop in data["route"]:
                route_stops.append(
                    RouteStop(
                        location_lat=stop.get("lat", 0),
                        location_lon=stop.get("lng", 0),
                        location_address=stop.get("formatted_address", ""),
                        shipment_ids=stop.get("shipment_ids", []),
                    )
                )

        # Handle missing route (PFA simple dispatch)
        if not route_stops and "shipments" in data:
            # Create minimal route: origin + destination
            route_stops = [
                RouteStop(0, 0, "Origin", []),
                RouteStop(0, 0, "Destination", data["shipments"]),
            ]

        return cls(
            batch_id=data.get("id", f"batch_{datetime.now().timestamp()}"),
            shipment_ids=data.get("shipments", []),
            vehicle_id=data.get("vehicle", ""),
            route_stops=route_stops,
            estimated_distance_km=data.get("estimated_distance", 0),
            estimated_duration_hours=data.get("estimated_duration", 0),
            estimated_cost=data.get("estimated_cost", 0),
            utilization_volume=data.get(
                "utilization_volume", data.get("utilization", 0)
            ),
            utilization_weight=data.get(
                "utilization_weight", data.get("utilization", 0)
            ),
        )


@dataclass(frozen=True)
class StandardAction:
    """
    Unified action format from all function classes.

    Replaces MetaDecision.action_details dict with strongly-typed structure.
    All function classes must return actions in this format.
    """

    action_type: ActionType
    batches: List[StandardBatch]
    function_class: str  # 'pfa', 'cfa', 'dla', 'vfa'
    reasoning: str
    confidence: float  # 0-1
    total_estimated_cost: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def validate(self) -> bool:
        """
        Validate action before execution

        Returns:
            True if action is valid
        """
        if self.action_type == ActionType.WAIT:
            return len(self.batches) == 0

        if self.action_type in [ActionType.DISPATCH, ActionType.DISPATCH_IMMEDIATE]:
            if len(self.batches) == 0:
                return False
            return all(batch.validate() for batch in self.batches)

        # Emergency actions don't need batches
        if self.action_type in [
            ActionType.EMERGENCY_NO_VEHICLE,
            ActionType.EMERGENCY_NO_CAPACITY,
        ]:
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to fully serializable dict for JSON storage
        """
        return {
            "action_type": self.action_type.value,
            "batches": [batch.to_dict() for batch in self.batches],
            "function_class": self.function_class,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "total_estimated_cost": self.total_estimated_cost,
            "timestamp": self.timestamp,
        }

    def to_meta_decision_details(self) -> dict:
        """
        Convert to MetaDecision.action_details format

        This maintains backward compatibility with existing decision_engine code.
        """
        if self.action_type == ActionType.WAIT:
            return {}

        if self.action_type in [ActionType.DISPATCH, ActionType.DISPATCH_IMMEDIATE]:
            return {
                "batches": [batch.to_dict() for batch in self.batches],
                "total_cost": self.total_estimated_cost,
            }

        # Emergency cases
        return {"reasoning": self.reasoning}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardAction":
        """
        Create from serialized dict
        """
        # Parse batches
        batches = []
        for batch_data in data.get("batches", []):
            batches.append(StandardBatch.from_dict(batch_data))

        return cls(
            action_type=ActionType(data["action_type"]),
            batches=batches,
            function_class=data["function_class"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            total_estimated_cost=data["total_estimated_cost"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
        )

    @classmethod
    def from_meta_decision(
        cls, decision, batches: List[StandardBatch] = None
    ) -> "StandardAction":
        """
        Create from existing MetaDecision

        Args:
            decision: MetaDecision from meta_controller
            batches: Optional pre-parsed batches
        """
        # Parse action type
        action_type_str = decision.action_type
        try:
            action_type = ActionType[action_type_str.upper()]
        except KeyError:
            # Handle legacy action types
            if "DISPATCH" in action_type_str:
                action_type = ActionType.DISPATCH
            elif "WAIT" in action_type_str:
                action_type = ActionType.WAIT
            else:
                action_type = ActionType.WAIT

        # Parse batches if not provided
        if batches is None:
            batches = []
            batch_dicts = decision.action_details.get("batches", [])
            for batch_dict in batch_dicts:
                batches.append(StandardBatch.from_dict(batch_dict))

        return cls(
            action_type=action_type,
            batches=batches,
            function_class=decision.function_class.value,
            reasoning=decision.reasoning,
            confidence=decision.confidence,
            total_estimated_cost=decision.action_details.get("total_cost", 0),
            timestamp=datetime.now().timestamp(),
        )


# Helper functions for conversion


def create_standard_batch_from_pfa(
    shipment_ids: List[str],
    vehicle_id: str,
    origin_lat: float,
    origin_lon: float,
    origin_addr: str,
    dest_lat: float,
    dest_lon: float,
    dest_addr: str,
    distance_km: float,
    duration_hours: float,
    cost: float,
    util_volume: float,
    util_weight: float,
) -> StandardBatch:
    """
    Create StandardBatch from PFA simple dispatch

    This is a helper for PFA to create properly formatted batches
    without needing to understand the complex route optimization.
    """
    route_stops = [
        RouteStop(origin_lat, origin_lon, origin_addr, []),
        RouteStop(
            dest_lat, dest_lon, dest_addr, shipment_ids, duration_hours * 60, 15.0
        ),
    ]

    return StandardBatch(
        batch_id=f"pfa_{datetime.now().strftime('%Y%m%d%H%M%S')}_{shipment_ids[0]}",
        shipment_ids=shipment_ids,
        vehicle_id=vehicle_id,
        route_stops=route_stops,
        estimated_distance_km=distance_km,
        estimated_duration_hours=duration_hours,
        estimated_cost=cost,
        utilization_volume=util_volume,
        utilization_weight=util_weight,
    )


def create_wait_action(
    function_class: str, reasoning: str, confidence: float = 0.7
) -> StandardAction:
    """
    Create standardized WAIT action

    Helper function for all function classes to create wait actions.
    """
    return StandardAction(
        action_type=ActionType.WAIT,
        batches=[],
        function_class=function_class,
        reasoning=reasoning,
        confidence=confidence,
        total_estimated_cost=0,
        timestamp=datetime.now().timestamp(),
    )


# Serialization helper function (also add this to state_manager.py)
def serialize_for_json(obj: Any) -> Any:
    """
    Helper function to serialize objects for JSON storage

    Handles StandardAction, StandardBatch, RouteStop, and other custom types
    """
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        # Handle dataclasses and regular objects
        return {
            k: serialize_for_json(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    elif isinstance(obj, Enum):
        return obj.value
    else:
        # For any other type, convert to string representation
        return str(obj)
