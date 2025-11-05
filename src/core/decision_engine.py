# src/core/decision_engine.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from uuid import uuid4
import asyncio

from .state_manager import (
    StateManager,
    SystemState,
    Shipment,
    ShipmentStatus,
    VehicleState,
    Route,
    DecisionEvent,
    VehicleStatus,
)
from .standard_types import StandardBatch
from .meta_controller import MetaController, MetaDecision, FunctionClass
from .vfa_neural import NeuralVFA as ValueFunctionApproximator
from .reward_calculator import RewardCalculator, RewardComponents
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class CycleResult:
    cycle_number: int
    timestamp: datetime
    state_before: SystemState
    decision: MetaDecision
    reward_components: RewardComponents
    state_after: SystemState
    execution_time_ms: float
    shipments_dispatched: int
    vehicles_utilized: int
    learning_updates_applied: bool
    vfa_value_before: float
    vfa_value_after: float
    td_error: float


@dataclass
class PerformanceMetrics:
    total_cycles: int
    total_shipments_processed: int
    total_dispatches: int
    avg_utilization: float
    avg_reward_per_cycle: float
    avg_cycle_time_ms: float
    function_class_usage: Dict[str, int]
    learning_convergence: float


class DecisionEngine:
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.meta_controller = MetaController()
        self.vfa = ValueFunctionApproximator()
        self.reward_calculator = RewardCalculator()
        self.status = EngineStatus.IDLE
        self.current_cycle = 0
        self.cycle_history: List[CycleResult] = []
        logger.info("Decision Engine initialized")

    @staticmethod
    def serialize_action_details(action_details: dict) -> dict:
        """Ensure action_details are JSON serializable"""

        def _serialize(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif hasattr(obj, "__dict__"):
                return {
                    k: _serialize(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
            elif isinstance(obj, (list, tuple)):
                return [_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: _serialize(value) for key, value in obj.items()}
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return obj.total_seconds()
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return str(obj)

        return _serialize(action_details)

    @staticmethod
    def serialize_for_json(obj):
        """
        Helper function to serialize objects for JSON storage
        """
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            # Handle dataclasses and regular objects
            return {
                k: DecisionEngine.serialize_for_json(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(obj, (list, tuple)):
            return [DecisionEngine.serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: DecisionEngine.serialize_for_json(value)
                for key, value in obj.items()
            }
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

    # Update the run_cycle method to serialize before creating DecisionEvent:
    def run_cycle(self, current_time: Optional[datetime] = None) -> CycleResult:
        if current_time is None:
            current_time = datetime.now()
        start_time = datetime.now()
        self.status = EngineStatus.RUNNING

        try:
            state_before = self.state_manager.get_current_state()
            logger.info(
                f"Cycle {self.current_cycle}: {len(state_before.pending_shipments)} pending shipments"
            )

            vfa_estimate_before = self.vfa.estimate_value(state_before)
            vfa_value_before = vfa_estimate_before.value

            decision = self.meta_controller.decide(state_before)

            action_type = decision.action_type
            if isinstance(action_type, dict):
                action_type = action_type.get("action_type", "WAIT")

            conf = decision.confidence
            if hasattr(conf, "value"):
                conf = conf.value

            logger.info(
                f"Decision: {decision.function_class.value} -> {action_type} (confidence: {conf:.2f})"
            )

            # SERIALIZE action_details BEFORE creating DecisionEvent
            serialized_action_details = self.serialize_action_details(
                decision.action_details
            )

            decision_event = DecisionEvent(
                id=str(uuid4()),
                timestamp=current_time,
                state_snapshot=state_before,
                decision_type=str(action_type),
                function_class=decision.function_class.value,
                action_details=serialized_action_details,  # Use serialized version
                reasoning=decision.reasoning,
                confidence=conf,
                vfa_value_before=vfa_value_before,
            )
            self.state_manager.log_decision(decision_event)

            normalized_action = self._normalize_action_for_transition(decision)
            transition_id = self.state_manager.log_state_transition(
                decision_id=decision_event.id,
                state_before=state_before,
                action=normalized_action,
            )

            execution_result = self._execute_decision(decision, state_before)
            state_after = self.state_manager.get_current_state()

            normalized_action_full = self._normalize_action(decision, execution_result)
            reward_dict = self.reward_calculator.calculate_reward(
                state_before, normalized_action_full, state_after
            )

            reward_components = RewardComponents(
                utilization_reward=reward_dict.get("utilization_bonus", 0.0),
                on_time_reward=reward_dict.get("timeliness_bonus", 0.0),
                cost_penalty=reward_dict.get("operational_cost", 0.0),
                late_penalty=reward_dict.get("delay_penalty", 0.0),
                efficiency_bonus=0.0,
                total_reward=reward_dict.get("total", 0.0),
                reasoning="",
            )

            vfa_estimate_after = self.vfa.estimate_value(state_after)
            vfa_value_after = vfa_estimate_after.value

            discount_factor = self.config.get("vfa.learning.discount_factor", 0.95)
            td_error = (
                reward_components.total_reward
                + discount_factor * vfa_value_after
                - vfa_value_before
            )

            self.state_manager.complete_state_transition(
                transition_id=transition_id,
                reward=reward_components.total_reward,
                state_after=state_after,
            )

            self.vfa.td_update(
                state_before,
                normalized_action_full,
                reward_components.total_reward,
                state_after,
            )

            self.meta_controller.update_from_outcome(
                decision=decision,
                state_before=state_before,
                state_after=state_after,
                reward=reward_components.total_reward,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            result = CycleResult(
                cycle_number=self.current_cycle,
                timestamp=current_time,
                state_before=state_before,
                decision=decision,
                reward_components=reward_components,
                state_after=state_after,
                execution_time_ms=execution_time,
                shipments_dispatched=execution_result["shipments_dispatched"],
                vehicles_utilized=execution_result["vehicles_utilized"],
                learning_updates_applied=True,
                vfa_value_before=vfa_value_before,
                vfa_value_after=vfa_value_after,
                td_error=td_error,
            )

            self.cycle_history.append(result)
            self.current_cycle += 1
            self.status = EngineStatus.IDLE

            logger.info(
                f"Cycle completed in {execution_time:.1f}ms | "
                f"Reward: {reward_components.total_reward:.1f} | "
                f"TD Error: {td_error:.2f}"
            )
            return result

        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Cycle failed: {str(e)}", exc_info=True)
            raise

    def _normalize_action_for_transition(self, decision: MetaDecision) -> dict:
        """Normalize action with proper serialization for state transition logging"""
        action_type = decision.action_type
        if isinstance(action_type, dict):
            action_type = action_type.get("action_type", "WAIT")

        conf = decision.confidence
        if hasattr(conf, "value"):
            conf = conf.value

        # Ensure the action details are serializable
        serialized_details = self.serialize_action_details(decision.action_details)

        return {
            "type": str(action_type).upper(),
            "function_class": decision.function_class.value,
            "confidence": float(conf),
            "summary": decision.reasoning[:200],
            "details": serialized_details,  # Use serialized version
        }

    def _dispatch_batch(
        self,
        shipment_ids: List[str],
        vehicle_id: Optional[str],
        batch_id: str,
        estimated_cost: float,
        estimated_distance: float,
        route_sequence: List = None,  # This contains dicts, not Location objects
    ):
        """
        Dispatch a batch of shipments

        FIXED: Convert dict route sequence to Location objects
        """
        from .state_manager import Location

        # Update shipment statuses
        for shipment_id in shipment_ids:
            self.state_manager.update_shipment_status(
                shipment_id=shipment_id,
                status=ShipmentStatus.EN_ROUTE,
                batch_id=batch_id,
            )

        # Update vehicle status
        if vehicle_id:
            vehicle = self.state_manager.get_vehicle_state(vehicle_id)
            if vehicle:
                vehicle.status = VehicleStatus.EN_ROUTE
                vehicle.current_route_id = batch_id
                self.state_manager.update_vehicle_state(vehicle)

        # Convert dict route sequence to Location objects
        location_sequence = []
        if route_sequence:
            for stop_dict in route_sequence:
                # Handle both dict formats (from RouteStop serialization and legacy)
                if isinstance(stop_dict, dict):
                    location_sequence.append(
                        Location(
                            place_id=stop_dict.get(
                                "place_id", f"stop_{len(location_sequence)}"
                            ),
                            lat=stop_dict.get("lat", stop_dict.get("location_lat", 0)),
                            lng=stop_dict.get("lng", stop_dict.get("location_lon", 0)),
                            formatted_address=stop_dict.get(
                                "formatted_address",
                                stop_dict.get("location_address", "Unknown"),
                            ),
                            zone_id=stop_dict.get("zone_id"),
                        )
                    )
                else:
                    # If it's already a Location object, use as-is
                    location_sequence.append(stop_dict)

        # Create Route with proper Location objects
        route = Route(
            id=batch_id,
            vehicle_id=vehicle_id or "UNASSIGNED",
            shipment_ids=shipment_ids,
            sequence=location_sequence,  # Now contains Location objects, not dicts
            estimated_duration=timedelta(hours=estimated_distance / 40),
            estimated_distance=estimated_distance,
            created_at=datetime.now(),
        )
        self.state_manager.add_route(route)

        logger.debug(f"Route {batch_id} created with {len(location_sequence)} stops")

    def _execute_decision(self, decision: MetaDecision, state: SystemState) -> Dict:
        """
        Execute decision and update state

        FIXED: Ensure route_sequence is properly formatted
        """
        action_type = decision.action_type
        if isinstance(action_type, dict):
            action_type = action_type.get("action_type", "WAIT").upper()
        else:
            action_type = str(action_type).upper()

        if action_type == "WAIT":
            logger.info("Decision: WAIT - no action taken")
            return {
                "shipments_dispatched": 0,
                "vehicles_utilized": 0,
                "batches_created": 0,
            }

        elif action_type == "DISPATCH":
            batches = decision.action_details.get("batches", [])
            if not batches:
                logger.warning("DISPATCH decision but no batches provided")
                return {
                    "shipments_dispatched": 0,
                    "vehicles_utilized": 0,
                    "batches_created": 0,
                }

            total_shipments = 0
            vehicles_used = set()

            for batch in batches:
                # Handle both serialized RouteStop objects and dicts
                shipments_in_batch = batch.get("shipments", [])
                vehicle = batch.get("vehicle")

                # Extract route sequence - ensure it's a list of dicts
                route_sequence = []
                if "route_stops" in batch:
                    # New serialized format - RouteStop objects converted to dicts
                    for stop in batch["route_stops"]:
                        if isinstance(stop, dict):
                            route_sequence.append(
                                {
                                    "lat": stop.get("location_lat", 0),
                                    "lng": stop.get("location_lon", 0),
                                    "formatted_address": stop.get(
                                        "location_address", ""
                                    ),
                                    "shipment_ids": stop.get("shipment_ids", []),
                                }
                            )
                elif "route" in batch:
                    # Legacy format - already dicts
                    route_sequence = batch.get("route", [])
                else:
                    # Fallback: create minimal route
                    route_sequence = [
                        {
                            "lat": 0,
                            "lng": 0,
                            "formatted_address": "Origin",
                            "shipment_ids": [],
                        },
                        {
                            "lat": 0,
                            "lng": 0,
                            "formatted_address": "Destination",
                            "shipment_ids": shipments_in_batch,
                        },
                    ]

                if not shipments_in_batch:
                    continue

                self._dispatch_batch(
                    shipment_ids=shipments_in_batch,
                    vehicle_id=vehicle,
                    batch_id=batch.get("id", f"batch_{uuid4()}"),
                    estimated_cost=batch.get("estimated_cost", 0),
                    estimated_distance=batch.get("estimated_distance", 0),
                    route_sequence=route_sequence,  # This is now a list of dicts
                )

                total_shipments += len(shipments_in_batch)
                if vehicle:
                    vehicles_used.add(vehicle)

            logger.info(
                f"Dispatched {total_shipments} shipments in {len(batches)} batches using {len(vehicles_used)} vehicles"
            )

            return {
                "shipments_dispatched": total_shipments,
                "vehicles_utilized": len(vehicles_used),
                "batches_created": len(batches),
            }

        else:
            logger.warning(f"Unknown action type: {action_type}")
            return {
                "shipments_dispatched": 0,
                "vehicles_utilized": 0,
                "batches_created": 0,
            }

    async def start_autonomous_mode(self, cycle_interval_minutes: int = 60):
        self.status = EngineStatus.AUTONOMOUS
        logger.info(
            f"Starting autonomous mode (cycle every {cycle_interval_minutes} minutes)"
        )

        while self.status == EngineStatus.AUTONOMOUS:
            try:
                self.run_cycle()
                await asyncio.sleep(cycle_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Autonomous cycle failed: {e}", exc_info=True)
                await asyncio.sleep(60)

    def stop_autonomous_mode(self):
        if self.status == EngineStatus.AUTONOMOUS:
            self.status = EngineStatus.IDLE
            logger.info("Autonomous mode stopped")

    def get_recent_cycles(self, n: int = 20) -> List[Dict]:
        recent = self.cycle_history[-n:]
        return [
            {
                "cycle_number": r.cycle_number,
                "timestamp": r.timestamp.isoformat(),
                "function_class": r.decision.function_class.value,
                "action_type": r.decision.action_type,
                "reward": r.reward_components.total_reward,
                "td_error": r.td_error,
                "vfa_value": r.vfa_value_before,
                "shipments_dispatched": r.shipments_dispatched,
            }
            for r in recent
        ]

    def _normalize_action(self, decision: MetaDecision, execution_result: Dict) -> Dict:
        """Normalize action with execution results for reward calculation"""
        action_type = decision.action_type
        if isinstance(action_type, dict):
            action_type = action_type.get("action_type", "WAIT")

        conf = decision.confidence
        if hasattr(conf, "value"):
            conf = conf.value

        normalized = {
            "type": str(action_type).upper(),
            "function_class": decision.function_class.value,
            "shipments_dispatched": execution_result["shipments_dispatched"],
            "vehicles_utilized": execution_result["vehicles_utilized"],
            "batches_created": execution_result.get("batches_created", 0),
            "confidence": float(conf),
            "reasoning": decision.reasoning,
        }

        if action_type == "DISPATCH":
            batches = decision.action_details.get("batches", [])
            normalized["batch_details"] = [
                {
                    "id": b.get("id"),
                    "shipments": b.get("shipments", []),
                    "vehicle": b.get("vehicle"),
                    "estimated_cost": b.get("estimated_cost", 0),
                    "utilization": b.get("utilization", 0),
                }
                for b in batches
            ]

        return normalized

    def get_performance_metrics(self) -> PerformanceMetrics:
        if not self.cycle_history:
            return PerformanceMetrics(
                total_cycles=0,
                total_shipments_processed=0,
                total_dispatches=0,
                avg_utilization=0.0,
                avg_reward_per_cycle=0.0,
                avg_cycle_time_ms=0.0,
                function_class_usage={},
                learning_convergence=0.0,
            )

        total_shipments = sum(c.shipments_dispatched for c in self.cycle_history)
        total_dispatches = sum(
            1 for c in self.cycle_history if c.shipments_dispatched > 0
        )
        avg_reward = sum(
            c.reward_components.total_reward for c in self.cycle_history
        ) / len(self.cycle_history)
        avg_cycle_time = sum(c.execution_time_ms for c in self.cycle_history) / len(
            self.cycle_history
        )

        function_usage = {}
        for c in self.cycle_history:
            fc = c.decision.function_class.value
            function_usage[fc] = function_usage.get(fc, 0) + 1

        recent_td_errors = [abs(c.td_error) for c in self.cycle_history[-100:]]
        learning_convergence = (
            1.0 / (1.0 + (sum(recent_td_errors) / len(recent_td_errors)))
            if recent_td_errors
            else 0.0
        )

        return PerformanceMetrics(
            total_cycles=len(self.cycle_history),
            total_shipments_processed=total_shipments,
            total_dispatches=total_dispatches,
            avg_utilization=0.0,
            avg_reward_per_cycle=avg_reward,
            avg_cycle_time_ms=avg_cycle_time,
            function_class_usage=function_usage,
            learning_convergence=learning_convergence,
        )
