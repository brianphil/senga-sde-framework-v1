# src/core/decision_engine.py
"""Decision Engine - FINAL FIX for all data structure issues"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from uuid import uuid4

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

            decision = self.meta_controller.decide(state_before)

            # FIX: action_type might be a dict instead of string (bug in meta_controller)
            action_type = decision.action_type
            if isinstance(action_type, dict):
                action_type = action_type.get("action_type", "WAIT")

            # Extract confidence (might be ValueEstimate or float)
            conf = decision.confidence
            if hasattr(conf, "value"):
                conf = conf.value

            logger.info(
                f"Decision: {decision.function_class.value} -> {action_type} (confidence: {conf:.2f})"
            )

            execution_result = self._execute_decision(decision, state_before)
            state_after = self.state_manager.get_current_state()

            normalized_action = self._normalize_action(decision, execution_result)
            reward_dict = self.reward_calculator.calculate_reward(
                state_before, normalized_action, state_after
            )

            # Convert IMMEDIATELY
            reward_components = RewardComponents(
                utilization_reward=reward_dict.get("utilization_bonus", 0.0),
                on_time_reward=reward_dict.get("timeliness_bonus", 0.0),
                cost_penalty=reward_dict.get("operational_cost", 0.0),
                late_penalty=reward_dict.get("delay_penalty", 0.0),
                efficiency_bonus=0.0,
                total_reward=reward_dict.get("total", 0.0),
                reasoning="",
            )

            learning_result = self._trigger_learning_update(
                state_before=state_before,
                action=normalized_action,
                reward=reward_components.total_reward,
                state_after=state_after,
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
                learning_updates_applied=learning_result["updated"],
                vfa_value_before=learning_result["value_before"],
                vfa_value_after=learning_result["value_after"],
                td_error=learning_result["td_error"],
            )

            self.cycle_history.append(result)
            self.current_cycle += 1
            self.status = EngineStatus.IDLE

            logger.info(
                f"Cycle completed in {execution_time:.1f}ms | "
                f"Reward: {reward_components.total_reward:.1f} | "
                f"TD Error: {learning_result['td_error']:.2f}"
            )
            return result

        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Cycle failed: {str(e)}", exc_info=True)
            raise

    def _execute_decision(self, decision: MetaDecision, state: SystemState) -> Dict:
        # Extract action_type (might be dict or string)
        action_type = decision.action_type
        if isinstance(action_type, dict):
            action_type = action_type.get("action_type", "WAIT").upper()
        else:
            action_type = str(action_type).upper()

        if action_type == "WAIT":
            logger.info("Decision: WAIT - no action taken")
            # FIX: Create DecisionEvent object and pass it to log_decision
            decision_event = DecisionEvent(
                id=str(uuid4()),
                timestamp=datetime.now(),
                decision_type="WAIT",
                function_class=decision.function_class.value,
                action_details={},
                state_snapshot=state,
                reasoning=decision.reasoning,
                confidence=self._extract_value(decision.confidence),
            )
            self.state_manager.log_decision(decision_event)
            return {
                "shipments_dispatched": 0,
                "vehicles_utilized": 0,
                "executed_batches": [],
            }

        elif action_type in ["DISPATCH", "DISPATCH_IMMEDIATE", "DISPATCH_SINGLE"]:
            batches = decision.action_details.get("batches", [])

            # Handle DISPATCH_SINGLE - create batch from shipment_ids
            if not batches and isinstance(decision.action_type, dict):
                shipment_ids = decision.action_type.get("shipment_ids", [])
                vehicle_id = decision.action_type.get("vehicle_id")
                if shipment_ids:
                    batches = [
                        {
                            "shipments": shipment_ids,
                            "vehicle": vehicle_id,
                            "batch_id": str(uuid4()),
                        }
                    ]

            if not batches:
                logger.warning("DISPATCH decision with no batches")
                return {
                    "shipments_dispatched": 0,
                    "vehicles_utilized": 0,
                    "executed_batches": [],
                }

            logger.info(f"Executing DISPATCH: {len(batches)} batches")

            executed_batches = []
            total_shipments = 0
            vehicles_used = set()

            for batch in batches:
                route = self._dispatch_batch(batch, state)
                executed_batches.append(
                    {
                        "batch_id": (
                            batch.batch_id
                            if hasattr(batch, "batch_id")
                            else batch.get("batch_id", str(uuid4()))
                        ),
                        "shipment_ids": (
                            [s.id for s in batch.shipments]
                            if hasattr(batch, "shipments")
                            else batch.get("shipments", [])
                        ),
                        "vehicle_id": (
                            batch.vehicle.id
                            if hasattr(batch, "vehicle")
                            else batch.get("vehicle", "unknown")
                        ),
                        "route_id": route.id,
                    }
                )
                total_shipments += (
                    len(batch.shipments)
                    if hasattr(batch, "shipments")
                    else len(batch.get("shipments", []))
                )
                vehicles_used.add(
                    batch.vehicle.id
                    if hasattr(batch, "vehicle")
                    else batch.get("vehicle", "unknown")
                )

            # FIX: Create DecisionEvent object and pass it to log_decision
            decision_event = DecisionEvent(
                id=str(uuid4()),
                timestamp=datetime.now(),
                decision_type=action_type,
                function_class=decision.function_class.value,
                action_details={
                    "batches": executed_batches,
                    "total_shipments": total_shipments,
                },
                state_snapshot=state,
                reasoning=decision.reasoning,
                confidence=self._extract_value(decision.confidence),
            )
            self.state_manager.log_decision(decision_event)

            logger.info(
                f"Executed DISPATCH: {total_shipments} shipments, {len(vehicles_used)} vehicles"
            )
            return {
                "shipments_dispatched": total_shipments,
                "vehicles_utilized": len(vehicles_used),
                "executed_batches": executed_batches,
            }

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _dispatch_batch(self, batch, state: SystemState) -> Route:
        route_id = f"route_{uuid4()}"

        # Handle both StandardBatch and dict formats
        if hasattr(batch, "shipments"):
            shipment_objs = batch.shipments
            vehicle_id = batch.vehicle.id if batch.vehicle else None
        else:
            shipment_ids = batch.get("shipments", [])
            shipment_objs = [s for s in state.pending_shipments if s.id in shipment_ids]
            vehicle_id = batch.get("vehicle")

        # CRITICAL: Assign vehicle if not provided (PFA doesn't assign vehicles)
        if not vehicle_id or vehicle_id == "unknown":
            available = state.get_available_vehicles()
            if not available:
                raise RuntimeError("No available vehicles for dispatch")
            vehicle_id = available[0].id

        # Get destination locations
        destinations = []
        for s in shipment_objs:
            if hasattr(s, "destinations") and s.destinations:
                destinations.extend(s.destinations)
            elif hasattr(s, "destination"):
                destinations.append(s.destination)

        route = Route(
            id=route_id,
            vehicle_id=vehicle_id,
            shipment_ids=[s.id for s in shipment_objs],
            sequence=destinations,
            estimated_duration=timedelta(hours=3),
            estimated_distance=50.0,
            created_at=datetime.now(),
        )

        for shipment in shipment_objs:
            shipment.status = ShipmentStatus.EN_ROUTE
            shipment.assigned_vehicle_id = vehicle_id
            shipment.assigned_route_id = route_id

        self.state_manager.add_route(route)
        return route

    def _normalize_action(self, decision: MetaDecision, execution_result: Dict) -> dict:
        # Extract action_type string
        action_type = decision.action_type
        if isinstance(action_type, dict):
            action_type = action_type.get("action_type", "WAIT")

        return {
            "type": action_type,
            "batches": execution_result.get("executed_batches", []),
            "shipment_ids": [
                sid
                for batch in execution_result.get("executed_batches", [])
                for sid in batch.get("shipment_ids", [])
            ],
        }

    def _trigger_learning_update(
        self,
        state_before: SystemState,
        action: dict,
        reward: float,
        state_after: SystemState,
    ) -> Dict:
        vfa_estimate_before = self.vfa.estimate_value(state_before)
        vfa_estimate_after = self.vfa.estimate_value(state_after)

        value_before = self._extract_value(vfa_estimate_before)
        value_after = self._extract_value(vfa_estimate_after)

        td_error = reward + value_after - value_before

        # FIX: Call update with positional arguments instead of keyword arguments
        self.vfa.update(state_before, action, reward, state_after)

        logger.debug(
            f"Learning: V(s_t)={value_before:.1f}, V(s_t+1)={value_after:.1f}, r={reward:.1f}, δ={td_error:.2f}"
        )
        return {
            "updated": True,
            "value_before": value_before,
            "value_after": value_after,
            "td_error": td_error,
        }

    def _extract_value(self, val):
        """Extract numeric value from ValueEstimate or return as-is"""
        return val.value if hasattr(val, "value") else float(val)

    def get_performance_metrics(self) -> PerformanceMetrics:
        if not self.cycle_history:
            return PerformanceMetrics(0, 0, 0, 0.0, 0.0, 0.0, {}, 0.0)

        total_shipments = sum(r.shipments_dispatched for r in self.cycle_history)
        total_dispatches = sum(
            1 for r in self.cycle_history if r.shipments_dispatched > 0
        )
        avg_reward = sum(
            r.reward_components.total_reward for r in self.cycle_history
        ) / len(self.cycle_history)
        avg_cycle_time = sum(r.execution_time_ms for r in self.cycle_history) / len(
            self.cycle_history
        )

        function_usage = {}
        for result in self.cycle_history:
            fc = result.decision.function_class.value
            function_usage[fc] = function_usage.get(fc, 0) + 1

        recent_td_errors = [abs(r.td_error) for r in self.cycle_history[-20:]]
        learning_convergence = 1.0 / (
            1.0 + sum(recent_td_errors) / len(recent_td_errors)
        )

        return PerformanceMetrics(
            len(self.cycle_history),
            total_shipments,
            total_dispatches,
            0.0,
            avg_reward,
            avg_cycle_time,
            function_usage,
            learning_convergence,
        )
