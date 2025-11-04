# src/core/meta_controller.py
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from .state_manager import SystemState
from datetime import datetime

logger = logging.getLogger(__name__)


class FunctionClass(Enum):
    PFA = "pfa"
    CFA = "cfa"
    DLA = "dla"


@dataclass
class MetaDecision:
    """Final decision from meta-controller"""

    function_class: FunctionClass
    action_type: str
    action_details: dict
    reasoning: str
    confidence: float


@dataclass
class ComplexityAssessment:
    """Assessment of problem complexity"""

    is_complex: bool
    complexity_score: float
    factors: dict
    reasoning: str


@dataclass
class StakesAssessment:
    """Assessment of decision stakes"""

    is_high_stakes: bool
    stakes_score: float
    factors: dict
    reasoning: str


class MetaController:
    def __init__(self):
        from .pfa import PolicyFunctionApproximator
        from .cfa import CostFunctionApproximator
        from .dla import DirectLookaheadApproximator
        from .vfa_neural import NeuralVFA
        from ..config.senga_config import SengaConfigurator

        config = SengaConfigurator()
        self.pfa = PolicyFunctionApproximator()
        self.cfa = CostFunctionApproximator(config)
        self.dla = DirectLookaheadApproximator()
        self.vfa = NeuralVFA()

        self.emergency_threshold = 2.0
        self.simple_max = 2

        # Complexity thresholds
        self.complexity_threshold = 0.6
        self.stakes_threshold = 0.7

        # Weights for complexity factors
        self.complexity_weights = {
            "shipment_count": 0.25,
            "vehicle_count": 0.20,
            "time_pressure": 0.30,
            "constraint_density": 0.25,
        }

    def decide(self, state: SystemState) -> MetaDecision:
        """
        Main decision coordination - CORRECTED for actual PFA API
        """
        # Step 1: Get PFA action using correct method name
        pfa_action = self.pfa.select_action(state)  # CORRECT: select_action not decide
        self._last_pfa_action = pfa_action

        # Handle dispatch_batch
        if pfa_action.action_type == "dispatch_batch":
            batch = {
                "batch_id": f"PFA_{datetime.now().timestamp()}",
                "shipments": pfa_action.shipment_ids,
                "vehicle": None,  # PFA doesn't assign vehicles
            }
            return MetaDecision(
                function_class=FunctionClass.PFA,
                action_type="DISPATCH",
                action_details={
                    "type": "DISPATCH",
                    "batches": [batch],
                },
                reasoning=pfa_action.reasoning,
                confidence=pfa_action.confidence,
            )

        # Handle dispatch_single
        if pfa_action.action_type == "dispatch_single":
            batch = {
                "batch_id": f"PFA_{datetime.now().timestamp()}",
                "shipments": pfa_action.shipment_ids,
                "vehicle": None,
            }
            return MetaDecision(
                function_class=FunctionClass.PFA,
                action_type="DISPATCH",
                action_details={
                    "type": "DISPATCH",
                    "batches": [batch],
                },
                reasoning=pfa_action.reasoning,
                confidence=pfa_action.confidence,
            )

        # Handle wait
        if pfa_action.action_type == "wait":
            # Only wait if confidence is high, otherwise try CFA
            if pfa_action.confidence > 0.7:
                return MetaDecision(
                    function_class=FunctionClass.PFA,
                    action_type="WAIT",
                    action_details={"type": "WAIT"},
                    reasoning=pfa_action.reasoning,
                    confidence=pfa_action.confidence,
                )

        # Step 2: If PFA didn't give confident answer, assess complexity
        complexity = self._assess_complexity(state)
        stakes = self._assess_stakes(state)

        # Step 3: Route to CFA or DLA
        if stakes.is_high_stakes or complexity.is_complex:
            return self._use_dla(state, complexity, stakes)
        else:
            return self._use_cfa(state)

    def _assess_complexity(self, state: SystemState) -> ComplexityAssessment:
        """
        Assess the complexity of the current decision situation.

        Complexity is determined by multiple factors:
        - Number of pending shipments
        - Number of available vehicles
        - Time pressure (approaching deadlines)
        - Constraint density (capacity, time windows, etc.)
        """
        factors = {}

        # Factor 1: Shipment count complexity
        shipment_count = len(state.pending_shipments)
        factors["shipment_count"] = self._normalize_shipment_complexity(shipment_count)

        # Factor 2: Vehicle availability complexity
        available_vehicles = len(
            [v for v in state.get_available_vehicles() if v.status == "idle"]
        )
        factors["vehicle_count"] = self._normalize_vehicle_complexity(
            available_vehicles
        )

        # Factor 3: Time pressure complexity
        factors["time_pressure"] = self._calculate_time_pressure(state)

        # Factor 4: Constraint density
        factors["constraint_density"] = self._calculate_constraint_density(state)

        # Calculate overall complexity score
        complexity_score = sum(
            weight * factors[factor]
            for factor, weight in self.complexity_weights.items()
        )

        # Determine if situation is complex
        is_complex = complexity_score > self.complexity_threshold

        reasoning = self._generate_complexity_reasoning(
            is_complex, complexity_score, factors, shipment_count, available_vehicles
        )

        return ComplexityAssessment(
            is_complex=is_complex,
            complexity_score=complexity_score,
            factors=factors,
            reasoning=reasoning,
        )

    def _normalize_shipment_complexity(self, shipment_count: int) -> float:
        """Normalize shipment count to 0-1 complexity score"""
        # Exponential growth - complexity increases rapidly with more shipments
        if shipment_count <= 5:
            return 0.2
        elif shipment_count <= 10:
            return 0.4
        elif shipment_count <= 20:
            return 0.7
        else:
            return 0.9

    def _normalize_vehicle_complexity(self, vehicle_count: int) -> float:
        """Normalize vehicle count to 0-1 complexity score"""
        # More vehicles mean more assignment combinations
        if vehicle_count <= 2:
            return 0.3
        elif vehicle_count <= 5:
            return 0.5
        elif vehicle_count <= 10:
            return 0.7
        else:
            return 0.9

    def _calculate_time_pressure(self, state: SystemState) -> float:
        """Calculate time pressure based on approaching deadlines"""
        if not state.pending_shipments:
            return 0.0

        current_time = state.timestamp
        urgent_count = 0

        for shipment in state.pending_shipments:
            hours_until_deadline = (
                shipment.deadline - current_time
            ).total_seconds() / 3600

            if hours_until_deadline < 2.0:  # Very urgent
                urgent_count += 1
            elif hours_until_deadline < 6.0:  # Moderately urgent
                urgent_count += 0.5

        # Normalize by total shipments
        time_pressure = min(1.0, urgent_count / len(state.pending_shipments) * 2)
        return time_pressure

    def _calculate_constraint_density(self, state: SystemState) -> float:
        """Calculate constraint density based on various constraints"""
        constraint_score = 0.0
        constraint_factors = 0

        # Vehicle capacity constraints
        available_vehicles = state.get_available_vehicles()
        if available_vehicles:
            # Weight capacity variation
            weight_capacities = [v.capacity.weight for v in available_vehicles]
            if weight_capacities:
                weight_variation = np.std(weight_capacities)
                constraint_score += min(1.0, weight_variation / 50)
                constraint_factors += 1

            # Volume capacity variation
            volume_capacities = [v.capacity.volume for v in available_vehicles]
            if volume_capacities:
                volume_variation = np.std(volume_capacities)
                constraint_score += min(1.0, volume_variation / 20)
                constraint_factors += 1

        # Shipment constraints
        if state.pending_shipments:
            # Weight variation
            shipment_weights = [s.weight for s in state.pending_shipments]
            if shipment_weights:
                weight_variation = np.std(shipment_weights)
                constraint_score += min(1.0, weight_variation / 20)
                constraint_factors += 1

            # Volume variation
            shipment_volumes = [s.volume for s in state.pending_shipments]
            if shipment_volumes:
                volume_variation = np.std(shipment_volumes)
                constraint_score += min(1.0, volume_variation / 10)
                constraint_factors += 1

        # Time window constraints
        time_window_constrained = any(
            hasattr(s, "pickup_time_window") and s.pickup_time_window is not None
            for s in state.pending_shipments
        )
        if time_window_constrained:
            constraint_score += 0.8
            constraint_factors += 1

        # Geographic constraints (if locations are available)
        if state.pending_shipments and hasattr(state.pending_shipments[0], "origin"):
            unique_locations = len(
                set(
                    s.origin.zone_id
                    for s in state.pending_shipments
                    if hasattr(s.origin, "zone_id")
                )
            )
            geographic_complexity = min(1.0, unique_locations / 5.0)
            constraint_score += geographic_complexity
            constraint_factors += 1

        # Return average constraint density
        return constraint_score / max(1, constraint_factors)

    def _generate_complexity_reasoning(
        self,
        is_complex: bool,
        score: float,
        factors: dict,
        shipment_count: int,
        vehicle_count: int,
    ) -> str:
        """Generate human-readable reasoning for complexity assessment"""

        if not is_complex:
            return f"Simple situation (score: {score:.2f}): {shipment_count} shipments, {vehicle_count} vehicles"

        # Identify primary complexity drivers
        drivers = []
        if factors["shipment_count"] > 0.7:
            drivers.append("high shipment volume")
        if factors["time_pressure"] > 0.7:
            drivers.append("urgent deadlines")
        if factors["constraint_density"] > 0.7:
            drivers.append("multiple constraints")
        if factors["vehicle_count"] > 0.7:
            drivers.append("complex vehicle assignments")

        drivers_text = ", ".join(drivers) if drivers else "multiple factors"
        return f"Complex situation (score: {score:.2f}) due to {drivers_text}"

    def _assess_stakes(self, state: SystemState) -> StakesAssessment:
        """
        Assess the stakes of the current decision.
        High stakes situations involve potential SLA violations, high-value shipments, or emergency conditions.
        """
        factors = {}

        # Factor 1: Emergency situation
        factors["emergency"] = 1.0 if self._is_emergency(state) else 0.0

        # Factor 2: High-value shipments
        factors["value_at_risk"] = self._calculate_value_at_risk(state)

        # Factor 3: SLA violation risk
        factors["sla_risk"] = self._calculate_sla_risk(state)

        # Calculate overall stakes score (use maximum of factors for high-stakes detection)
        stakes_score = max(factors.values())
        is_high_stakes = stakes_score > self.stakes_threshold

        reasoning = self._generate_stakes_reasoning(
            is_high_stakes, stakes_score, factors
        )

        return StakesAssessment(
            is_high_stakes=is_high_stakes,
            stakes_score=stakes_score,
            factors=factors,
            reasoning=reasoning,
        )

    def _calculate_value_at_risk(self, state: SystemState) -> float:
        """Calculate the value at risk from pending shipments"""
        if not state.pending_shipments:
            return 0.0

        total_value = sum(getattr(s, "value", 0) for s in state.pending_shipments)
        # Normalize - assume typical shipment value around 100, scale accordingly
        return min(1.0, total_value / (len(state.pending_shipments) * 200))

    def _calculate_sla_risk(self, state: SystemState) -> float:
        """Calculate risk of SLA violations"""
        if not state.pending_shipments:
            return 0.0

        current_time = state.timestamp
        at_risk_count = 0

        for shipment in state.pending_shipments:
            hours_until_deadline = (
                shipment.deadline - current_time
            ).total_seconds() / 3600
            # Consider shipments at risk if they have less than 4 hours until deadline
            if hours_until_deadline < 4.0:
                at_risk_count += 1

        return min(1.0, at_risk_count / len(state.pending_shipments))

    def _generate_stakes_reasoning(
        self, is_high_stakes: bool, score: float, factors: dict
    ) -> str:
        """Generate human-readable reasoning for stakes assessment"""

        if not is_high_stakes:
            return f"Normal stakes (score: {score:.2f})"

        # Identify stake drivers
        drivers = []
        if factors["emergency"] > 0:
            drivers.append("emergency situation")
        if factors["value_at_risk"] > 0.7:
            drivers.append("high-value shipments")
        if factors["sla_risk"] > 0.7:
            drivers.append("SLA violation risk")

        drivers_text = ", ".join(drivers) if drivers else "high stakes factors"
        return f"High stakes (score: {score:.2f}) due to {drivers_text}"

    def _is_emergency(self, state) -> bool:
        # CORRECT: state.pending_shipments is a List
        for s in state.pending_shipments:
            if (
                s.deadline - state.timestamp
            ).total_seconds() / 3600 < self.emergency_threshold:
                return True
        return False

    def _use_dla(
        self,
        state: SystemState,
        complexity: ComplexityAssessment,
        stakes: StakesAssessment,
    ) -> MetaDecision:
        """Use Direct Lookahead Approximator for complex/high-stakes situations"""
        logger.info(f"Using DLA for complex situation: {complexity.reasoning}")

        # Call DLA to get optimized decision
        dla_action = self.dla.optimize(state)

        return MetaDecision(
            function_class=FunctionClass.DLA,
            action_type=dla_action.action_type,
            action_details=dla_action.details,
            reasoning=f"DLA: {complexity.reasoning} | {stakes.reasoning}",
            confidence=dla_action.confidence,
        )

    # In src/core/meta_controller.py - update the _use_cfa method:

    def _use_cfa(self, state: SystemState) -> MetaDecision:
        """Use Cost Function Approximator for simple situations"""
        logger.info("Using CFA for simple situation")

        # Call CFA to get cost-optimized decision
        cfa_action = self.cfa.solve(state)

        return MetaDecision(
            function_class=FunctionClass.CFA,
            action_type=cfa_action.action_type,
            action_details=cfa_action.details,
            reasoning=cfa_action.reasoning,
            confidence=cfa_action.confidence,
        )

    def update_from_outcome(
        self, decision: MetaDecision, state_before, state_after, reward: float
    ):
        self.vfa.td_update(state_before, reward, state_after)

        if decision.function_class == FunctionClass.PFA:
            from .pfa import PFAAction

            pfa_action = PFAAction(
                decision.action["action_type"],
                decision.action["shipment_ids"],
                decision.reasoning,
                decision.confidence,
            )
            self.pfa.update_from_outcome(pfa_action, reward)

        elif decision.function_class == FunctionClass.CFA:
            if "estimated_cost" in decision.action:
                self.cfa.update_parameters(
                    decision.action["estimated_cost"],
                    -reward if reward < 0 else decision.action["estimated_cost"],
                    decision.action.get("estimated_utilization", 0),
                    decision.action.get("estimated_utilization", 0),
                )
