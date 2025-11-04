# src/core/meta_controller.py
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import logging
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

    def _is_emergency(self, state) -> bool:
        # CORRECT: state.pending_shipments is a List
        for s in state.pending_shipments:
            if (
                s.deadline - state.timestamp
            ).total_seconds() / 3600 < self.emergency_threshold:
                return True
        return False

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
