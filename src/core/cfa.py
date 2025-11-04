# src/core/cfa.py
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from ortools.linear_solver import pywraplp
import logging
from ..config.senga_config import SengaConfigurator
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CFAParameters:
    base_transport_multiplier: float = 1.0
    urgency_penalty_weight: float = 50.0
    consolidation_bonus: float = 30.0
    deadline_violation_penalty: float = 200.0
    traffic_uncertainty_buffer: float = 1.2


@dataclass
class CFASolution:
    batches: List[List[str]]
    assignments: Dict[str, str]
    estimated_cost: float
    estimated_utilization: float
    computation_time_ms: float


@dataclass
class CFAAction:
    """CFA action that matches the expected interface"""

    action_type: str
    details: dict
    reasoning: str
    confidence: float


class CostFunctionApproximator:
    def __init__(self, config: SengaConfigurator):
        self.config = config
        self.theta = CFAParameters()
        logger.info("CFA initialized")

    def solve(self, state, vfa_values: Optional[Dict] = None) -> CFAAction:
        import time

        start_time = time.time()

        pending = state.pending_shipments
        available_vehicles = state.get_available_vehicles()
        logger.info("Vehicles available for CFA: %d", len(available_vehicles))
        logger.info("Pending shipments for CFA: %d", len(pending))
        if not pending or not available_vehicles:
            return self._create_wait_action("No shipments or vehicles available")

        solver = pywraplp.Solver.CreateSolver("CBC")
        logger.info("Solver created: %s", "CBC" if solver else "None")
        if not solver:
            return self._create_wait_action("Solver initialization failed")

        # Decision variables
        x = {s.id: solver.BoolVar(f"x_{s.id}") for s in pending}

        # Objective
        objective = solver.Objective()
        objective.SetMinimization()

        for s in pending:
            cost = self._estimate_cost(s, state)
            urgency = self._calc_urgency(s, state.timestamp)
            objective.SetCoefficient(
                x[s.id],
                self.theta.base_transport_multiplier * cost
                + self.theta.urgency_penalty_weight * urgency,
            )

        # Capacity constraint
        vehicle = available_vehicles[0]
        solver.Add(sum(s.weight * x[s.id] for s in pending) <= vehicle.capacity.weight)

        # Solve
        status = solver.Solve()
        comp_time = (time.time() - start_time) * 1000

        if status != pywraplp.Solver.OPTIMAL:
            return self._create_wait_action(
                f"No optimal solution found (status: {status})"
            )

        # Extract solution
        dispatched = [s.id for s in pending if x[s.id].solution_value() > 0.5]

        if not dispatched:
            return self._create_wait_action(
                "CFA recommends waiting - no cost-effective batches"
            )

        # Create dispatch action
        batches = [dispatched] if dispatched else []
        assignments = {sid: vehicle.id for sid in dispatched}
        total_weight = sum(s.weight for s in pending if s.id in dispatched)
        utilization = total_weight / vehicle.capacity.weight if dispatched else 0
        estimated_cost = objective.Value()

        # Create batches for dispatch
        dispatch_batches = []
        for i, batch_shipment_ids in enumerate(batches):
            dispatch_batches.append(
                {
                    "batch_id": f"CFA_{datetime.now().timestamp()}_{i}",
                    "shipments": batch_shipment_ids,
                    "vehicle": vehicle.id,
                    "estimated_cost": estimated_cost,
                    "estimated_utilization": utilization,
                }
            )

        return CFAAction(
            action_type="DISPATCH",
            details={
                "type": "DISPATCH",
                "batches": dispatch_batches,
                "estimated_cost": estimated_cost,
                "estimated_utilization": utilization,
                "computation_time_ms": comp_time,
            },
            reasoning=f"CFA optimized dispatch: {len(dispatched)} shipments, {utilization:.1%} utilization, cost: {estimated_cost:.2f}",
            confidence=min(
                0.9, utilization * 0.8 + 0.2
            ),  # Higher confidence for better utilization
        )

    def _create_wait_action(self, reason: str) -> CFAAction:
        """Create a wait action with reasoning"""
        return CFAAction(
            action_type="WAIT",
            details={"type": "WAIT"},
            reasoning=f"CFA: {reason}",
            confidence=0.6,
        )

    def _estimate_cost(self, shipment, state) -> float:
        if not shipment.destinations:
            return 50.0
        dest = shipment.destinations[0]
        # Get first vehicle location
        if state.fleet_state:
            loc = state.fleet_state[0].current_location
            dist = self._haversine((loc.lat, loc.lng), (dest.lat, dest.lng))
            return dist * 2.0 * self.theta.traffic_uncertainty_buffer
        return 50.0

    def _calc_urgency(self, shipment, current_time) -> float:
        hours = (shipment.deadline - current_time).total_seconds() / 3600
        return max(0, min(1, 1.0 - hours / 24))

    def _haversine(self, loc1, loc2) -> float:
        R = 6371
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

    def update_parameters(
        self, predicted_cost, actual_cost, predicted_util, actual_util
    ):
        # Simple parameter adjustment based on performance
        cost_ratio = actual_cost / predicted_cost if predicted_cost > 0 else 1.0
        util_ratio = actual_util / predicted_util if predicted_util > 0 else 1.0

        # Adjust parameters based on performance
        if cost_ratio > 1.1:  # Underestimated costs
            self.theta.base_transport_multiplier *= 1.05
        elif cost_ratio < 0.9:  # Overestimated costs
            self.theta.base_transport_multiplier *= 0.95

        if util_ratio < 0.8:  # Overestimated utilization
            self.theta.consolidation_bonus *= 0.9


NeuralCFA = CostFunctionApproximator
