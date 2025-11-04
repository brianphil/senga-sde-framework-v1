# src/core/cfa.py
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from ortools.linear_solver import pywraplp
import logging
from ..config.senga_config import SengaConfigurator

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


class CostFunctionApproximator:
    def __init__(self, config: SengaConfigurator):
        self.config = config
        self.theta = CFAParameters()
        logger.info("CFA initialized")

    def solve(self, state, vfa_values: Optional[Dict] = None) -> CFASolution:
        import time

        start_time = time.time()

        # CORRECT: state.pending_shipments is a List
        pending = state.pending_shipments
        # CORRECT: state.get_available_vehicles() returns a List
        available_vehicles = state.get_available_vehicles()

        if not pending or not available_vehicles:
            return CFASolution([], {}, 0, 0, 0)

        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            return CFASolution([], {}, 0, 0, 0)

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
            return CFASolution([], {}, 0, 0, comp_time)

        # Extract solution
        dispatched = [s.id for s in pending if x[s.id].solution_value() > 0.5]
        batches = [dispatched] if dispatched else []
        assignments = {sid: vehicle.id for sid in dispatched}

        total_weight = sum(s.weight for s in pending if s.id in dispatched)
        util = total_weight / vehicle.capacity.weight if dispatched else 0

        return CFASolution(batches, assignments, objective.Value(), util, comp_time)

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
        pass  # Simple version


NeuralCFA = CostFunctionApproximator
