# src/core/dla.py
"""
Direct Lookahead Approximation - CORRECTED
Solve optimization over horizon, NOT Monte Carlo
NO SIGNATURE CHANGES
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from ortools.linear_solver import pywraplp
import logging

logger = logging.getLogger(__name__)


@dataclass
class LookaheadResult:
    """Result from DLA - SAME AS BEFORE"""

    immediate_action: Dict
    horizon_plan: List[Dict]
    expected_cost: float
    expected_reward: float
    computation_time_ms: float
    scenarios_evaluated: int


class DirectLookaheadApproximator:
    """Powell's DLA: Optimization over horizon"""

    def __init__(self, horizon_hours: int = 4, num_scenarios: int = 50):
        self.horizon_hours = horizon_hours
        self.num_scenarios = num_scenarios
        self.gamma = 0.95
        logger.info(
            f"DLA initialized: deterministic lookahead, {horizon_hours}hr horizon"
        )

    def solve_deterministic_lookahead(self, state, vfa=None) -> LookaheadResult:
        """Solve MIP over horizon - SAME SIGNATURE"""
        import time

        start = time.time()

        pending = [s for s in state.shipments.values() if s.status == "pending"]
        if not pending:
            return LookaheadResult({}, [], 0, 0, 0, 0)

        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            return LookaheadResult({}, [], 0, 0, 0, 0)

        # Decision variables over horizon
        periods = list(range(self.horizon_hours + 1))
        x = {
            t: {s.id: solver.BoolVar(f"x_{t}_{s.id}") for s in pending} for t in periods
        }

        # Objective: sum costs over horizon
        objective = solver.Objective()
        objective.SetMinimization()

        for t in periods[:-1]:
            discount = self.gamma**t
            for s in pending:
                cost = self._estimate_cost(s, state, t)
                urgency = self._calc_urgency(s, state.current_time + timedelta(hours=t))
                objective.SetCoefficient(x[t][s.id], discount * (cost + 50 * urgency))

        # Constraints
        for s in pending:
            solver.Add(sum(x[t][s.id] for t in periods) <= 1)

        vehicles = list(state.vehicles.values())
        if vehicles:
            capacity = vehicles[0].capacity_kg
            for t in periods[:-1]:
                solver.Add(sum(s.weight_kg * x[t][s.id] for s in pending) <= capacity)

        status = solver.Solve()
        comp_time = (time.time() - start) * 1000

        if status != pywraplp.Solver.OPTIMAL:
            return LookaheadResult({}, [], 0, 0, comp_time, 1)

        # Extract immediate action (t=0)
        immediate = [s.id for s in pending if x[0][s.id].solution_value() > 0.5]

        action = {
            "action_type": (
                "dispatch_batch"
                if len(immediate) > 1
                else "dispatch_single" if immediate else "wait"
            ),
            "shipment_ids": immediate,
            "vehicle_id": vehicles[0].id if vehicles and immediate else None,
        }

        return LookaheadResult(
            action, [], objective.Value(), -objective.Value(), comp_time, 1
        )

    def _estimate_cost(self, shipment, state, time_offset: int) -> float:
        if shipment.destinations:
            dest = shipment.destinations[0]
            dist = np.sqrt((dest["lat"] + 1.286) ** 2 + (dest["lng"] - 36.817) ** 2)
            return dist * 100
        return 50.0

    def _calc_urgency(self, shipment, current_time) -> float:
        hours = (shipment.deadline - current_time).total_seconds() / 3600
        return max(0, min(1, 1.0 - hours / 24))
