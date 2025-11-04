# src/core/multi_scale_coordinator.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .state_manager import StateManager
from .vfa_neural import NeuralVFA
from .cfa import CostFunctionApproximator
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)


@dataclass
class RouteOutcome:
    route_id: str
    completed_at: datetime
    initial_state: Dict
    shipments_delivered: int
    total_shipments: int
    actual_cost: float
    predicted_cost: float
    actual_duration_hours: float
    predicted_duration_hours: float
    utilization: float
    sla_compliance: bool
    delays: List[Dict]
    issues: List[str]


@dataclass
class DailyAnalytics:
    date: datetime
    total_routes: int
    total_shipments: int
    avg_utilization: float
    sla_compliance_rate: float
    cost_prediction_error: float
    duration_prediction_error: float
    function_class_performance: Dict[str, float]
    top_delay_causes: List[str]


class MultiScaleCoordinator:
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.vfa = NeuralVFA()
        self.cfa_neural = CostFunctionApproximator(self.config)
        logger.info("Multi-Scale Coordinator initialized")

    def process_completed_route(self, route_outcome: RouteOutcome):
        logger.info(f"Processing completed route {route_outcome.route_id}")

        actual_reward = -route_outcome.actual_cost
        if not route_outcome.sla_compliance:
            sla_penalty = self.config.business_config.get("sla_penalty_per_hour", 1000)
            actual_reward -= sla_penalty

        predicted_value = -route_outcome.predicted_cost
        td_error = actual_reward - predicted_value

        self.vfa.update(
            state=None, action_value=predicted_value, actual_outcome=actual_reward
        )

        logger.info(f"Route {route_outcome.route_id}: TD error = {td_error:.2f}")

        self.state_manager.log_learning_update(
            update_type="route_completion",
            state_features=route_outcome.initial_state,
            td_error=td_error,
            actual_reward=actual_reward,
            predicted_reward=predicted_value,
        )

    def process_completed_route_with_cfa_learning(
        self, route_outcome: RouteOutcome, batch_formation: Dict
    ):
        self.process_completed_route(route_outcome)

        if self.cfa_neural:
            self.cfa_neural.update_parameters(
                predicted_cost=route_outcome.predicted_cost,
                actual_cost=route_outcome.actual_cost,
                predicted_util=batch_formation.get("predicted_utilization", 0),
                actual_util=route_outcome.utilization,
            )

            logger.info(f"CFA parameters updated from route {route_outcome.route_id}")

    def run_daily_strategic_update(
        self, target_date: Optional[datetime] = None
    ) -> DailyAnalytics:
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)

        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        logger.info(f"Running daily update for {day_start.date()}")

        cursor = self.state_manager.conn.execute(
            """
            SELECT COUNT(*), AVG(utilization), AVG(CASE WHEN sla_compliance = 1 THEN 1 ELSE 0 END)
            FROM route_outcomes
            WHERE completed_at >= ? AND completed_at < ?
            """,
            (day_start, day_end),
        )
        route_stats = cursor.fetchone()
        total_routes = route_stats[0] or 0
        avg_util = route_stats[1] or 0.0
        avg_on_time = route_stats[2] or 0.0

        cursor = self.state_manager.conn.execute(
            """
            SELECT COUNT(*), AVG(ABS(td_error))
            FROM learning_updates
            WHERE timestamp >= ? AND timestamp < ?
            """,
            (day_start, day_end),
        )
        learning_stats = cursor.fetchone()
        total_updates = learning_stats[0] or 0
        avg_td_error = learning_stats[1] or 0.0

        cursor = self.state_manager.conn.execute(
            """
            SELECT function_class, COUNT(*)
            FROM decision_log
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY function_class
            """,
            (day_start, day_end),
        )
        fc_distribution = dict(cursor.fetchall())

        analytics = DailyAnalytics(
            date=day_start,
            total_routes=total_routes,
            total_shipments=0,
            avg_utilization=avg_util,
            sla_compliance_rate=avg_on_time,
            cost_prediction_error=0.0,
            duration_prediction_error=0.0,
            function_class_performance=fc_distribution,
            top_delay_causes=[],
        )

        logger.info(
            f"Daily analytics: {total_routes} routes, {avg_util:.2%} utilization, "
            f"{avg_on_time:.2%} on-time, {total_updates} learning updates"
        )

        return analytics

    def get_learning_convergence_metrics(self, days: int = 7) -> Dict:
        cutoff = datetime.now() - timedelta(days=days)

        cursor = self.state_manager.conn.execute(
            """
            SELECT 
                DATE(timestamp) as day,
                AVG(ABS(td_error)) as avg_td_error,
                COUNT(*) as num_updates
            FROM learning_updates
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY day
            """,
            (cutoff,),
        )

        daily_metrics = [
            {"date": row[0], "avg_td_error": row[1], "num_updates": row[2]}
            for row in cursor.fetchall()
        ]

        if not daily_metrics:
            return {
                "daily_metrics": [],
                "trend": "insufficient_data",
                "convergence_score": 0.0,
            }

        recent_errors = [m["avg_td_error"] for m in daily_metrics[-3:]]
        avg_recent_error = sum(recent_errors) / len(recent_errors)

        all_errors = [m["avg_td_error"] for m in daily_metrics]
        trend = "improving" if all_errors[0] > all_errors[-1] else "stable"

        convergence_score = 1.0 / (1.0 + avg_recent_error)

        return {
            "daily_metrics": daily_metrics,
            "trend": trend,
            "convergence_score": convergence_score,
            "avg_recent_td_error": avg_recent_error,
        }
