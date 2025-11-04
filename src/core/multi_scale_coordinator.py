# src/core/multi_scale_coordinator.py
"""
Multi-Scale Coordinator - CORRECTED
NO signature changes, uses corrected CFA
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import numpy as np

from .state_manager import StateManager
from .vfa_neural import NeuralVFA
from .cfa import CostFunctionApproximator
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)


@dataclass
class RouteOutcome:
    """Outcome from completed route - SAME AS BEFORE"""

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
    """Daily performance analytics - SAME AS BEFORE"""

    date: datetime
    total_routes: int
    total_shipments: int
    avg_utilization: float
    sla_compliance_rate: float
    cost_prediction_error: float
    duration_prediction_error: float
    function_class_performance: Dict[str, float]
    top_delay_causes: List[str]


@dataclass
class WeeklyInsights:
    """Weekly strategic insights - SAME AS BEFORE"""

    week_start: datetime
    route_patterns_discovered: List[Dict]
    optimal_consolidation_windows: Dict[str, float]
    fleet_utilization_by_day: Dict[str, float]
    recommended_fleet_adjustments: List[str]
    network_topology_updates: Dict
    customer_pattern_insights: List[str]


class MultiScaleCoordinator:
    """
    Coordinates learning across multiple time scales
    CORRECTED to use proper CFA initialization
    """

    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.vfa = NeuralVFA()

        # CORRECTED: Pass config to CFA
        self.cfa_neural = CostFunctionApproximator(self.config)

        self.last_daily_update = None
        self.last_weekly_update = None

        logger.info("Multi-Scale Coordinator initialized")
        logger.info("Neural CFA initialized for learning")

    def process_completed_route(self, route_outcome: RouteOutcome):
        """
        Process completed route for learning - SAME SIGNATURE
        """
        logger.info(f"Processing completed route {route_outcome.route_id}")

        # Calculate actual reward
        actual_reward = -route_outcome.actual_cost
        if not route_outcome.sla_compliance:
            sla_penalty = self.config.business_config.get("sla_penalty_per_hour", 1000)
            actual_reward -= sla_penalty

        predicted_value = -route_outcome.predicted_cost
        td_error = actual_reward - predicted_value

        # Update VFA
        self.vfa.update(
            state=None, action_value=predicted_value, actual_outcome=actual_reward
        )

        logger.info(f"Route {route_outcome.route_id}: TD error = {td_error:.2f}")

        # Log for aggregation
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
        """
        Process route completion with CFA parameter learning - SAME SIGNATURE
        """
        # First do standard processing
        self.process_completed_route(route_outcome)

        # Then update CFA parameters
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
        """
        Run daily strategic update - SAME SIGNATURE
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)

        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        logger.info(f"Running daily update for {day_start.date()}")

        routes = self._fetch_routes_for_period(day_start, day_end)

        if not routes:
            logger.warning(f"No routes for {day_start.date()}")
            return self._empty_daily_analytics(day_start)

        total_shipments = sum(r.shipments_delivered for r in routes)
        avg_util = np.mean([r.utilization for r in routes])
        sla_rate = sum(1 for r in routes if r.sla_compliance) / len(routes)

        cost_errors = [
            abs(r.actual_cost - r.predicted_cost) / r.actual_cost
            for r in routes
            if r.actual_cost > 0
        ]
        cost_error = np.mean(cost_errors) if cost_errors else 0

        duration_errors = [
            abs(r.actual_duration_hours - r.predicted_duration_hours)
            / r.predicted_duration_hours
            for r in routes
            if r.predicted_duration_hours > 0
        ]
        duration_error = np.mean(duration_errors) if duration_errors else 0

        fc_performance = self._analyze_function_class_performance(routes)

        all_delays = [delay for r in routes for delay in r.delays]
        top_causes = self._analyze_delay_causes(all_delays)

        analytics = DailyAnalytics(
            date=day_start,
            total_routes=len(routes),
            total_shipments=total_shipments,
            avg_utilization=avg_util,
            sla_compliance_rate=sla_rate,
            cost_prediction_error=cost_error,
            duration_prediction_error=duration_error,
            function_class_performance=fc_performance,
            top_delay_causes=top_causes,
        )

        self.last_daily_update = day_start
        logger.info(
            f"Daily update complete: {len(routes)} routes, {sla_rate:.1%} SLA compliance"
        )

        return analytics

    def run_weekly_strategic_analysis(
        self, target_week: Optional[datetime] = None
    ) -> WeeklyInsights:
        """
        Run weekly strategic analysis - SAME SIGNATURE
        """
        if target_week is None:
            target_week = datetime.now() - timedelta(days=7)

        week_start = target_week
        week_end = week_start + timedelta(days=7)

        logger.info(
            f"Running weekly analysis: {week_start.date()} to {week_end.date()}"
        )

        routes = self._fetch_routes_for_period(week_start, week_end)

        if not routes:
            logger.warning(f"No routes for week {week_start.date()}")
            return self._empty_weekly_insights(week_start)

        route_patterns = self._discover_route_patterns(routes)
        optimal_windows = self._optimize_consolidation_windows(routes)
        utilization_by_day = self._analyze_utilization_by_day(routes)
        fleet_recommendations = self._generate_fleet_recommendations(utilization_by_day)
        topology_updates = {}
        customer_insights = []

        insights = WeeklyInsights(
            week_start=week_start,
            route_patterns_discovered=route_patterns,
            optimal_consolidation_windows=optimal_windows,
            fleet_utilization_by_day=utilization_by_day,
            recommended_fleet_adjustments=fleet_recommendations,
            network_topology_updates=topology_updates,
            customer_pattern_insights=customer_insights,
        )

        self.last_weekly_update = week_start
        logger.info(
            f"Weekly analysis complete: {len(route_patterns)} patterns discovered"
        )

        return insights

    def _fetch_routes_for_period(
        self, start: datetime, end: datetime
    ) -> List[RouteOutcome]:
        """Fetch completed routes for period"""
        # Implementation depends on your database structure
        # Placeholder for now
        return []

    def _empty_daily_analytics(self, date: datetime) -> DailyAnalytics:
        """Empty analytics when no data"""
        return DailyAnalytics(
            date=date,
            total_routes=0,
            total_shipments=0,
            avg_utilization=0.0,
            sla_compliance_rate=0.0,
            cost_prediction_error=0.0,
            duration_prediction_error=0.0,
            function_class_performance={},
            top_delay_causes=[],
        )

    def _empty_weekly_insights(self, week_start: datetime) -> WeeklyInsights:
        """Empty insights when no data"""
        return WeeklyInsights(
            week_start=week_start,
            route_patterns_discovered=[],
            optimal_consolidation_windows={},
            fleet_utilization_by_day={},
            recommended_fleet_adjustments=[],
            network_topology_updates={},
            customer_pattern_insights=[],
        )

    def _analyze_function_class_performance(
        self, routes: List[RouteOutcome]
    ) -> Dict[str, float]:
        """Analyze which function class performed best"""
        performance = defaultdict(list)

        for route in routes:
            fc = route.initial_state.get("function_class", "cfa")
            reward = -route.actual_cost
            if not route.sla_compliance:
                reward -= self.config.business_config.get("sla_penalty_per_hour", 1000)
            performance[fc].append(reward)

        return {fc: np.mean(rewards) for fc, rewards in performance.items() if rewards}

    def _analyze_delay_causes(self, delays: List[Dict]) -> List[str]:
        """Identify top delay causes"""
        if not delays:
            return []

        cause_counter = Counter(delay.get("cause", "unknown") for delay in delays)
        return [cause for cause, _ in cause_counter.most_common(5)]

    def _discover_route_patterns(self, routes: List[RouteOutcome]) -> List[Dict]:
        """Discover common route patterns"""
        # Simplified implementation
        return []

    def _optimize_consolidation_windows(
        self, routes: List[RouteOutcome]
    ) -> Dict[str, float]:
        """Find optimal consolidation time windows"""
        # Simplified implementation
        return {"morning": 2.0, "afternoon": 1.5, "evening": 3.0}

    def _analyze_utilization_by_day(
        self, routes: List[RouteOutcome]
    ) -> Dict[str, float]:
        """Analyze utilization by day of week"""
        by_day = defaultdict(list)

        for route in routes:
            day = route.completed_at.strftime("%A")
            by_day[day].append(route.utilization)

        return {day: np.mean(utils) for day, utils in by_day.items() if utils}

    def _generate_fleet_recommendations(
        self, utilization_by_day: Dict[str, float]
    ) -> List[str]:
        """Generate fleet adjustment recommendations"""
        recommendations = []

        for day, util in utilization_by_day.items():
            if util > 0.95:
                recommendations.append(
                    f"Consider adding capacity on {day} (util={util:.1%})"
                )
            elif util < 0.60:
                recommendations.append(
                    f"Consider reducing capacity on {day} (util={util:.1%})"
                )

        return recommendations
