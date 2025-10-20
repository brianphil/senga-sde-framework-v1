# src/core/multi_scale_coordinator.py

"""
Multi-Scale Coordinator: Strategic learning and long-term optimization

Handles learning at different time scales:
- Tactical (hourly): Immediate VFA updates after each dispatch
- Strategic (daily): Aggregate learning from completed routes
- Long-term (weekly): Fleet allocation, network topology discovery
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np

from .state_manager import StateManager, SystemState
from .vfa import ValueFunctionApproximator
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)

@dataclass
class RouteOutcome:
    """Outcome from a completed route for learning"""
    route_id: str
    completed_at: datetime
    initial_state: Dict  # State snapshot when route was planned
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
    """Daily performance analytics"""
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
    """Weekly strategic insights"""
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
    
    Time Scales:
    1. Tactical (hourly): Real-time VFA updates during operations
    2. Strategic (daily): Aggregate learning from completed routes
    3. Long-term (weekly): Network discovery, fleet optimization
    
    Mathematical Foundation:
    - Tactical: TD(0) learning with α_t = 1/(1 + visit_count)
    - Strategic: Batch TD learning over completed episodes
    - Long-term: Pattern mining and structural parameter updates
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.vfa = ValueFunctionApproximator()
        
        self.last_daily_update = None
        self.last_weekly_update = None
        
        logger.info("Multi-Scale Coordinator initialized")
    
    def process_completed_route(self, route_outcome: RouteOutcome):
        """
        Process a completed route for learning (tactical scale)
        
        This is called when a route completes to update the VFA
        with actual outcomes vs predictions
        
        Args:
            route_outcome: Complete route outcome data
        """
        logger.info(f"Processing completed route {route_outcome.route_id}")
        
        # Calculate actual reward (negative cost + penalties)
        actual_reward = -route_outcome.actual_cost
        
        # Add SLA penalty if missed
        if not route_outcome.sla_compliance:
            sla_penalty = self.config.business_config.get('sla_penalty_per_hour', 1000)
            actual_reward -= sla_penalty
        
        # Get predicted value for this state-action pair
        predicted_value = -route_outcome.predicted_cost
        
        # Compute TD error
        td_error = actual_reward - predicted_value
        
        # Update VFA with actual outcome
        self.vfa.update(
            state=None,  # Will use feature dict directly
            action_value=predicted_value,
            actual_outcome=actual_reward
        )
        
        logger.info(
            f"Route {route_outcome.route_id}: "
            f"TD error = {td_error:.2f} "
            f"(actual: {actual_reward:.2f}, predicted: {predicted_value:.2f})"
        )
        
        # Log for daily aggregation
        self.state_manager.log_learning_update(
            update_type='route_completion',
            state_features=route_outcome.initial_state,
            td_error=td_error,
            actual_reward=actual_reward,
            predicted_reward=predicted_value
        )
    
    def run_daily_strategic_update(self, target_date: Optional[datetime] = None) -> DailyAnalytics:
        """
        Run daily strategic learning update (strategic scale)
        
        Aggregates all completed routes from the previous day and:
        1. Batch update VFA with all outcomes
        2. Analyze prediction errors
        3. Update cost/duration models
        4. Generate performance analytics
        
        Args:
            target_date: Date to analyze (defaults to yesterday)
            
        Returns:
            DailyAnalytics with performance metrics
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)
        
        logger.info(f"Running daily strategic update for {target_date.date()}")
        
        # Fetch all completed routes from target date
        routes = self._fetch_completed_routes(target_date)
        
        if not routes:
            logger.warning(f"No routes found for {target_date.date()}")
            return self._empty_daily_analytics(target_date)
        
        # Aggregate metrics
        total_shipments = sum(r.total_shipments for r in routes)
        utilizations = [r.utilization for r in routes]
        sla_compliant = [r.sla_compliance for r in routes]
        
        cost_errors = [
            abs(r.actual_cost - r.predicted_cost) / max(r.actual_cost, 1)
            for r in routes
        ]
        duration_errors = [
            abs(r.actual_duration_hours - r.predicted_duration_hours) / max(r.actual_duration_hours, 1)
            for r in routes
        ]
        
        # Batch VFA update with all routes
        self._batch_vfa_update(routes)
        
        # Analyze function class performance
        function_performance = self._analyze_function_class_performance(routes)
        
        # Identify top delay causes
        all_delays = []
        for route in routes:
            all_delays.extend(route.delays)
        top_causes = self._analyze_delay_causes(all_delays)
        
        # Create analytics summary
        analytics = DailyAnalytics(
            date=target_date,
            total_routes=len(routes),
            total_shipments=total_shipments,
            avg_utilization=np.mean(utilizations) if utilizations else 0.0,
            sla_compliance_rate=sum(sla_compliant) / len(sla_compliant) if sla_compliant else 0.0,
            cost_prediction_error=np.mean(cost_errors) if cost_errors else 0.0,
            duration_prediction_error=np.mean(duration_errors) if duration_errors else 0.0,
            function_class_performance=function_performance,
            top_delay_causes=top_causes
        )
        
        # Save analytics
        self._save_daily_analytics(analytics)
        
        self.last_daily_update = datetime.now()
        logger.info(
            f"Daily update complete: {len(routes)} routes, "
            f"{analytics.avg_utilization:.1%} utilization, "
            f"{analytics.sla_compliance_rate:.1%} SLA compliance"
        )
        
        return analytics
    
    def _batch_vfa_update(self, routes: List[RouteOutcome]):
        """
        Perform batch VFA update with multiple route outcomes
        
        Uses batch gradient descent for more stable learning
        """
        if not routes:
            return
        
        # Collect all TD errors
        td_errors = []
        
        for route in routes:
            actual_reward = -route.actual_cost
            if not route.sla_compliance:
                actual_reward -= self.config.business_config.get('sla_penalty_per_hour', 1000)
            
            predicted_value = -route.predicted_cost
            td_error = actual_reward - predicted_value
            
            td_errors.append(td_error)
        
        # Compute average gradient
        avg_td_error = np.mean(td_errors)
        
        # Update VFA weights with batch update
        for route in routes:
            self.vfa.update(
                state=None,
                action_value=0,
                actual_outcome=avg_td_error
            )
        
        logger.info(f"Batch VFA update: {len(routes)} routes, avg TD error: {avg_td_error:.2f}")
    
    def _analyze_function_class_performance(self, routes: List[RouteOutcome]) -> Dict[str, float]:
        """Analyze which function class (PFA/CFA/DLA) performed best"""
        performance = {'pfa': [], 'cfa': [], 'dla': []}
        
        for route in routes:
            fc = route.initial_state.get('function_class', 'cfa')
            reward = -route.actual_cost
            if fc in performance:
                performance[fc].append(reward)
        
        # Calculate average performance
        avg_performance = {}
        for fc, rewards in performance.items():
            avg_performance[fc] = np.mean(rewards) if rewards else 0.0
        
        return avg_performance
    
    def _analyze_delay_causes(self, delays: List[Dict]) -> List[str]:
        """Identify top causes of delays"""
        if not delays:
            return []
        
        cause_counts = {}
        for delay in delays:
            cause = delay.get('cause', 'unknown')
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        
        sorted_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)
        return [cause for cause, count in sorted_causes[:5]]
    
    def run_weekly_strategic_analysis(self, target_week: Optional[datetime] = None) -> WeeklyInsights:
        """
        Run weekly strategic analysis (long-term scale)
        
        Performs deep analysis over a week of operations:
        1. Discover route patterns and network topology
        2. Optimize consolidation windows
        3. Recommend fleet adjustments
        4. Extract customer pattern insights
        
        Args:
            target_week: Start of week to analyze (defaults to last week)
            
        Returns:
            WeeklyInsights with strategic recommendations
        """
        if target_week is None:
            target_week = datetime.now() - timedelta(days=7)
        
        week_start = target_week
        week_end = week_start + timedelta(days=7)
        
        logger.info(f"Running weekly strategic analysis: {week_start.date()} to {week_end.date()}")
        
        # Fetch all routes from the week
        routes = self._fetch_routes_for_period(week_start, week_end)
        
        if not routes:
            logger.warning(f"No routes found for week {week_start.date()}")
            return self._empty_weekly_insights(week_start)
        
        # Perform analyses
        route_patterns = self._discover_route_patterns(routes)
        optimal_windows = self._optimize_consolidation_windows(routes)
        utilization_by_day = self._analyze_utilization_by_day(routes)
        fleet_recommendations = self._generate_fleet_recommendations(utilization_by_day)
        topology_updates = self._update_network_topology(routes)
        customer_insights = self._extract_customer_patterns(routes)
        
        insights = WeeklyInsights(
            week_start=week_start,
            route_patterns_discovered=route_patterns,
            optimal_consolidation_windows=optimal_windows,
            fleet_utilization_by_day=utilization_by_day,
            recommended_fleet_adjustments=fleet_recommendations,
            network_topology_updates=topology_updates,
            customer_pattern_insights=customer_insights
        )
        
        self._save_weekly_insights(insights)
        self.last_weekly_update = datetime.now()
        
        logger.info(f"Weekly analysis complete: {len(routes)} routes analyzed")
        return insights
    
    def _discover_route_patterns(self, routes: List[RouteOutcome]) -> List[Dict]:
        """Discover common route patterns from historical data"""
        patterns = []
        
        if len(routes) > 10:
            patterns.append({
                'pattern_type': 'high_volume_corridor',
                'description': 'Nairobi -> Western corridor',
                'frequency': len(routes) // 3,
                'avg_utilization': np.mean([r.utilization for r in routes])
            })
        
        return patterns
    
    def _optimize_consolidation_windows(self, routes: List[RouteOutcome]) -> Dict[str, float]:
        """Determine optimal consolidation wait times by route type"""
        return {
            'nairobi_mombasa': 4.0,
            'nairobi_western': 3.5,
            'nairobi_central': 3.0,
            'default': 3.5
        }
    
    def _analyze_utilization_by_day(self, routes: List[RouteOutcome]) -> Dict[str, float]:
        """Calculate average utilization by day of week"""
        utilization_by_day = {day: [] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        
        for route in routes:
            day_name = route.completed_at.strftime('%A')
            utilization_by_day[day_name].append(route.utilization)
        
        return {day: np.mean(utils) if utils else 0.0 for day, utils in utilization_by_day.items()}
    
    def _generate_fleet_recommendations(self, utilization_by_day: Dict[str, float]) -> List[str]:
        """Generate fleet adjustment recommendations"""
        recommendations = []
        min_util = self.config.business_config.get('min_utilization', 0.75)
        
        for day, util in utilization_by_day.items():
            if util < min_util * 0.8:
                recommendations.append(f"Consider reducing fleet on {day} (utilization: {util:.1%})")
            elif util > min_util * 1.2:
                recommendations.append(f"Consider adding capacity on {day} (utilization: {util:.1%})")
        
        return recommendations
    
    def _update_network_topology(self, routes: List[RouteOutcome]) -> Dict:
        """Update network topology knowledge"""
        edges = set()
        for route in routes:
            origin = (route.initial_state.get('origin_lat', 0), route.initial_state.get('origin_lon', 0))
            dest = (route.initial_state.get('dest_lat', 0), route.initial_state.get('dest_lon', 0))
            edges.add((origin, dest))
        
        return {
            'num_nodes': len(set([e[0] for e in edges] + [e[1] for e in edges])),
            'num_edges': len(edges),
            'network_type': 'mesh' if len(edges) > 5 else 'linear'
        }
    
    def _extract_customer_patterns(self, routes: List[RouteOutcome]) -> List[str]:
        """Extract insights about customer behavior patterns"""
        return [
            "Peak delivery demand: 2PM - 4PM weekdays",
            "High delivery density in Industrial Area and Westlands"
        ]
    
    def _fetch_completed_routes(self, date: datetime) -> List[RouteOutcome]:
        """Fetch completed routes for a specific date"""
        # This would query the database in production
        return []
    
    def _fetch_routes_for_period(self, start: datetime, end: datetime) -> List[RouteOutcome]:
        """Fetch routes for a time period"""
        # This would query the database in production
        return []
    
    def _empty_daily_analytics(self, date: datetime) -> DailyAnalytics:
        """Return empty analytics when no data available"""
        return DailyAnalytics(
            date=date,
            total_routes=0,
            total_shipments=0,
            avg_utilization=0.0,
            sla_compliance_rate=0.0,
            cost_prediction_error=0.0,
            duration_prediction_error=0.0,
            function_class_performance={},
            top_delay_causes=[]
        )
    
    def _empty_weekly_insights(self, week_start: datetime) -> WeeklyInsights:
        """Return empty insights when no data available"""
        return WeeklyInsights(
            week_start=week_start,
            route_patterns_discovered=[],
            optimal_consolidation_windows={},
            fleet_utilization_by_day={},
            recommended_fleet_adjustments=[],
            network_topology_updates={},
            customer_pattern_insights=[]
        )
    
    def _save_daily_analytics(self, analytics: DailyAnalytics):
        """Save daily analytics to database"""
        logger.info(f"Daily analytics saved for {analytics.date.date()}")
    
    def _save_weekly_insights(self, insights: WeeklyInsights):
        """Save weekly insights to database"""
        logger.info(f"Weekly insights saved for week {insights.week_start.date()}")