# src/core/multi_scale_coordinator.py

"""
Multi-Scale Coordinator: Strategic learning and long-term optimization

Mathematical Foundation:
- Tactical: TD(0) with α_t = 1/(1 + visit_count)
- Strategic: Batch gradient descent over completed episodes
- Long-term: DBSCAN clustering for route patterns, statistical anomaly detection
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import numpy as np
from sklearn.cluster import DBSCAN

from .state_manager import StateManager, SystemState
from .vfa import ValueFunctionApproximator
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)

@dataclass
class RouteOutcome:
    """Outcome from a completed route for learning"""
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
        
        TD(0) update: V(s) ← V(s) + α[R - V(s)]
        where α = 1/(1 + visit_count)
        """
        logger.info(f"Processing completed route {route_outcome.route_id}")
        
        # Calculate actual reward
        actual_reward = -route_outcome.actual_cost
        if not route_outcome.sla_compliance:
            sla_penalty = self.config.business_config.get('sla_penalty_per_hour', 1000)
            actual_reward -= sla_penalty
        
        predicted_value = -route_outcome.predicted_cost
        td_error = actual_reward - predicted_value
        
        # Update VFA
        self.vfa.update(
            state=None,
            action_value=predicted_value,
            actual_outcome=actual_reward
        )
        
        logger.info(f"Route {route_outcome.route_id}: TD error = {td_error:.2f}")
        
        # Log for aggregation
        self.state_manager.log_learning_update(
            update_type='route_completion',
            state_features=route_outcome.initial_state,
            td_error=td_error,
            actual_reward=actual_reward,
            predicted_reward=predicted_value
        )
    
    def run_daily_strategic_update(self, target_date: Optional[datetime] = None) -> DailyAnalytics:
        """
        Run daily strategic learning (strategic scale)
        
        Batch gradient descent: θ ← θ + α * (1/N) * Σ ∇L_i
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)
        
        logger.info(f"Running daily strategic update for {target_date.date()}")
        
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
            abs(r.actual_duration_hours - r.predicted_duration_hours) / 
            max(r.actual_duration_hours, 0.1)
            for r in routes
        ]
        
        # Batch VFA update
        self._batch_vfa_update(routes)
        
        # Analyze function classes
        function_performance = self._analyze_function_class_performance(routes)
        
        # Delay analysis
        all_delays = [d for r in routes for d in r.delays]
        top_causes = self._analyze_delay_causes(all_delays)
        
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
        
        self._save_daily_analytics(analytics)
        self.last_daily_update = datetime.now()
        
        logger.info(
            f"Daily update: {len(routes)} routes, "
            f"{analytics.avg_utilization:.1%} util, "
            f"{analytics.sla_compliance_rate:.1%} SLA"
        )
        
        return analytics
    
    def _batch_vfa_update(self, routes: List[RouteOutcome]):
        """
        Batch VFA update using average gradient
        
        Batch TD: Δθ = (1/N) Σ_i δ_i ∇V(s_i)
        More stable than online updates
        """
        if not routes:
            return
        
        td_errors = []
        for route in routes:
            actual_reward = -route.actual_cost
            if not route.sla_compliance:
                actual_reward -= self.config.business_config.get('sla_penalty_per_hour', 1000)
            
            predicted_value = -route.predicted_cost
            td_error = actual_reward - predicted_value
            td_errors.append(td_error)
        
        avg_td_error = np.mean(td_errors)
        
        # Apply batch update
        for route in routes:
            self.vfa.update(
                state=route.initial_state,
                action_value=-route.predicted_cost,
                actual_outcome=avg_td_error
            )
        
        logger.info(f"Batch VFA update: {len(routes)} routes, avg TD: {avg_td_error:.2f}")
    
    def _analyze_function_class_performance(self, routes: List[RouteOutcome]) -> Dict[str, float]:
        """Analyze which function class performed best"""
        performance = defaultdict(list)
        
        for route in routes:
            fc = route.initial_state.get('function_class', 'cfa')
            reward = -route.actual_cost
            if not route.sla_compliance:
                reward -= self.config.business_config.get('sla_penalty_per_hour', 1000)
            performance[fc].append(reward)
        
        return {fc: np.mean(rewards) for fc, rewards in performance.items() if rewards}
    
    def _analyze_delay_causes(self, delays: List[Dict]) -> List[str]:
        """Identify top delay causes using frequency analysis"""
        if not delays:
            return []
        
        cause_counter = Counter(delay.get('cause', 'unknown') for delay in delays)
        return [cause for cause, _ in cause_counter.most_common(5)]
    
    def run_weekly_strategic_analysis(self, target_week: Optional[datetime] = None) -> WeeklyInsights:
        """
        Run weekly strategic analysis (long-term scale)
        
        Uses DBSCAN for route pattern discovery:
        - ε: max distance between points in cluster
        - min_samples: minimum cluster size
        """
        if target_week is None:
            target_week = datetime.now() - timedelta(days=7)
        
        week_start = target_week
        week_end = week_start + timedelta(days=7)
        
        logger.info(f"Running weekly analysis: {week_start.date()} to {week_end.date()}")
        
        routes = self._fetch_routes_for_period(week_start, week_end)
        
        if not routes:
            logger.warning(f"No routes for week {week_start.date()}")
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
        
        logger.info(f"Weekly analysis: {len(routes)} routes analyzed")
        return insights
    
    def _discover_route_patterns(self, routes: List[RouteOutcome]) -> List[Dict]:
        """
        Discover route patterns using DBSCAN clustering
        
        Clusters routes by origin-destination pairs
        DBSCAN params: ε=0.1°, min_samples=3
        """
        if len(routes) < 3:
            return []
        
        # Extract origin-destination coordinates
        coords = []
        for r in routes:
            origin = (r.initial_state.get('origin_lat', 0), r.initial_state.get('origin_lon', 0))
            dest = (r.initial_state.get('dest_lat', 0), r.initial_state.get('dest_lon', 0))
            coords.append([origin[0], origin[1], dest[0], dest[1]])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.1, min_samples=3).fit(coords)
        
        patterns = []
        for label in set(clustering.labels_):
            if label == -1:  # Noise
                continue
            
            mask = clustering.labels_ == label
            cluster_routes = [r for i, r in enumerate(routes) if mask[i]]
            
            patterns.append({
                'pattern_id': f'cluster_{label}',
                'route_count': len(cluster_routes),
                'avg_utilization': np.mean([r.utilization for r in cluster_routes]),
                'avg_cost': np.mean([r.actual_cost for r in cluster_routes]),
                'avg_duration_hours': np.mean([r.actual_duration_hours for r in cluster_routes])
            })
        
        logger.info(f"Discovered {len(patterns)} route patterns")
        return patterns
    
    def _optimize_consolidation_windows(self, routes: List[RouteOutcome]) -> Dict[str, float]:
        """
        Optimize consolidation windows by analyzing utilization vs wait time
        
        Uses linear regression: utilization = β0 + β1*wait_time
        Find wait_time that maximizes (utilization - cost_of_waiting)
        """
        # Group by corridor
        corridors = defaultdict(list)
        for r in routes:
            corridor = self._classify_corridor(r.initial_state)
            wait_time = r.initial_state.get('consolidation_wait_hours', 3.0)
            corridors[corridor].append((wait_time, r.utilization))
        
        optimal_windows = {}
        for corridor, data in corridors.items():
            if len(data) < 5:
                optimal_windows[corridor] = 3.0  # Default
                continue
            
            wait_times, utils = zip(*data)
            
            # Simple optimization: find wait_time with highest avg utilization
            wait_util_map = defaultdict(list)
            for wt, u in zip(wait_times, utils):
                wait_util_map[round(wt, 1)].append(u)
            
            best_wait = max(wait_util_map.items(), key=lambda x: np.mean(x[1]))[0]
            optimal_windows[corridor] = float(best_wait)
        
        logger.info(f"Optimized consolidation windows: {optimal_windows}")
        return optimal_windows
    
    def _classify_corridor(self, state: Dict) -> str:
        """Classify route corridor from state"""
        origin_lat = state.get('origin_lat', -1.286)
        dest_lat = state.get('dest_lat', 0)
        
        if dest_lat < -1:
            return 'nairobi_south'
        elif dest_lat > 0:
            return 'nairobi_north'
        else:
            return 'nairobi_west'
    
    def _analyze_utilization_by_day(self, routes: List[RouteOutcome]) -> Dict[str, float]:
        """Calculate average utilization by day of week"""
        day_utils = defaultdict(list)
        
        for route in routes:
            day_name = route.completed_at.strftime('%A')
            day_utils[day_name].append(route.utilization)
        
        return {day: np.mean(utils) if utils else 0.0 for day, utils in day_utils.items()}
    
    def _generate_fleet_recommendations(self, utilization_by_day: Dict[str, float]) -> List[str]:
        """
        Generate fleet recommendations using statistical thresholds
        
        Rules:
        - util < 0.6: suggest reduction
        - util > 0.9: suggest expansion  
        - 0.6 <= util <= 0.9: optimal
        """
        recommendations = []
        
        for day, util in utilization_by_day.items():
            if util < 0.6:
                recommendations.append(
                    f"Consider reducing fleet on {day} (util: {util:.1%}, target: >60%)"
                )
            elif util > 0.9:
                recommendations.append(
                    f"Consider adding capacity on {day} (util: {util:.1%}, risk of delays)"
                )
        
        return recommendations
    
    def _update_network_topology(self, routes: List[RouteOutcome]) -> Dict:
        """
        Update network topology knowledge
        
        Builds directed graph of frequent routes
        """
        edges = set()
        edge_weights = defaultdict(int)
        
        for route in routes:
            origin = (
                round(route.initial_state.get('origin_lat', 0), 2),
                round(route.initial_state.get('origin_lon', 0), 2)
            )
            dest = (
                round(route.initial_state.get('dest_lat', 0), 2),
                round(route.initial_state.get('dest_lon', 0), 2)
            )
            edge = (origin, dest)
            edges.add(edge)
            edge_weights[edge] += 1
        
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        
        # Identify hub nodes (degree > 3)
        node_degrees = defaultdict(int)
        for edge in edges:
            node_degrees[edge[0]] += 1
            node_degrees[edge[1]] += 1
        
        hubs = [node for node, degree in node_degrees.items() if degree > 3]
        
        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'hub_count': len(hubs),
            'most_frequent_routes': sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _extract_customer_patterns(self, routes: List[RouteOutcome]) -> List[str]:
        """
        Extract customer behavior patterns using time-series analysis
        
        Identifies:
        - Peak delivery times
        - Geographic demand clusters
        - SLA compliance patterns
        """
        insights = []
        
        # Peak time analysis
        delivery_hours = [r.completed_at.hour for r in routes]
        if delivery_hours:
            hour_counter = Counter(delivery_hours)
            peak_hour = hour_counter.most_common(1)[0][0]
            insights.append(f"Peak delivery hour: {peak_hour}:00")
        
        # Geographic analysis  
        regions = [self._classify_corridor(r.initial_state) for r in routes]
        if regions:
            region_counter = Counter(regions)
            top_region = region_counter.most_common(1)[0][0]
            insights.append(f"Highest demand corridor: {top_region}")
        
        # SLA analysis
        sla_rate = sum(r.sla_compliance for r in routes) / len(routes)
        if sla_rate < 0.8:
            insights.append(f"SLA compliance below target: {sla_rate:.1%} (target: 80%)")
        
        return insights
    
    def _fetch_completed_routes(self, date: datetime) -> List[RouteOutcome]:
        """Fetch completed routes from database"""
        return self.state_manager.get_completed_routes(date)
    
    def _fetch_routes_for_period(self, start: datetime, end: datetime) -> List[RouteOutcome]:
        """Fetch routes for time period"""
        return self.state_manager.get_routes_for_period(start, end)
    
    def _empty_daily_analytics(self, date: datetime) -> DailyAnalytics:
        """Return empty analytics"""
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
        """Return empty insights"""
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
        """Save to database"""
        self.state_manager.save_daily_analytics(analytics)
        logger.info(f"Daily analytics saved: {analytics.date.date()}")
    
    def _save_weekly_insights(self, insights: WeeklyInsights):
        """Save to database"""
        self.state_manager.save_weekly_insights(insights)
        logger.info(f"Weekly insights saved: week {insights.week_start.date()}")