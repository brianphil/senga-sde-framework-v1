# src/core/state_manager_extensions.py

"""
StateManager extensions for Week 5 multi-scale learning
Adds route outcome storage and retrieval
"""

from typing import List
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StateManagerExtensions:
    """Extensions to StateManager for route outcome tracking"""
    
    def __init__(self, db_path: str = "data/senga_state.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_outcome_tables()
    
    def _init_outcome_tables(self):
        """Initialize route outcome tables"""
        
        # Route outcomes table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS route_outcomes (
                route_id TEXT PRIMARY KEY,
                completed_at TIMESTAMP NOT NULL,
                initial_state TEXT NOT NULL,
                shipments_delivered INTEGER NOT NULL,
                total_shipments INTEGER NOT NULL,
                actual_cost REAL NOT NULL,
                predicted_cost REAL NOT NULL,
                actual_duration_hours REAL NOT NULL,
                predicted_duration_hours REAL NOT NULL,
                utilization REAL NOT NULL,
                sla_compliance INTEGER NOT NULL,
                delays TEXT,
                issues TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index - fix: check column exists first
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_route_outcomes_date 
                ON route_outcomes(completed_at)
            """)
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not create index on completed_at: {e}")
        
        # Daily analytics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_analytics (
                date TEXT PRIMARY KEY,
                total_routes INTEGER,
                total_shipments INTEGER,
                avg_utilization REAL,
                sla_compliance_rate REAL,
                cost_prediction_error REAL,
                duration_prediction_error REAL,
                function_class_performance TEXT,
                top_delay_causes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Weekly insights table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_insights (
                week_start TEXT PRIMARY KEY,
                route_patterns TEXT,
                consolidation_windows TEXT,
                utilization_by_day TEXT,
                fleet_recommendations TEXT,
                topology_updates TEXT,
                customer_insights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info("Route outcome tables initialized")
    
    def save_route_outcome(self, outcome):
        """Save completed route outcome"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO route_outcomes
                (route_id, completed_at, initial_state, shipments_delivered, total_shipments,
                 actual_cost, predicted_cost, actual_duration_hours, predicted_duration_hours,
                 utilization, sla_compliance, delays, issues)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.route_id,
                outcome.completed_at.isoformat(),
                json.dumps(outcome.initial_state),
                outcome.shipments_delivered,
                outcome.total_shipments,
                outcome.actual_cost,
                outcome.predicted_cost,
                outcome.actual_duration_hours,
                outcome.predicted_duration_hours,
                outcome.utilization,
                1 if outcome.sla_compliance else 0,
                json.dumps(outcome.delays),
                json.dumps(outcome.issues)
            ))
            self.conn.commit()
            logger.info(f"Saved route outcome: {outcome.route_id}")
        except Exception as e:
            logger.error(f"Failed to save route outcome: {e}")
            self.conn.rollback()
    
    def get_completed_routes(self, date: datetime) -> List:
        """Get all routes completed on a date"""
        from .multi_scale_coordinator import RouteOutcome
        
        try:
            cursor = self.conn.execute("""
                SELECT route_id, completed_at, initial_state, shipments_delivered, 
                       total_shipments, actual_cost, predicted_cost, actual_duration_hours,
                       predicted_duration_hours, utilization, sla_compliance, delays, issues
                FROM route_outcomes
                WHERE DATE(completed_at) = DATE(?)
            """, (date.isoformat(),))
            
            routes = []
            for row in cursor.fetchall():
                routes.append(RouteOutcome(
                    route_id=row[0],
                    completed_at=datetime.fromisoformat(row[1]),
                    initial_state=json.loads(row[2]),
                    shipments_delivered=row[3],
                    total_shipments=row[4],
                    actual_cost=row[5],
                    predicted_cost=row[6],
                    actual_duration_hours=row[7],
                    predicted_duration_hours=row[8],
                    utilization=row[9],
                    sla_compliance=bool(row[10]),
                    delays=json.loads(row[11]) if row[11] else [],
                    issues=json.loads(row[12]) if row[12] else []
                ))
            
            return routes
        except Exception as e:
            logger.error(f"Failed to get completed routes: {e}")
            return []
    
    def get_routes_for_period(self, start: datetime, end: datetime) -> List:
        """Get all routes in time period"""
        from .multi_scale_coordinator import RouteOutcome
        
        try:
            cursor = self.conn.execute("""
                SELECT route_id, completed_at, initial_state, shipments_delivered,
                       total_shipments, actual_cost, predicted_cost, actual_duration_hours,
                       predicted_duration_hours, utilization, sla_compliance, delays, issues
                FROM route_outcomes
                WHERE completed_at BETWEEN ? AND ?
            """, (start.isoformat(), end.isoformat()))
            
            routes = []
            for row in cursor.fetchall():
                routes.append(RouteOutcome(
                    route_id=row[0],
                    completed_at=datetime.fromisoformat(row[1]),
                    initial_state=json.loads(row[2]),
                    shipments_delivered=row[3],
                    total_shipments=row[4],
                    actual_cost=row[5],
                    predicted_cost=row[6],
                    actual_duration_hours=row[7],
                    predicted_duration_hours=row[8],
                    utilization=row[9],
                    sla_compliance=bool(row[10]),
                    delays=json.loads(row[11]) if row[11] else [],
                    issues=json.loads(row[12]) if row[12] else []
                ))
            
            return routes
        except Exception as e:
            logger.error(f"Failed to get routes for period: {e}")
            return []
    
    def save_daily_analytics(self, analytics):
        """Save daily analytics"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO daily_analytics
                (date, total_routes, total_shipments, avg_utilization, sla_compliance_rate,
                 cost_prediction_error, duration_prediction_error, function_class_performance,
                 top_delay_causes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analytics.date.date().isoformat(),
                analytics.total_routes,
                analytics.total_shipments,
                analytics.avg_utilization,
                analytics.sla_compliance_rate,
                analytics.cost_prediction_error,
                analytics.duration_prediction_error,
                json.dumps(analytics.function_class_performance),
                json.dumps(analytics.top_delay_causes)
            ))
            self.conn.commit()
            logger.info(f"Saved daily analytics for {analytics.date.date()}")
        except Exception as e:
            logger.error(f"Failed to save daily analytics: {e}")
            self.conn.rollback()
    
    def save_weekly_insights(self, insights):
        """Save weekly insights"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO weekly_insights
                (week_start, route_patterns, consolidation_windows, utilization_by_day,
                 fleet_recommendations, topology_updates, customer_insights)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                insights.week_start.date().isoformat(),
                json.dumps(insights.route_patterns_discovered),
                json.dumps(insights.optimal_consolidation_windows),
                json.dumps(insights.fleet_utilization_by_day),
                json.dumps(insights.recommended_fleet_adjustments),
                json.dumps(insights.network_topology_updates),
                json.dumps(insights.customer_pattern_insights)
            ))
            self.conn.commit()
            logger.info(f"Saved weekly insights for week {insights.week_start.date()}")
        except Exception as e:
            logger.error(f"Failed to save weekly insights: {e}")
            self.conn.rollback()