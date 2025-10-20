# src/core/vfa.py

"""
Value Function Approximation (VFA): Learn long-term value of states
Uses Temporal Difference learning to update value function weights
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json

from .state_manager import StateManager, SystemState, Shipment, VehicleState
from ..config.senga_config import SengaConfigurator

@dataclass
class ValueEstimate:
    """Value estimate with uncertainty"""
    value: float
    confidence: float  # 0-1, based on prediction variance
    features: np.ndarray

class ValueFunctionApproximator:
    """
    VFA: Learn V(S_t | θ) = θ^T · φ(S_t)
    
    Linear value function with hand-crafted features
    Updated via Temporal Difference learning
    """
    
    def __init__(self, weights_path: str = "data/vfa_weights.json"):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.weights_path = Path(weights_path)
        
        # Learning parameters
        self.learning_rate = self.config.vfa_learning_rate
        self.discount_factor = self.config.get('vfa.learning.discount_factor', 0.95)
        self.epsilon = self.config.get('vfa.exploration.initial_epsilon', 0.2)
        
        # Value function weights
        self.weights: Optional[np.ndarray] = None
        self.num_features = self._get_num_features()
        
        # Learning statistics
        self.num_updates = 0
        self.recent_errors = []
        self.max_recent_errors = 1000
        
        # Initialize or load weights
        self._initialize_weights()
    
    def _get_num_features(self) -> int:
        """
        Get number of features in feature vector
        
        Features (approximately 25-30):
        - Time features: 5
        - Consolidation features: 8
        - Fleet features: 6
        - Outcome features: 6
        """
        return 25
    
    def _initialize_weights(self):
        """Initialize weights from config or load from file"""
        if self.weights_path.exists():
            self._load_weights()
        else:
            # Initialize with domain knowledge
            self.weights = np.zeros(self.num_features)
            
            if self.config.get('vfa.bootstrap.use_domain_initialization', True):
                # Positive weights for good things
                self.weights[5] = 5.0   # High utilization is good
                self.weights[10] = 2.0  # Consolidation opportunities are good
                
                # Negative weights for bad things
                self.weights[2] = -3.0  # Time pressure is bad
                
            self._save_weights()
    
    def extract_features(self, state: SystemState) -> np.ndarray:
        """
        Extract feature vector φ(S_t) from state
        
        Feature groups:
        1. Time-based (5 features)
        2. Consolidation (8 features)
        3. Fleet (6 features)
        4. Outcome (6 features)
        """
        features = []
        
        # === Time-based features (5) ===
        
        # 1. Hour of day (normalized)
        hour_of_day = state.timestamp.hour / 24.0
        features.append(hour_of_day)
        
        # 2. Day of week (normalized, Monday=0)
        day_of_week = state.timestamp.weekday() / 7.0
        features.append(day_of_week)
        
        # 3. Average time pressure across pending shipments
        if state.pending_shipments:
            avg_time_pressure = np.mean([
                s.time_pressure(state.timestamp) 
                for s in state.pending_shipments
            ])
        else:
            avg_time_pressure = 0.0
        features.append(avg_time_pressure)
        
        # 4. Fraction of SLA consumed (most urgent shipment)
        if state.pending_shipments:
            max_time_pressure = max(
                s.time_pressure(state.timestamp)
                for s in state.pending_shipments
            )
        else:
            max_time_pressure = 0.0
        features.append(max_time_pressure)
        
        # 5. Hours until next deadline
        if state.pending_shipments:
            min_time_to_deadline = min(
                s.time_to_deadline(state.timestamp).total_seconds() / 3600
                for s in state.pending_shipments
            )
            normalized_time = min_time_to_deadline / 48.0  # Normalize by SLA
        else:
            normalized_time = 1.0
        features.append(normalized_time)
        
        # === Consolidation features (8) ===
        
        # 6. Number of pending shipments (normalized)
        num_pending = len(state.pending_shipments) / 20.0  # Assume max 20
        features.append(min(num_pending, 1.0))
        
        # 7. Average shipments per pending batch (consolidation potential)
        # Estimate: group by origin zone
        if state.pending_shipments:
            origin_zones = {}
            for s in state.pending_shipments:
                zone = s.origin.zone_id or 'unknown'
                origin_zones[zone] = origin_zones.get(zone, 0) + 1
            
            if origin_zones:
                avg_per_zone = np.mean(list(origin_zones.values()))
            else:
                avg_per_zone = 1.0
        else:
            avg_per_zone = 0.0
        features.append(avg_per_zone / 5.0)  # Normalize
        
        # 8. Spatial density (shipments per geographic cluster)
        spatial_density = self._calculate_spatial_density(state.pending_shipments)
        features.append(spatial_density)
        
        # 9. Total pending volume (normalized by fleet capacity)
        if state.pending_shipments and state.fleet_state:
            total_volume = sum(s.volume for s in state.pending_shipments)
            total_capacity = sum(v.capacity.volume for v in state.fleet_state)
            volume_ratio = total_volume / total_capacity if total_capacity > 0 else 0.0
        else:
            volume_ratio = 0.0
        features.append(min(volume_ratio, 2.0))  # Cap at 2x capacity
        
        # 10. Total pending weight (normalized)
        if state.pending_shipments and state.fleet_state:
            total_weight = sum(s.weight for s in state.pending_shipments)
            total_capacity = sum(v.capacity.weight for v in state.fleet_state)
            weight_ratio = total_weight / total_capacity if total_capacity > 0 else 0.0
        else:
            weight_ratio = 0.0
        features.append(min(weight_ratio, 2.0))
        
        # 11. Destination clustering (how clustered are destinations)
        dest_clustering = self._calculate_destination_clustering(state.pending_shipments)
        features.append(dest_clustering)
        
        # 12. Expected new arrivals (learned arrival rate)
        expected_arrivals = self._estimate_arrival_rate(state.timestamp)
        features.append(expected_arrivals / 10.0)  # Normalize
        
        # 13. Consolidation window remaining
        if state.pending_shipments:
            max_wait = self.config.get('max_consolidation_wait_hours', 24)
            min_time_remaining = min(
                s.time_to_deadline(state.timestamp).total_seconds() / 3600
                for s in state.pending_shipments
            )
            consolidation_window = min(min_time_remaining / max_wait, 1.0)
        else:
            consolidation_window = 1.0
        features.append(consolidation_window)
        
        # === Fleet features (6) ===
        
        # 14. Fraction of vehicles idle
        if state.fleet_state:
            idle_fraction = sum(
                1 for v in state.fleet_state
                if v.is_available(state.timestamp)
            ) / len(state.fleet_state)
        else:
            idle_fraction = 0.0
        features.append(idle_fraction)
        
        # 15. Fraction of vehicles en route
        if state.fleet_state:
            enroute_fraction = sum(
                1 for v in state.fleet_state
                if v.status.value == 'en_route'
            ) / len(state.fleet_state)
        else:
            enroute_fraction = 0.0
        features.append(enroute_fraction)
        
        # 16. Fleet capacity available (volume)
        if state.fleet_state:
            available_vehicles = state.get_available_vehicles()
            if available_vehicles:
                available_capacity = sum(v.capacity.volume for v in available_vehicles)
                total_capacity = sum(v.capacity.volume for v in state.fleet_state)
                capacity_ratio = available_capacity / total_capacity
            else:
                capacity_ratio = 0.0
        else:
            capacity_ratio = 0.0
        features.append(capacity_ratio)
        
        # 17. Number of active routes (normalized)
        num_active_routes = len(state.active_routes) / 10.0  # Assume max 10
        features.append(min(num_active_routes, 1.0))
        
        # 18. Average vehicle utilization (current hour estimate)
        # Based on active routes
        if state.active_routes and state.fleet_state:
            utilization_estimate = len(state.active_routes) / len(state.fleet_state)
        else:
            utilization_estimate = 0.0
        features.append(min(utilization_estimate, 1.0))
        
        # 19. Fleet geographic spread
        fleet_spread = self._calculate_fleet_spread(state.fleet_state)
        features.append(fleet_spread)
        
        # === Outcome features (6) - learned from historical data ===
        
        # 20. Recent utilization trend
        recent_util = self._get_recent_utilization_trend()
        features.append(recent_util)
        
        # 21. Recent on-time performance
        recent_ontime = self._get_recent_ontime_rate()
        features.append(recent_ontime)
        
        # 22. Recent cost efficiency
        recent_cost = self._get_recent_cost_efficiency()
        features.append(recent_cost)
        
        # 23. Recent delay penalties (normalized)
        recent_delays = self._get_recent_delay_penalties()
        features.append(min(recent_delays / 10000.0, 1.0))  # Normalize by 10k
        
        # 24. Customer satisfaction proxy
        customer_satisfaction = (recent_ontime * 0.7 + (1 - recent_delays / 10000.0) * 0.3)
        features.append(customer_satisfaction)
        
        # 25. Decision cycle efficiency (time since last decision)
        time_since_decision = self._time_since_last_decision(state.timestamp)
        features.append(min(time_since_decision / 2.0, 1.0))  # Normalize by 2 hours
        
        return np.array(features)
    
    def evaluate(self, state: SystemState) -> ValueEstimate:
        """
        Evaluate V(S_t | θ) = θ^T · φ(S_t)
        
        Returns: ValueEstimate with value and confidence
        """
        features = self.extract_features(state)
        value = np.dot(self.weights, features)
        
        # Confidence based on recent prediction errors
        if len(self.recent_errors) > 10:
            error_std = np.std(self.recent_errors[-100:])
            confidence = 1.0 / (1.0 + error_std)  # Higher error = lower confidence
        else:
            confidence = 0.5  # Low confidence initially
        
        return ValueEstimate(
            value=value,
            confidence=min(confidence, 1.0),
            features=features
        )
    
    def update(self, state: SystemState, action: dict,
              reward: float, next_state: SystemState):
        """
        TD update: θ ← θ + α·(R + γ·V(S') - V(S))·φ(S)
        
        Args:
            state: State before action
            action: Action taken
            reward: Immediate reward observed
            next_state: State after action
        """
        
        # Compute current value estimate
        current_estimate = self.evaluate(state)
        current_value = current_estimate.value
        current_features = current_estimate.features
        
        # Compute next state value
        next_estimate = self.evaluate(next_state)
        next_value = next_estimate.value
        
        # TD target
        td_target = reward + self.discount_factor * next_value
        
        # TD error
        td_error = td_target - current_value
        
        # Update weights
        self.weights += self.learning_rate * td_error * current_features
        
        # Track error for confidence estimation
        self.recent_errors.append(abs(td_error))
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        # Update statistics
        self.num_updates += 1
        
        # Decay learning rate
        if self.num_updates % 1000 == 0:
            decay = self.config.get('vfa.learning.learning_rate_decay', 0.9995)
            min_lr = self.config.get('vfa.learning.min_learning_rate', 0.001)
            self.learning_rate = max(self.learning_rate * decay, min_lr)
        
        # Periodically save weights
        if self.num_updates % 100 == 0:
            self._save_weights()
    
    def _calculate_spatial_density(self, shipments: List[Shipment]) -> float:
        """Calculate spatial density of shipments"""
        if len(shipments) < 2:
            return 0.0
        
        # Calculate average distance between shipments
        locations = [s.origin for s in shipments]
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                dist = self._haversine_distance(locations[i], locations[j])
                total_distance += dist
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        
        # Normalize: closer shipments = higher density
        # Assume max relevant distance is 100km
        density = 1.0 - min(avg_distance / 100.0, 1.0)
        return density
    
    def _calculate_destination_clustering(self, shipments: List[Shipment]) -> float:
        """Calculate how clustered destinations are"""
        if not shipments:
            return 0.0
        
        all_destinations = []
        for s in shipments:
            all_destinations.extend(s.destinations)
        
        if len(all_destinations) < 2:
            return 0.0
        
        # Group by zone
        zone_counts = {}
        for dest in all_destinations:
            zone = dest.zone_id or 'unknown'
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        # High clustering = few zones with many destinations each
        num_zones = len(zone_counts)
        avg_per_zone = len(all_destinations) / num_zones
        
        # Normalize
        clustering = min(avg_per_zone / 5.0, 1.0)
        return clustering
    
    def _calculate_fleet_spread(self, fleet: List[VehicleState]) -> float:
        """Calculate geographic spread of fleet"""
        if len(fleet) < 2:
            return 0.0
        
        locations = [v.current_location for v in fleet]
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                dist = self._haversine_distance(locations[i], locations[j])
                total_distance += dist
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        
        # Normalize by 100km
        spread = min(avg_distance / 100.0, 1.0)
        return spread
    
    def _haversine_distance(self, loc1, loc2) -> float:
        """Calculate haversine distance in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371
        
        lat1, lon1 = radians(loc1.lat), radians(loc1.lng)
        lat2, lon2 = radians(loc2.lat), radians(loc2.lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _estimate_arrival_rate(self, timestamp: datetime) -> float:
        """Estimate expected shipment arrivals (learned from history)"""
        # Simple time-of-day based estimate
        # In production, this would use historical arrival patterns
        
        hour = timestamp.hour
        
        # Business hours (9am-5pm) have higher arrival rates
        if 9 <= hour <= 17:
            return 2.0  # Expected 2 shipments per hour
        else:
            return 0.5  # Lower rate outside business hours
    
    def _get_recent_utilization_trend(self) -> float:
        """Get recent utilization from historical data"""
        stats = self.state_manager.get_utilization_stats(days=7)
        return stats['avg_utilization']
    
    def _get_recent_ontime_rate(self) -> float:
        """Get recent on-time delivery rate"""
        return self.state_manager.get_on_time_performance(days=7)
    
    def _get_recent_cost_efficiency(self) -> float:
        """Get recent cost efficiency"""
        stats = self.state_manager.get_cost_efficiency(days=7)
        if stats['num_routes'] == 0:
            return 0.5
        
        # Normalize: lower cost is better
        # Assume typical cost is around 5000 KES per route
        normalized_cost = 1.0 - min(stats['avg_actual_cost'] / 10000.0, 1.0)
        return max(normalized_cost, 0.0)
    
    def _get_recent_delay_penalties(self) -> float:
        """Get recent delay penalties"""
        stats = self.state_manager.get_cost_efficiency(days=7)
        return stats['total_delay_penalties']
    
    def _time_since_last_decision(self, timestamp: datetime) -> float:
        """Time since last decision in hours"""
        decisions = self.state_manager.get_decision_history(limit=1)
        if not decisions:
            return 0.0
        
        last_decision = decisions[0]
        delta = timestamp - last_decision.timestamp
        return delta.total_seconds() / 3600.0
    
    def _save_weights(self):
        """Save weights to file"""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'weights': self.weights.tolist(),
            'num_updates': self.num_updates,
            'learning_rate': self.learning_rate,
            'recent_avg_error': np.mean(self.recent_errors[-100:]) if self.recent_errors else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.weights_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_weights(self):
        """Load weights from file"""
        with open(self.weights_path, 'r') as f:
            data = json.load(f)
        
        self.weights = np.array(data['weights'])
        self.num_updates = data.get('num_updates', 0)
        self.learning_rate = data.get('learning_rate', self.config.vfa_learning_rate)