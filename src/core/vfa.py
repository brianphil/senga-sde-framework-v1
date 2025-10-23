# src/core/vfa.py

"""
Value Function Approximation (VFA): Learn long-term value of states
Uses Temporal Difference learning with eligibility traces TD(λ)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json
import logging

from .state_manager import StateManager, SystemState, Shipment, VehicleState
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)

@dataclass
class ValueEstimate:
    """Value estimate with uncertainty"""
    value: float
    confidence: float  # 0-1, based on prediction variance
    features: np.ndarray

class ValueFunctionApproximator:
    """
    VFA: Learn V(S_t | θ) = θ^T · φ(S_t)
    
    Mathematical Foundation:
    - Linear value function: V(s) = Σ θ_i * φ_i(s)
    - TD(λ) update rule: θ ← θ + α * δ_t * e_t
      where:
        δ_t = r_t + γ*V(s_{t+1}) - V(s_t)  [TD error]
        e_t = λ*γ*e_{t-1} + ∇V(s_t)        [eligibility trace]
    
    Features (25 total):
    - Time features (5): hour, day_of_week, time_pressure, etc.
    - Consolidation features (8): pending_shipments, avg_volume, etc.
    - Fleet features (6): available_vehicles, capacity_utilization, etc.
    - Network features (6): destination_spread, route_density, etc.
    """
    
    def __init__(self, weights_path: str = "data/vfa_weights.json"):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.weights_path = Path(weights_path)
        
        # Learning parameters
        self.learning_rate = self.config.get('vfa.learning.initial_learning_rate', 0.01)
        self.learning_rate_decay = self.config.get('vfa.learning.learning_rate_decay', 0.9995)
        self.min_learning_rate = self.config.get('vfa.learning.min_learning_rate', 0.0001)
        
        self.discount_factor = self.config.get('vfa.learning.discount_factor', 0.95)
        self.lambda_trace = self.config.get('vfa.learning.lambda_trace', 0.7)
        
        # Exploration
        self.epsilon = self.config.get('vfa.exploration.initial_epsilon', 0.1)
        self.epsilon_decay = self.config.get('vfa.exploration.epsilon_decay', 0.995)
        self.min_epsilon = self.config.get('vfa.exploration.final_epsilon', 0.01)
        
        # Feature configuration
        self.num_features = 25
        self.feature_means = None
        self.feature_stds = None
        
        # Value function weights
        self.weights: np.ndarray = None
        self.eligibility_traces: np.ndarray = None
        
        # Learning statistics
        self.num_updates = 0
        self.recent_errors = []
        self.max_recent_errors = 1000
        self.feature_importance_history = []
        
        # Initialize or load weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights from config or load from file"""
        if self.weights_path.exists():
            self._load_weights()
        else:
            # Initialize with domain knowledge
            self.weights = np.zeros(self.num_features)
            self.eligibility_traces = np.zeros(self.num_features)
            
            if self.config.get('vfa.bootstrap.use_domain_initialization', True):
                # Positive weights for good things
                self.weights[5] = 500.0   # High utilization potential is good
                self.weights[8] = 300.0   # More consolidation opportunities are good
                self.weights[10] = 200.0  # Route density is good
                
                # Negative weights for bad things
                self.weights[2] = -400.0  # Time pressure is bad
                self.weights[3] = -300.0  # Urgent shipments are concerning
                self.weights[14] = -200.0 # Low vehicle availability is bad
                
            self._save_weights()
            logger.info(f"Initialized new VFA weights: {self.num_features} features")
    
    def estimate_value(self, state: SystemState) -> ValueEstimate:
        """
        Estimate value V(s) for given state
        
        Returns:
            ValueEstimate with value, confidence, and feature vector
        """
        features = self.extract_features(state)
        value = np.dot(self.weights, features)
        
        # Confidence based on number of updates (increases over time)
        confidence = min(1.0, self.num_updates / 1000.0)
        
        return ValueEstimate(
            value=value,
            confidence=confidence,
            features=features
        )
    
    def update(self, s_t: SystemState, action: Dict, reward: float, s_tp1: SystemState):
        """
        TD(λ) Update: Learn from experience (s_t, a_t, r_t, s_{t+1})
        
        Update Rule:
        1. Calculate TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        2. Update eligibility trace: e_t = γ*λ*e_{t-1} + ∇V(s_t)
        3. Update weights: θ ← θ + α * δ_t * e_t
        
        Args:
            s_t: Current state
            action: Action taken
            reward: Immediate reward r_t
            s_tp1: Next state
        """
        
        # Extract features
        phi_t = self.extract_features(s_t)
        phi_tp1 = self.extract_features(s_tp1)
        
        # Normalize features if we have statistics
        if self.feature_means is not None:
            phi_t = (phi_t - self.feature_means) / (self.feature_stds + 1e-8)
            phi_tp1 = (phi_tp1 - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Compute values
        v_t = np.dot(self.weights, phi_t)
        v_tp1 = np.dot(self.weights, phi_tp1)
        
        # TD error
        td_error = reward + self.discount_factor * v_tp1 - v_t
        
        # Update eligibility traces (accumulating traces)
        self.eligibility_traces *= self.discount_factor * self.lambda_trace
        self.eligibility_traces += phi_t
        
        # Weight update
        self.weights += self.learning_rate * td_error * self.eligibility_traces
        
        # Track statistics
        self.recent_errors.append(abs(td_error))
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        self.num_updates += 1
        
        # Decay learning rate
        self.learning_rate = max(
            self.min_learning_rate,
            self.learning_rate * self.learning_rate_decay
        )
        
        # Decay epsilon
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )
        
        # Periodic saves
        if self.num_updates % 100 == 0:
            self._save_weights()
            logger.info(
                f"VFA Update #{self.num_updates}: "
                f"TD_error={td_error:.2f}, "
                f"V(s_t)={v_t:.2f}, "
                f"reward={reward:.2f}, "
                f"lr={self.learning_rate:.6f}"
            )
        
        # Update feature statistics periodically
        if self.num_updates % 50 == 0:
            self._update_feature_statistics()
    
    def extract_features(self, state: SystemState) -> np.ndarray:
        """
        Extract 25-dimensional feature vector φ(S_t) from state
        
        Feature groups:
        [0-4]   Time features (5)
        [5-12]  Consolidation features (8)
        [13-18] Fleet features (6)
        [19-24] Network features (6)
        """
        
        features = np.zeros(self.num_features)
        
        # ===== TIME FEATURES (0-4) =====
        hour = state.timestamp.hour
        features[0] = np.sin(2 * np.pi * hour / 24)  # Cyclical hour
        features[1] = np.cos(2 * np.pi * hour / 24)
        
        day_of_week = state.timestamp.weekday()
        features[2] = 1.0 if day_of_week >= 5 else 0.0  # Weekend indicator
        
        # Time pressure: fraction of pending shipments that are urgent
        if state.pending_shipments:
            urgent_count = len(state.get_urgent_shipments(threshold_hours=6))
            features[3] = urgent_count / len(state.pending_shipments)
        
        # Average time to deadline for pending shipments (normalized)
        if state.pending_shipments:
            avg_hours_to_deadline = np.mean([
                (s.delivery_deadline - state.timestamp).total_seconds() / 3600
                for s in state.pending_shipments
            ])
            features[4] = max(0, min(1, avg_hours_to_deadline / 48))  # Normalize to 0-1
        
        # ===== CONSOLIDATION FEATURES (5-12) =====
        n_pending = len(state.pending_shipments)
        features[5] = np.log1p(n_pending)  # Log scale for pending count
        
        if n_pending > 0:
            # Total volume and weight
            total_volume = sum(s.volume for s in state.pending_shipments)
            total_weight = sum(s.weight for s in state.pending_shipments)
            features[6] = total_volume / 100.0  # Normalize (assume max ~100 m³)
            features[7] = total_weight / 10000.0  # Normalize (assume max ~10 tons)
            
            # Average shipment size
            features[8] = total_volume / n_pending
            features[9] = total_weight / n_pending
            
            # Priority distribution
            high_priority = sum(1 for s in state.pending_shipments if s.priority == 'high')
            features[10] = high_priority / n_pending
            
            # Value density (declared value per volume)
            total_value = sum(s.declared_value for s in state.pending_shipments)
            features[11] = total_value / max(total_volume, 1.0)
        
        # Consolidation potential (could we fill a vehicle well?)
        available_vehicles = state.get_available_vehicles()
        if available_vehicles and n_pending > 0:
            best_vehicle = max(available_vehicles, key=lambda v: v.capacity.volume)
            total_volume = sum(s.volume for s in state.pending_shipments)
            features[12] = min(1.0, total_volume / best_vehicle.capacity.volume)
        
        # ===== FLEET FEATURES (13-18) =====
        n_available = len(available_vehicles)
        total_fleet = len(state.fleet_state)
        
        features[13] = n_available / max(total_fleet, 1)  # Fraction available
        features[14] = np.log1p(n_available)  # Absolute count (log scale)
        
        if available_vehicles:
            # Average vehicle capacity
            avg_capacity = np.mean([v.capacity.volume for v in available_vehicles])
            features[15] = avg_capacity / 50.0  # Normalize (assume max ~50 m³)
            
            # Fleet diversity (different vehicle types)
            unique_types = len(set(v.type for v in available_vehicles))
            features[16] = unique_types / max(total_fleet, 1)
        
        # Active routes count
        features[17] = np.log1p(len(state.active_routes))
        
        # Fleet utilization (vehicles in use)
        in_use = total_fleet - n_available
        features[18] = in_use / max(total_fleet, 1)
        
        # ===== NETWORK FEATURES (19-24) =====
        if n_pending > 0:
            # Geographic spread of destinations
            lats = []
            lngs = []
            for s in state.pending_shipments:
                for dest in s.destinations:
                    lats.append(dest.lat)
                    lngs.append(dest.lng)
            
            if lats:
                lat_spread = np.std(lats) if len(lats) > 1 else 0
                lng_spread = np.std(lngs) if len(lngs) > 1 else 0
                features[19] = lat_spread * 100  # Scale up for meaningful values
                features[20] = lng_spread * 100
                
                # Centroid distance from fleet base (if applicable)
                mean_lat = np.mean(lats)
                mean_lng = np.mean(lngs)
                features[21] = abs(mean_lat + 1.286)  # Distance from Nairobi center
                features[22] = abs(mean_lng - 36.817)
            
            # Route density: shipments per unit area
            if lat_spread > 0 and lng_spread > 0:
                area = lat_spread * lng_spread
                features[23] = n_pending / max(area, 0.01)
            
            # Multi-destination shipments (complexity indicator)
            multi_dest_count = sum(1 for s in state.pending_shipments if len(s.destinations) > 1)
            features[24] = multi_dest_count / n_pending
        
        return features
    
    def _update_feature_statistics(self):
        """Update running statistics for feature normalization"""
        # In a production system, we'd maintain running mean and std
        # For now, we'll skip normalization or use simple scaling
        pass
    
    def _save_weights(self):
        """Save weights to disk"""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'weights': self.weights.tolist(),
            'eligibility_traces': self.eligibility_traces.tolist(),
            'num_updates': self.num_updates,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'recent_avg_error': np.mean(self.recent_errors) if self.recent_errors else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.weights_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_weights(self):
        """Load weights from disk"""
        with open(self.weights_path, 'r') as f:
            data = json.load(f)
        
        self.weights = np.array(data['weights'])
        self.eligibility_traces = np.array(data.get('eligibility_traces', np.zeros(self.num_features)))
        self.num_updates = data.get('num_updates', 0)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.epsilon = data.get('epsilon', self.epsilon)
        
        logger.info(
            f"Loaded VFA weights from {self.weights_path}: "
            f"{self.num_updates} updates, "
            f"avg_error={data.get('recent_avg_error', 0):.2f}"
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on absolute weight values"""
        feature_names = [
            # Time features
            "hour_sin", "hour_cos", "is_weekend", "urgency_fraction", "avg_time_to_deadline",
            # Consolidation features
            "log_pending_count", "total_volume", "total_weight", "avg_volume_per_shipment",
            "avg_weight_per_shipment", "high_priority_fraction", "value_density", "consolidation_potential",
            # Fleet features
            "fraction_available", "log_available_count", "avg_vehicle_capacity", "fleet_diversity",
            "active_routes_count", "fleet_utilization",
            # Network features
            "lat_spread", "lng_spread", "centroid_lat_offset", "centroid_lng_offset",
            "route_density", "multi_dest_fraction"
        ]
        
        importance = {}
        for i, name in enumerate(feature_names):
            importance[name] = abs(self.weights[i])
        
        return importance
    
    def get_learning_metrics(self) -> Dict:
        """Get current learning metrics for monitoring"""
        return {
            'num_updates': self.num_updates,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'avg_td_error': np.mean(self.recent_errors) if self.recent_errors else 0.0,
            'max_td_error': np.max(self.recent_errors) if self.recent_errors else 0.0,
            'weight_norm': np.linalg.norm(self.weights),
            'feature_importance': self.get_feature_importance()
        }