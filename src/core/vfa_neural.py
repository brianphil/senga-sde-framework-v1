# src/core/vfa_neural.py
"""
Neural Network Value Function Approximator
Replaces linear VFA with deep learning for better expressiveness
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pathlib import Path
import logging
from collections import deque

from .state_manager import SystemState
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)

@dataclass
class ValueEstimate:
    """Value estimate with uncertainty"""
    value: float
    confidence: float
    features: np.ndarray


class ValueNetwork(nn.Module):
    """
    Deep neural network: V(s; θ) = f_θ(φ(s))
    Architecture: [50] → 128 → 64 → 32 → 1
    """
    
    def __init__(self, input_dim: int = 50):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        return self.network(x).squeeze()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ExperienceReplayBuffer:
    """Circular buffer for experience replay"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state_features, reward, next_state_features):
        self.buffer.append((state_features, reward, next_state_features))
    
    def sample(self, batch_size: int):
        import random
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([b[0] for b in batch])
        rewards = torch.FloatTensor([b[1] for b in batch])
        next_states = torch.FloatTensor([b[2] for b in batch])
        
        return states, rewards, next_states
    
    def __len__(self):
        return len(self.buffer)


class NeuralVFA:
    """
    Neural Network Value Function Approximator
    
    Mathematical Foundation:
    - TD Error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
    - Update: θ ← θ - α*∇_θ[(δ_t)²]
    - Uses target network and experience replay for stability
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.value_net = ValueNetwork(input_dim=50).to(self.device)
        self.target_net = ValueNetwork(input_dim=50).to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.config.get('vfa.learning.lr', 0.001)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config.get('vfa.learning.lr_decay', 0.995)
        )
        
        # Loss
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=self.config.get('vfa.replay.capacity', 10000)
        )
        
        # Hyperparameters
        self.gamma = self.config.get('vfa.learning.discount_factor', 0.95)
        self.batch_size = self.config.get('vfa.replay.batch_size', 32)
        self.target_update_freq = self.config.get('vfa.target_update_freq', 100)
        
        # Statistics
        self.update_count = 0
        self.td_errors = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        
        # Feature statistics for normalization
        self.feature_means = None
        self.feature_stds = None
        
        logger.info(f"Neural VFA initialized on {self.device}")
    
    def evaluate(self, state: SystemState) -> ValueEstimate:
        """Evaluate V(s)"""
        features = self._extract_features(state)
        features_normalized = self._normalize_features(features)
        features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(self.device)
        
        self.value_net.eval()  # Set to eval mode to disable BatchNorm
        with torch.no_grad():
            value = self.value_net(features_tensor).item()
        
        # Confidence based on recent TD errors
        if len(self.td_errors) > 10:
            error_std = np.std(list(self.td_errors)[-100:])
            confidence = 1.0 / (1.0 + error_std / 100.0)
        else:
            confidence = 0.5
        
        return ValueEstimate(
            value=value,
            confidence=min(confidence, 1.0),
            features=features
        )
    
    def estimate_value(self, state: SystemState) -> ValueEstimate:
        """Alias for evaluate() for backwards compatibility"""
        return self.evaluate(state)
    
    def td_update(self, s_t: SystemState, action: dict, reward: float, s_tp1: SystemState):
        """
        TD learning update with experience replay
        
        Algorithm:
        1. Store experience
        2. Sample minibatch
        3. Compute TD targets
        4. Update network
        """
        # Extract and store experience
        features_t = self._extract_features(s_t)
        features_tp1 = self._extract_features(s_tp1)
        
        self.replay_buffer.add(features_t, reward, features_tp1)
        
        # Update if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._update_from_batch()
    
    def update(self, s_t: SystemState, action: dict, reward: float, s_tp1: SystemState):
        """Alias for td_update() for backwards compatibility"""
        self.td_update(s_t, action, reward, s_tp1)
    
    def _update_from_batch(self):
        """Perform batch gradient update"""
        self.value_net.train()  # Set to training mode for BatchNorm
        
        # Sample batch
        states, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        
        # Normalize
        states = self._normalize_tensor(states)
        next_states = self._normalize_tensor(next_states)
        
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Current values
        current_values = self.value_net(states)
        
        # TD targets using target network
        with torch.no_grad():
            next_values = self.target_net(next_states)
            td_targets = rewards + self.gamma * next_values
        
        # Compute loss
        loss = self.criterion(current_values, td_targets)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track metrics
        self.losses.append(loss.item())
        td_error = torch.abs(td_targets - current_values).mean().item()
        self.td_errors.append(td_error)
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
            logger.info(f"Updated target network at step {self.update_count}")
        
        # Learning rate decay
        if self.update_count % 1000 == 0:
            self.scheduler.step()
            logger.info(
                f"VFA Update #{self.update_count}: "
                f"Loss={loss.item():.4f}, TD_error={td_error:.4f}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
            )
    
    def _extract_features(self, state: SystemState) -> np.ndarray:
        """
        Extract 50-dimensional feature vector - FIXED for correct data structures
        
        SystemState has:
        - fleet_state (not vehicle_states)
        
        Shipment has:
        - destinations (list of Location, not dest_address string)
        - origin (Location object)
        """
        features = np.zeros(50)
        
        # Time features [0-4]
        hour = state.timestamp.hour
        features[0] = np.sin(2 * np.pi * hour / 24)
        features[1] = np.cos(2 * np.pi * hour / 24)
        features[2] = 1.0 if state.timestamp.weekday() >= 5 else 0.0
        features[3] = hour / 24.0
        features[4] = state.timestamp.weekday() / 7.0
        
        # Shipment features [5-14]
        pending = state.pending_shipments
        features[5] = len(pending)
        
        if pending:
            features[6] = np.mean([s.volume for s in pending])
            features[7] = np.mean([s.weight for s in pending])
            features[8] = np.mean([s.priority for s in pending])
            
            # Urgency metrics
            urgent = state.get_urgent_shipments(threshold_hours=6)
            features[9] = len(urgent) / len(pending) if pending else 0.0
            features[10] = len([s for s in pending if s.priority >= 2.0]) / len(pending)
            
            # Time to deadlines
            deadlines = [(s.deadline - state.timestamp).total_seconds() / 3600 for s in pending]
            features[11] = np.mean(deadlines) if deadlines else 24.0
            features[12] = np.min(deadlines) if deadlines else 24.0
            features[13] = np.max(deadlines) if deadlines else 24.0
            features[14] = np.std(deadlines) if len(deadlines) > 1 else 0.0
        
        # Fleet features [15-24]
        available = state.get_available_vehicles()
        features[15] = len(available)
        features[16] = len(state.fleet_state)
        features[17] = len(available) / max(len(state.fleet_state), 1)
        
        if available:
            features[18] = np.mean([v.capacity.volume for v in available])
            features[19] = np.mean([v.capacity.weight for v in available])
            
            # Capacity utilization potential
            if pending:
                total_vol_pending = sum(s.volume for s in pending)
                total_cap_vol = sum(v.capacity.volume for v in available)
                features[20] = total_vol_pending / max(total_cap_vol, 1.0)
                
                total_wt_pending = sum(s.weight for s in pending)
                total_cap_wt = sum(v.capacity.weight for v in available)
                features[21] = total_wt_pending / max(total_cap_wt, 1.0)
        
        # Geographic features [25-34]
        if pending:
            # Shipment has destinations (list), get first destination's formatted_address
            dest_addresses = []
            for s in pending:
                if s.destinations:
                    dest_addresses.append(s.destinations[0].formatted_address)
            
            if dest_addresses:
                unique_dests = len(set(dest_addresses))
                features[25] = unique_dests / len(pending)
                features[26] = unique_dests
                features[27] = len(pending) / max(unique_dests, 1)
        
        # Consolidation potential [35-44]
        if pending:
            # Group by destination (using first destination's address)
            dest_groups = {}
            for s in pending:
                if s.destinations:
                    dest = s.destinations[0].formatted_address
                    if dest not in dest_groups:
                        dest_groups[dest] = []
                    dest_groups[dest].append(s)
            
            if dest_groups:
                group_sizes = [len(g) for g in dest_groups.values()]
                features[35] = np.mean(group_sizes)
                features[36] = np.max(group_sizes)
                features[37] = len([g for g in group_sizes if g >= 2])
                features[38] = len([g for g in group_sizes if g >= 5])
        
        # Network features [45-49]
        features[45] = len(state.active_routes)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics"""
        if self.feature_means is None:
            return features
        
        return (features - self.feature_means) / (self.feature_stds + 1e-8)
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor batch"""
        if self.feature_means is None:
            self._update_feature_statistics(tensor.cpu().numpy())
        
        means = torch.FloatTensor(self.feature_means).to(tensor.device)
        stds = torch.FloatTensor(self.feature_stds).to(tensor.device)
        
        return (tensor - means) / (stds + 1e-8)
    
    def _update_feature_statistics(self, features: np.ndarray):
        """Update running feature statistics"""
        if self.feature_means is None:
            self.feature_means = np.mean(features, axis=0) if features.ndim > 1 else features
            self.feature_stds = np.std(features, axis=0) + 1e-8 if features.ndim > 1 else np.ones_like(features)
        else:
            # Exponential moving average
            alpha = 0.01
            new_means = np.mean(features, axis=0) if features.ndim > 1 else features
            self.feature_means = alpha * new_means + (1 - alpha) * self.feature_means
            
            new_stds = np.std(features, axis=0) if features.ndim > 1 else np.ones_like(features)
            self.feature_stds = alpha * new_stds + (1 - alpha) * self.feature_stds
    
    def save_model(self, path: str = "data/vfa_model.pt"):
        """Save model weights"""
        torch.save({
            'value_net': self.value_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'update_count': self.update_count
        }, path)
        logger.info(f"Saved VFA model to {path}")
    
    def load_model(self, path: str = "data/vfa_model.pt"):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.feature_means = checkpoint['feature_means']
        self.feature_stds = checkpoint['feature_stds']
        self.update_count = checkpoint['update_count']
        logger.info(f"Loaded VFA model from {path}")
    
    def get_learning_metrics(self) -> Dict:
        """
        Get learning metrics matching API expectations
        
        Returns all fields expected by /learning/vfa-metrics endpoint
        """
        if len(self.td_errors) < 10:
            recent_td = [0.0]
        else:
            recent_td = list(self.td_errors)[-100:]
        
        # Calculate weight norm (L2 norm of network parameters)
        total_norm = 0.0
        for param in self.value_net.parameters():
            total_norm += param.data.norm(2).item() ** 2
        weight_norm = total_norm ** 0.5
        
        return {
            'num_updates': self.update_count,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epsilon': 0.0,  # Neural VFA doesn't use epsilon-greedy
            'avg_td_error': np.mean(recent_td),
            'max_td_error': np.max(recent_td),
            'weight_norm': weight_norm,
            'feature_importance': self.get_feature_importance()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from first layer weights
        
        Returns dictionary of feature names to importance scores
        """
        # Feature names (50 features)
        feature_names = [
            # Time features [0-4]
            "hour_sin", "hour_cos", "is_weekend", "hour_normalized", "day_of_week",
            
            # Shipment features [5-14]
            "pending_count", "avg_volume", "avg_weight", "avg_priority",
            "urgency_ratio", "high_priority_ratio", "avg_time_to_deadline",
            "min_time_to_deadline", "max_time_to_deadline", "time_to_deadline_std",
            
            # Fleet features [15-24]
            "available_vehicles", "total_vehicles", "availability_ratio",
            "avg_vehicle_volume_cap", "avg_vehicle_weight_cap",
            "volume_utilization_potential", "weight_utilization_potential",
            "fleet_feature_18", "fleet_feature_19", "fleet_feature_20",
            
            # Geographic features [25-34]
            "destination_diversity", "unique_destinations", "orders_per_destination",
            "geo_feature_28", "geo_feature_29", "geo_feature_30",
            "geo_feature_31", "geo_feature_32", "geo_feature_33", "geo_feature_34",
            
            # Consolidation features [35-44]
            "avg_group_size", "max_group_size", "groups_with_2plus",
            "groups_with_5plus", "consol_feature_39", "consol_feature_40",
            "consol_feature_41", "consol_feature_42", "consol_feature_43", "consol_feature_44",
            
            # Network features [45-49]
            "active_routes_count", "network_feature_46", "network_feature_47",
            "network_feature_48", "network_feature_49"
        ]
        
        # Get first layer weights
        first_layer = self.value_net.network[0]  # First Linear layer
        weights = first_layer.weight.data.cpu().numpy()  # Shape: [128, 50]
        
        # Calculate importance as L2 norm of weights for each input feature
        importance_scores = np.linalg.norm(weights, axis=0)  # Shape: [50]
        
        # Normalize to sum to 1
        importance_scores = importance_scores / (importance_scores.sum() + 1e-8)
        
        # Create dictionary
        importance = {}
        for i, name in enumerate(feature_names[:50]):  # Ensure we don't exceed array size
            importance[name] = float(importance_scores[i])
        
        # Return top 10 for readability
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])
    
    def get_convergence_metrics(self) -> Dict:
        """Get convergence metrics (enhanced version)"""
        if len(self.td_errors) < 10:
            return {
                'converged': False, 
                'updates': self.update_count,
                'num_updates': self.update_count,
                'avg_td_error': 0.0,
                'max_td_error': 0.0,
                'avg_loss': 0.0,
                'td_error_std': 0.0
            }
        
        recent_td = list(self.td_errors)[-100:]
        recent_loss = list(self.losses)[-100:]
        
        return {
            'converged': np.mean(recent_td) < 50.0,
            'updates': self.update_count,
            'num_updates': self.update_count,
            'avg_td_error': np.mean(recent_td),
            'max_td_error': np.max(recent_td),
            'avg_loss': np.mean(recent_loss),
            'td_error_std': np.std(recent_td)
        }