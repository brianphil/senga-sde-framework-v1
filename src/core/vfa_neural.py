# src/core/vfa_neural.py
"""
Neural Network Value Function Approximator - FIXED IMPLEMENTATION
Fixed PyTorch warning about slow tensor creation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValueEstimate:
    value: float
    confidence: float
    features: np.ndarray

class ValueNetwork(nn.Module):
    """V(s;θ): State → Value"""
    def __init__(self, input_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class NeuralVFA:
    """EXACT NAME - No breaking changes"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.value_net = ValueNetwork().to(self.device)
        self.target_net = ValueNetwork().to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=0.001)
        self.gamma = 0.95
        self.replay_buffer = deque(maxlen=10000)
        self.td_errors = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        self.update_count = 0
        
        self.feature_mean = None
        self.feature_std = None
        
        logger.info(f"Neural VFA initialized on {self.device}")
    
    def estimate_value(self, state) -> ValueEstimate:
        """Primary method"""
        features = self._extract_features(state)
        features_norm = self._normalize(features)
        
        self.value_net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)
            value = self.value_net(x).item()
        
        confidence = 1.0 / (1.0 + np.std(list(self.td_errors)[-100:]) / 100.0) if len(self.td_errors) > 10 else 0.5
        return ValueEstimate(value=value, confidence=min(confidence, 1.0), features=features)
    
    def evaluate(self, state) -> ValueEstimate:
        """Alias for backward compatibility"""
        return self.estimate_value(state)
    
    def td_update(self, s_t, action, reward, s_tp1):
        """TD-Learning update - FIXED PyTorch warning"""
        feat_t = self._extract_features(s_t)
        feat_tp1 = self._extract_features(s_tp1)
        self.replay_buffer.append((feat_t, reward, feat_tp1))
        
        if len(self.replay_buffer) < 32:
            return
        
        import random
        batch = random.sample(list(self.replay_buffer), 32)
        
        # FIXED: Convert to numpy array first, then to tensor
        states_np = np.array([self._normalize(b[0]) for b in batch])
        rewards_np = np.array([b[1] for b in batch])
        next_states_np = np.array([self._normalize(b[2]) for b in batch])
        
        states = torch.FloatTensor(states_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        
        self.target_net.eval()
        with torch.no_grad():
            next_values = self.target_net(next_states)
            targets = rewards + self.gamma * next_values
        
        self.value_net.train()
        predictions = self.value_net(states)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()
        
        td_error = (predictions - targets).abs().mean().item()
        self.td_errors.append(td_error)
        self.losses.append(loss.item())
        self.update_count += 1
        
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
            logger.info(f"VFA Update #{self.update_count}: TD_error={td_error:.2f}")
    
    def update(self, s_t, action, reward, s_tp1):
        """Alias for td_update"""
        self.td_update(s_t, action, reward, s_tp1)
    
    def _extract_features(self, state) -> np.ndarray:
        """Extract 50 features"""
        import math
        
        features = np.zeros(50, dtype=np.float32)
        
        # Time [0-4]
        hour = state.timestamp.hour
        features[0] = math.sin(2 * math.pi * hour / 24)
        features[1] = math.cos(2 * math.pi * hour / 24)
        features[2] = 1.0 if state.timestamp.weekday() >= 5 else 0.0
        features[3] = hour / 24.0
        features[4] = state.timestamp.weekday() / 7.0
        
        # Shipments [5-14]
        pending = state.pending_shipments
        features[5] = len(pending)
        
        if pending:
            features[6] = np.mean([s.volume for s in pending])
            features[7] = np.mean([s.weight for s in pending])
            features[8] = np.mean([s.priority for s in pending])
            
            urgent = state.get_urgent_shipments(threshold_hours=6)
            features[9] = len(urgent) / len(pending) if pending else 0.0
            
            deadlines = [(s.deadline - state.timestamp).total_seconds() / 3600 for s in pending]
            features[11] = np.mean(deadlines) if deadlines else 24.0
            features[12] = np.min(deadlines) if deadlines else 24.0
            features[13] = np.max(deadlines) if deadlines else 24.0
            features[14] = np.std(deadlines) if len(deadlines) > 1 else 0.0
        
        # Fleet [15-24]
        available = state.get_available_vehicles()
        features[15] = len(available)
        features[16] = len(state.fleet_state)
        features[17] = len(available) / max(len(state.fleet_state), 1)
        
        if available:
            features[18] = np.mean([v.capacity.volume for v in available])
            features[19] = np.mean([v.capacity.weight for v in available])
            
            if pending:
                total_pending_vol = state.total_pending_volume()
                total_cap_vol = sum(v.capacity.volume for v in available)
                features[20] = total_pending_vol / total_cap_vol if total_cap_vol > 0 else 0
        
        # Geographic [25-34]
        if pending:
            lats = [s.origin.lat for s in pending]
            lngs = [s.origin.lng for s in pending]
            features[25] = np.std(lats) + np.std(lngs) if len(lats) > 1 else 0
            
            dest_locations = set()
            for s in pending:
                for d in s.destinations:
                    dest_locations.add((round(d.lat, 4), round(d.lng, 4)))
            features[26] = len(dest_locations)
            features[27] = len(pending) / len(dest_locations) if dest_locations else 0
        
        # Consolidation [35-44]
        if pending:
            dest_counts = {}
            for s in pending:
                for dest in s.destinations:
                    key = (round(dest.lat, 4), round(dest.lng, 4))
                    dest_counts[key] = dest_counts.get(key, 0) + 1
            
            if dest_counts:
                features[35] = np.mean(list(dest_counts.values()))
                features[36] = max(dest_counts.values())
                features[37] = sum(1 for c in dest_counts.values() if c >= 2)
        
        # Network [45-49]
        features[45] = len(state.active_routes)
        features[46] = state.fleet_utilization()
        
        return features
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None:
            self.feature_mean = features.copy()
            self.feature_std = np.ones_like(features)
        else:
            alpha = 0.01
            self.feature_mean = alpha * features + (1 - alpha) * self.feature_mean
            self.feature_std = alpha * np.abs(features - self.feature_mean) + (1 - alpha) * self.feature_std
        
        return (features - self.feature_mean) / (self.feature_std + 1e-8)
    
    def get_learning_metrics(self) -> Dict:
        """EXACT signature"""
        recent_td = list(self.td_errors)[-100:] if self.td_errors else [0.0]
        
        total_norm = 0.0
        for param in self.value_net.parameters():
            total_norm += param.data.norm(2).item() ** 2
        weight_norm = total_norm ** 0.5
        
        return {
            'num_updates': self.update_count,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epsilon': 0.0,
            'avg_td_error': np.mean(recent_td),
            'max_td_error': np.max(recent_td),
            'weight_norm': weight_norm,
            'feature_importance': {}
        }
    
    def get_convergence_metrics(self) -> Dict:
        """EXACT signature"""
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