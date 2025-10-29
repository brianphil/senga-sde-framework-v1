# src/core/pfa.py
"""
Policy Function Approximator - FIXED IMPLEMENTATION
Maintains exact class name: PolicyFunctionApproximator (no breaking changes)
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class PFAAction:
    action_type: str
    shipments: list
    vehicle: object
    confidence: float
    reasoning: str
    feature_vector: np.ndarray = None
    state_features: np.ndarray = None

class PolicyFunctionApproximator:
    """EXACT NAME - No breaking changes"""
    
    def __init__(self):
        self.theta = np.zeros(35)
        self.lr = 0.01
        self.gamma = 0.95
        self.experience = deque(maxlen=1000)
        logger.info("PFA initialized with policy gradient learning")
    
    def decide(self, state) -> PFAAction:
        """EXACT signature - FIXED action types"""
        # Emergency check - use DISPATCH_IMMEDIATE for emergencies
        urgent = [s for s in state.pending_shipments 
                 if (s.deadline - state.timestamp).total_seconds() / 3600 < 2]
        
        available = state.get_available_vehicles()
        
        if urgent and available:
            return PFAAction(
                action_type='DISPATCH_IMMEDIATE',  # FIXED: was EMERGENCY_DISPATCH
                shipments=urgent[:1],
                vehicle=available[0],
                confidence=1.0,
                reasoning='Emergency: deadline < 2hrs'
            )
        
        # Generate candidates
        candidates = self._generate_candidates(state)
        
        if not candidates:
            return PFAAction(
                action_type='WAIT',
                shipments=[],
                vehicle=None,
                confidence=0.7,
                reasoning='No viable candidates'
            )
        
        # Policy evaluation
        action_probs = self._policy_forward(state, candidates)
        action_idx = np.random.choice(len(candidates), p=action_probs)
        selected = candidates[action_idx]
        
        return PFAAction(
            action_type='DISPATCH_IMMEDIATE',
            shipments=selected['shipments'],
            vehicle=selected['vehicle'],
            confidence=float(action_probs[action_idx]),
            reasoning=f"Policy π(a|s)={action_probs[action_idx]:.3f}"
        )
    
    def policy_gradient_update(self, s_t, action, reward, vfa_baseline):
        """Policy gradient update"""
        features = self._extract_state_action_features(s_t, action)
        
        value_estimate = vfa_baseline.estimate_value(s_t) if vfa_baseline else 0
        advantage = reward - (value_estimate.value if hasattr(value_estimate, 'value') else value_estimate)
        
        gradient = features
        self.theta += self.lr * gradient * advantage
        
        self.experience.append((features, reward, advantage))
    
    def _policy_forward(self, state, candidates) -> np.ndarray:
        """Compute π(a|s;θ) for all candidates"""
        logits = []
        for cand in candidates:
            features = self._extract_state_action_features(state, cand)
            logit = np.dot(self.theta, features)
            logits.append(logit)
        
        logits = np.array(logits)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        return probs
    
    def _generate_candidates(self, state) -> List[Dict]:
        """Generate candidate actions - FIXED"""
        candidates = []
        available_veh = state.get_available_vehicles()
        pending = state.pending_shipments
        
        if not available_veh or not pending:
            return []
        
        # Candidate 1: Most urgent shipment
        urgent = sorted(pending, key=lambda s: (s.deadline - state.timestamp).total_seconds())
        if urgent:
            candidates.append({
                'shipments': [urgent[0]],
                'vehicle': available_veh[0],
                'type': 'urgent_dispatch'
            })
        
        # Candidate 2: Consolidation (same zone)
        from collections import defaultdict
        zone_groups = defaultdict(list)
        for s in pending:
            if s.destinations:
                zone = s.destinations[0].zone_id or 'unknown'
                zone_groups[zone].append(s)
        
        for zone, ships in zone_groups.items():
            if len(ships) >= 2:
                candidates.append({
                    'shipments': ships[:3],
                    'vehicle': available_veh[0],
                    'type': 'consolidated'
                })
        
        return candidates
    
    def _extract_state_action_features(self, state, action) -> np.ndarray:
        """Extract 35 features - FIXED"""
        import math
        
        # State features [20]
        hour = state.timestamp.hour
        hour_sin = math.sin(2*math.pi*hour/24)
        hour_cos = math.cos(2*math.pi*hour/24)
        
        pending = state.pending_shipments
        n_pending = len(pending)
        n_available = len(state.get_available_vehicles())
        
        avg_urgency = 0
        if pending:
            urgencies = [(s.deadline - state.timestamp).total_seconds() / 3600 for s in pending]
            avg_urgency = np.mean(urgencies)
        
        state_feat = np.array([
            hour_sin, hour_cos, state.timestamp.weekday()/7.0,
            n_pending, n_available, n_available/(len(state.fleet_state)+1e-8),
            avg_urgency,
            min([(s.deadline - state.timestamp).total_seconds() / 3600 for s in pending] or [24]),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ])
        
        # Action features [8]
        if isinstance(action, dict):
            shipments = action.get('shipments', [])
            vehicle = action.get('vehicle')
        else:
            shipments = getattr(action, 'shipments', [])
            vehicle = getattr(action, 'vehicle', None)
        
        n_ships = len(shipments)
        total_vol = sum(s.volume for s in shipments) if shipments else 0
        utilization = total_vol / vehicle.capacity.volume if vehicle and vehicle.capacity.volume > 0 else 0
        
        action_feat = np.array([
            n_ships, utilization,
            np.mean([(s.deadline - state.timestamp).total_seconds() / 3600 for s in shipments]) if shipments else 0,
            0, 0, 0, 0, 0
        ])
        
        # Interactions [7]
        interact_feat = np.array([n_pending * n_ships, avg_urgency * utilization, 0, 0, 0, 0, 0])
        
        features = np.concatenate([state_feat, action_feat, interact_feat])
        return features[:35]
    
    def get_learning_metrics(self) -> Dict:
        """EXACT signature - backward compatible"""
        recent_rewards = [e[1] for e in list(self.experience)[-100:]]
        recent_advantages = [e[2] for e in list(self.experience)[-100:]]
        
        return {
            'num_updates': len(self.experience),
            'learning_rate': self.lr,
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_advantage': np.mean(recent_advantages) if recent_advantages else 0,
            'theta_norm': np.linalg.norm(self.theta)
        }