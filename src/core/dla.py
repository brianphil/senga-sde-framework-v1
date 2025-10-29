# src/core/dla.py
"""
Direct Lookahead Approximator - FIXED IMPLEMENTATION
Maintains exact class name: DirectLookaheadApproximator (no breaking changes)
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from datetime import timedelta
import copy
import logging

logger = logging.getLogger(__name__)

@dataclass
class LookaheadResult:
    best_action: dict
    expected_value: float
    scenario_values: List[float]
    reasoning: str
    action: dict = None  # Alias for backward compatibility
    
    def __post_init__(self):
        if self.action is None:
            self.action = self.best_action

class DirectLookaheadApproximator:
    """EXACT NAME - No breaking changes"""
    
    def __init__(self):
        from ..config.senga_config import SengaConfigurator
        self.config = SengaConfigurator()
        self.n_scenarios = self.config.get('dla.lookahead.num_monte_carlo_samples', 100)
        self.horizon = self.config.get('dla.lookahead.horizon_hours', 4)
        self.gamma = 0.95
        
        self.traffic_std = 0.3
        self.breakdown_prob = 0.05
        self.new_demand_rate = 2.0
        
        logger.info(f"DLA initialized: {self.n_scenarios} scenarios, {self.horizon}hr horizon")
    
    def evaluate_with_lookahead(self, state, candidates, value_function=None) -> LookaheadResult:
        """EXACT signature - backward compatible"""
        if not candidates:
            return LookaheadResult({}, 0, [], 'No candidates')
        
        action_values = {}
        
        for cand in candidates:
            scenario_values = []
            
            for _ in range(self.n_scenarios):
                scenario = self._generate_scenario(state)
                value = self._simulate_scenario(state, cand, scenario, value_function)
                scenario_values.append(value)
            
            action_values[str(cand)] = {
                'action': cand,
                'expected_value': np.mean(scenario_values),
                'std': np.std(scenario_values),
                'scenarios': scenario_values
            }
        
        best_key = max(action_values.keys(), key=lambda k: action_values[k]['expected_value'])
        best = action_values[best_key]
        
        return LookaheadResult(
            best_action=best['action'],
            expected_value=best['expected_value'],
            scenario_values=best['scenarios'],
            reasoning=f"Expected value: {best['expected_value']:.1f} ± {best['std']:.1f}"
        )
    
    def _generate_scenario(self, state) -> List[Dict]:
        """Generate realistic scenario - FIXED"""
        scenario = []
        
        for t in range(int(self.horizon)):
            events = {}
            
            # Traffic - Nairobi patterns
            hour = (state.timestamp + timedelta(hours=t)).hour
            base_multiplier = self._get_traffic_multiplier(hour)
            traffic_noise = np.random.normal(0, self.traffic_std)
            events['traffic_multiplier'] = max(0.5, base_multiplier + traffic_noise)
            
            # New demand
            events['new_shipments'] = np.random.poisson(self.new_demand_rate)
            
            # Breakdowns - FIXED to use state.fleet_state
            events['breakdown_vehicle_ids'] = []
            for v in state.fleet_state:
                if np.random.random() < self.breakdown_prob:
                    events['breakdown_vehicle_ids'].append(v.id)
            
            # Weather
            events['weather_delay_factor'] = 1.0 + np.random.exponential(0.1) if np.random.random() < 0.2 else 1.0
            
            scenario.append(events)
        
        return scenario
    
    def _simulate_scenario(self, initial_state, initial_action, scenario, vfa) -> float:
        """Simulate trajectory - FIXED"""
        state = copy.deepcopy(initial_state)
        total_value = 0.0
        
        # Execute initial action
        reward = self._execute_action_sim(state, initial_action, scenario[0] if scenario else {})
        total_value += reward
        
        state = self._transition_state(state, initial_action, scenario[0] if scenario else {})
        
        # Simulate forward
        for t in range(1, len(scenario)):
            action = self._greedy_action(state)
            reward = self._execute_action_sim(state, action, scenario[t])
            total_value += (self.gamma ** t) * reward
            state = self._transition_state(state, action, scenario[t])
        
        # Terminal value
        if vfa:
            terminal_value = vfa.estimate_value(state).value
            total_value += (self.gamma ** len(scenario)) * terminal_value
        
        return total_value
    
    def _execute_action_sim(self, state, action, events) -> float:
        """Simulate action execution"""
        if action.get('type') == 'WAIT':
            return -10.0
        
        batches = action.get('batches', [])
        total_reward = 0.0
        
        for batch in batches:
            util = batch.utilization if hasattr(batch, 'utilization') else 0.6
            util_bonus = util * 100 * 50
            
            traffic_mult = events.get('traffic_multiplier', 1.0)
            distance = getattr(batch, 'total_distance_km', 50)
            cost = distance * 50 * traffic_mult
            
            weather_factor = events.get('weather_delay_factor', 1.0)
            on_time_prob = 0.9 / (traffic_mult * weather_factor)
            on_time_bonus = 1000 * on_time_prob
            
            batch_reward = util_bonus + on_time_bonus - cost
            total_reward += batch_reward
        
        return total_reward
    
    def _transition_state(self, state, action, events):
        """State transition - FIXED to use state.fleet_state"""
        new_state = copy.deepcopy(state)
        new_state.timestamp += timedelta(hours=1)
        
        if action.get('type') == 'DISPATCH':
            batches = action.get('batches', [])
            for batch in batches:
                for ship in (batch.shipments if hasattr(batch, 'shipments') else []):
                    if ship in new_state.pending_shipments:
                        new_state.pending_shipments.remove(ship)
        
        # Apply breakdowns - FIXED
        breakdown_ids = events.get('breakdown_vehicle_ids', [])
        for v in new_state.fleet_state:
            if v.id in breakdown_ids:
                from .state_manager import VehicleStatus
                v.status = VehicleStatus.OFFLINE
        
        return new_state
    
    def _greedy_action(self, state) -> dict:
        """Greedy action - FIXED"""
        pending = state.pending_shipments
        available = state.get_available_vehicles()  # FIXED
        
        if not pending or not available:
            return {'type': 'WAIT'}
        
        # Greedy: most urgent
        urgent = min(pending, key=lambda s: (s.deadline - state.timestamp).total_seconds())
        return {
            'type': 'DISPATCH',
            'batches': [type('Batch', (), {
                'shipments': [urgent],
                'vehicle': available[0],
                'utilization': 0.5,
                'total_distance_km': 30
            })()]
        }
    
    def _get_traffic_multiplier(self, hour: int) -> float:
        """Nairobi traffic patterns"""
        if 7 <= hour <= 10:
            return 1.8
        elif 16 <= hour <= 19:
            return 2.0
        elif hour >= 20 or hour <= 6:
            return 0.7
        else:
            return 1.0