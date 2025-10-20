# src/core/dla.py

"""
Direct Lookahead Approximation (DLA): Forward simulation for decision evaluation
Performs Monte Carlo lookahead to evaluate actions by simulating future outcomes
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from copy import deepcopy

from .state_manager import StateManager, SystemState, Shipment, VehicleState
from .cfa import CostFunctionApproximator
from .vfa import ValueFunctionApproximator
from ..config.senga_config import SengaConfigurator

@dataclass
class LookaheadResult:
    """Result from lookahead simulation"""
    action: dict
    expected_value: float
    value_variance: float
    num_scenarios: int
    reasoning: str

class DirectLookaheadApproximator:
    """
    DLA: Monte Carlo lookahead for high-stakes decisions
    
    Process:
    1. Generate candidate actions
    2. For each action, simulate K scenarios forward H time steps
    3. Evaluate outcomes using VFA for terminal states
    4. Choose action with best expected value
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.cfa = CostFunctionApproximator()
        self.vfa = ValueFunctionApproximator()
    
    def evaluate_with_lookahead(self, state: SystemState,
                                candidate_actions: List[dict],
                                horizon: int = None,
                                num_scenarios: int = None) -> LookaheadResult:
        """
        Evaluate candidate actions using Monte Carlo lookahead
        
        Args:
            state: Current state S_t
            candidate_actions: List of actions to evaluate
            horizon: Number of time steps to look ahead (hours)
            num_scenarios: Number of Monte Carlo samples per action
            
        Returns:
            LookaheadResult with best action and expected value
        """
        
        if horizon is None:
            horizon = self.config.dla_lookahead_horizon
        
        if num_scenarios is None:
            num_scenarios = self.config.get('dla.lookahead.num_monte_carlo_samples', 50)
        
        best_action = None
        best_value = -np.inf
        best_variance = 0.0
        
        for action in candidate_actions:
            # Simulate this action across multiple scenarios
            scenario_values = []
            
            for scenario_idx in range(num_scenarios):
                # Sample uncertain future (new arrivals, delays, etc.)
                scenario = self._sample_future_scenario(state, horizon)
                
                # Simulate forward with this action
                total_value = self._simulate_forward(
                    state=state,
                    initial_action=action,
                    scenario=scenario,
                    horizon=horizon
                )
                
                scenario_values.append(total_value)
            
            # Calculate expected value and variance
            expected_value = np.mean(scenario_values)
            value_variance = np.var(scenario_values)
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
                best_variance = value_variance
        
        return LookaheadResult(
            action=best_action,
            expected_value=best_value,
            value_variance=best_variance,
            num_scenarios=num_scenarios,
            reasoning=f"Lookahead over {horizon}h with {num_scenarios} scenarios: expected value={best_value:.2f}"
        )
    
    def _sample_future_scenario(self, state: SystemState, 
                               horizon: int) -> List[Dict]:
        """
        Sample one possible future scenario
        
        Returns: List of events for each hour in horizon
        Each event contains: {new_shipments: [...], delays: [...], etc.}
        """
        scenario = []
        
        for hour in range(horizon):
            # Sample new shipment arrivals
            arrival_rate = self._estimate_arrival_rate(
                state.timestamp + timedelta(hours=hour)
            )
            num_arrivals = np.random.poisson(arrival_rate)
            
            # Generate synthetic shipments (in production, sample from historical distribution)
            new_shipments = [
                self._generate_synthetic_shipment(state.timestamp + timedelta(hours=hour))
                for _ in range(num_arrivals)
            ]
            
            # Sample potential delays (simplified)
            delay_probability = 0.1  # 10% chance of delay per active route
            delays = {}
            for route in state.active_routes:
                if np.random.random() < delay_probability:
                    delay_minutes = np.random.exponential(30)  # Average 30min delay
                    delays[route.id] = timedelta(minutes=delay_minutes)
            
            scenario.append({
                'hour': hour,
                'new_shipments': new_shipments,
                'delays': delays
            })
        
        return scenario
    
    def _simulate_forward(self, state: SystemState,
                         initial_action: dict,
                         scenario: List[Dict],
                         horizon: int) -> float:
        """
        Simulate forward from state, applying initial_action at t=0
        then using CFA for subsequent decisions
        
        Returns: Total discounted value over horizon
        """
        
        # Make a copy to simulate on
        sim_state = deepcopy(state)
        total_value = 0.0
        gamma = self.config.get('vfa.learning.discount_factor', 0.95)
        
        for t in range(horizon):
            # Apply action (initial action at t=0, CFA decisions after)
            if t == 0:
                action = initial_action
            else:
                # Use CFA to decide at each subsequent step
                cfa_solution = self.cfa.optimize(sim_state, value_function=self.vfa)
                if cfa_solution.status in ['OPTIMAL', 'FEASIBLE'] and cfa_solution.batches:
                    action = {'type': 'DISPATCH', 'batches': cfa_solution.batches}
                else:
                    action = {'type': 'WAIT'}
            
            # Execute action and observe reward
            reward = self._execute_action_simulation(sim_state, action)
            
            # Add discounted reward
            total_value += (gamma ** t) * reward
            
            # Update state with scenario events
            if t < len(scenario):
                sim_state = self._update_state_with_scenario(sim_state, scenario[t])
            
            # Advance time
            sim_state.timestamp += timedelta(hours=1)
        
        # Add terminal value estimate from VFA
        terminal_value = self.vfa.evaluate(sim_state).value
        total_value += (gamma ** horizon) * terminal_value
        
        return total_value
    
    def _execute_action_simulation(self, state: SystemState, action: dict) -> float:
        """
        Simulate executing an action and compute immediate reward
        
        Reward = utilization_bonus + on_time_bonus - cost - delay_penalties
        """
        
        if action['type'] == 'WAIT':
            # Small negative reward for waiting (opportunity cost)
            return -1.0
        
        elif action['type'] == 'DISPATCH':
            batches = action.get('batches', [])
            
            total_reward = 0.0
            
            for batch in batches:
                # Utilization bonus
                util_bonus = self.config.get('utilization_bonus_per_percent', 100)
                utilization_reward = batch.utilization * 100 * util_bonus
                
                # Cost
                cost = batch.total_cost
                
                # Estimate on-time delivery (simplified)
                on_time_bonus = self.config.get('on_time_delivery_bonus', 1000)
                estimated_on_time_rate = 0.95  # Assume 95% on-time
                on_time_reward = on_time_bonus * len(batch.shipments) * estimated_on_time_rate
                
                # Delay penalties (simplified - assume some risk)
                estimated_delay_penalty = 0.0
                for shipment in batch.shipments:
                    if shipment.time_pressure(state.timestamp) > 0.8:
                        # High time pressure = higher delay risk
                        delay_risk = 0.2  # 20% chance
                        penalty = self.config.get('delay_penalty_per_hour', 500)
                        estimated_delay_penalty += delay_risk * penalty
                
                batch_reward = (
                    utilization_reward +
                    on_time_reward -
                    cost -
                    estimated_delay_penalty
                )
                
                total_reward += batch_reward
            
            return total_reward
        
        return 0.0
    
    def _update_state_with_scenario(self, state: SystemState, 
                                    event: Dict) -> SystemState:
        """Update state with scenario events (new shipments, delays)"""
        
        # Add new shipments
        for shipment in event['new_shipments']:
            state.pending_shipments.append(shipment)
        
        # Apply delays (update route completion times)
        for route_id, delay in event['delays'].items():
            for route in state.active_routes:
                if route.id == route_id:
                    if route.estimated_duration:
                        route.estimated_duration += delay
        
        return state
    
    def _estimate_arrival_rate(self, timestamp: datetime) -> float:
        """Estimate shipment arrival rate (shipments per hour)"""
        
        # Time-of-day pattern
        hour = timestamp.hour
        
        if 9 <= hour <= 17:  # Business hours
            base_rate = 2.0
        elif 17 < hour <= 20:  # Evening
            base_rate = 1.0
        else:  # Night
            base_rate = 0.3
        
        # Day-of-week pattern
        weekday = timestamp.weekday()
        if weekday < 5:  # Monday-Friday
            day_multiplier = 1.0
        elif weekday == 5:  # Saturday
            day_multiplier = 0.6
        else:  # Sunday
            day_multiplier = 0.3
        
        return base_rate * day_multiplier
    
    def _generate_synthetic_shipment(self, timestamp: datetime) -> Shipment:
        """Generate a synthetic shipment for simulation"""
        
        from .state_manager import Location, ShipmentStatus
        import uuid
        
        # Sample volume and weight from typical distributions
        volume = np.random.lognormal(mean=1.0, sigma=0.5)  # m³
        weight = np.random.lognormal(mean=6.0, sigma=0.8)  # kg
        
        # Generate random locations (in production, sample from historical distribution)
        origin = Location(
            place_id=f"sim_origin_{uuid.uuid4().hex[:8]}",
            lat=-1.286389 + np.random.normal(0, 0.1),
            lng=36.817223 + np.random.normal(0, 0.1),
            formatted_address="Simulated Origin",
            zone_id="NAIROBI_CBD"
        )
        
        num_destinations = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        destinations = []
        for _ in range(num_destinations):
            dest = Location(
                place_id=f"sim_dest_{uuid.uuid4().hex[:8]}",
                lat=-1.286389 + np.random.normal(0, 0.5),
                lng=36.817223 + np.random.normal(0, 0.5),
                formatted_address="Simulated Destination",
                zone_id=np.random.choice(["NAKURU", "ELDORET", "KITALE", "NAIROBI_CBD"])
            )
            destinations.append(dest)
        
        sla_hours = self.config.sla_hours
        deadline = timestamp + timedelta(hours=sla_hours)
        
        return Shipment(
            id=f"sim_{uuid.uuid4().hex[:8]}",
            customer_id=f"sim_customer_{uuid.uuid4().hex[:8]}",
            origin=origin,
            destinations=destinations,
            volume=volume,
            weight=weight,
            creation_time=timestamp,
            deadline=deadline,
            status=ShipmentStatus.PENDING,
            priority=1.0
        )