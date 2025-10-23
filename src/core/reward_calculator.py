# src/core/reward_calculator.py

"""
Reward Function for Sequential Decision Learning

Captures Senga's business objectives:
- High vehicle utilization
- On-time delivery (SLA compliance)
- Cost minimization
- Customer satisfaction
- Operational efficiency
"""

from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from .state_manager import SystemState, Shipment, Route, VehicleState
from ..config.senga_config import SengaConfigurator

@dataclass
class RewardComponents:
    """Breakdown of reward calculation for interpretability"""
    utilization_reward: float
    on_time_reward: float
    cost_penalty: float
    late_penalty: float
    efficiency_bonus: float
    total_reward: float
    reasoning: str

class RewardCalculator:
    """
    Reward Function: r(s_t, a_t, s_{t+1})
    
    Mathematical Foundation:
    r = w_util * r_utilization 
        + w_sla * r_sla 
        - w_cost * cost 
        - w_late * late_penalty 
        + w_eff * efficiency_bonus
    
    Where:
    - r_utilization: Vehicle capacity utilization score
    - r_sla: On-time delivery reward
    - cost: Operational costs (fuel, driver time)
    - late_penalty: Heavy penalty for missed deadlines
    - efficiency_bonus: Bonus for route quality (low deadhead miles, good sequencing)
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        
        # Reward weights (tunable hyperparameters)
        self.w_utilization = self.config.get('reward.weights.utilization', 1000.0)
        self.w_sla = self.config.get('reward.weights.sla_compliance', 2000.0)
        self.w_cost = self.config.get('reward.weights.cost', 1.0)
        self.w_late = self.config.get('reward.weights.late_penalty', 5000.0)
        self.w_efficiency = self.config.get('reward.weights.efficiency', 500.0)
        
        # Thresholds
        self.min_acceptable_utilization = self.config.min_utilization
        self.sla_target_hours = self.config.sla_hours
    
    def calculate_reward(self, 
                        state_before: SystemState,
                        action: Dict,
                        state_after: SystemState) -> RewardComponents:
        """
        Calculate reward for transitioning from state_before to state_after via action
        
        Args:
            state_before: State S_t before decision
            action: Action a_t taken (dispatch or wait)
            state_after: Resulting state S_{t+1}
            
        Returns:
            RewardComponents with detailed breakdown
        """
        
        if action['type'] == 'WAIT':
            return self._reward_for_wait(state_before, state_after)
        elif action['type'] == 'DISPATCH':
            return self._reward_for_dispatch(state_before, action, state_after)
        else:
            # Unknown action type
            return RewardComponents(
                utilization_reward=0.0,
                on_time_reward=0.0,
                cost_penalty=0.0,
                late_penalty=0.0,
                efficiency_bonus=0.0,
                total_reward=0.0,
                reasoning="Unknown action type"
            )
    
    def _reward_for_wait(self, state_before: SystemState, state_after: SystemState) -> RewardComponents:
        """
        Reward for WAIT decision
        
        Waiting is good if:
        - We're close to achieving high utilization with more shipments
        - No urgent deliveries are at risk
        
        Waiting is bad if:
        - Shipments are approaching deadlines
        - Already have good consolidation opportunities
        """
        
        # Check if any shipments became urgent
        urgent_before = len(state_before.get_urgent_shipments(self.sla_target_hours))
        urgent_after = len(state_after.get_urgent_shipments(self.sla_target_hours))
        
        urgency_penalty = 0.0
        if urgent_after > urgent_before:
            # Shipments became more urgent while waiting
            urgency_penalty = -100.0 * (urgent_after - urgent_before)
        
        # Opportunity cost: could we have dispatched with good utilization?
        potential_utilization = self._estimate_potential_utilization(state_before)
        opportunity_cost = 0.0
        if potential_utilization >= self.min_acceptable_utilization:
            # We had a good dispatch opportunity but chose to wait
            opportunity_cost = -50.0
        
        # Small penalty for holding shipments
        holding_penalty = -10.0 * len(state_before.pending_shipments)
        
        total_reward = urgency_penalty + opportunity_cost + holding_penalty
        
        return RewardComponents(
            utilization_reward=0.0,
            on_time_reward=urgency_penalty,
            cost_penalty=holding_penalty,
            late_penalty=0.0,
            efficiency_bonus=opportunity_cost,
            total_reward=total_reward,
            reasoning=f"WAIT: urgency_penalty={urgency_penalty:.1f}, opportunity_cost={opportunity_cost:.1f}"
        )
    
    def _reward_for_dispatch(self, state_before: SystemState, 
                           action: Dict, 
                           state_after: SystemState) -> RewardComponents:
        """
        Reward for DISPATCH decision
        
        Evaluates the quality of the dispatched routes
        """
        
        # Extract dispatched batches from action
        batches = action.get('batches', [])
        
        if not batches:
            return RewardComponents(
                utilization_reward=0.0,
                on_time_reward=0.0,
                cost_penalty=0.0,
                late_penalty=0.0,
                efficiency_bonus=0.0,
                total_reward=0.0,
                reasoning="No batches in dispatch action"
            )
        
        # Component 1: Utilization Reward
        utilizations = [b['utilization'] for b in batches]
        avg_utilization = np.mean(utilizations)
        
        # Reward high utilization, penalize low utilization
        if avg_utilization >= self.min_acceptable_utilization:
            utilization_reward = self.w_utilization * (avg_utilization - self.min_acceptable_utilization)
        else:
            # Heavy penalty for dispatching under-utilized vehicles
            utilization_reward = -self.w_utilization * (self.min_acceptable_utilization - avg_utilization) * 2
        
        # Component 2: On-Time Delivery Reward
        # Estimate if shipments will be delivered on-time based on dispatch time + route duration
        on_time_reward = 0.0
        late_penalty = 0.0
        
        for batch in batches:
            batch_shipment_ids = set(batch['shipments'])
            estimated_delivery_time = state_before.timestamp + timedelta(hours=batch.get('estimated_duration_hours', 3))
            
            for shipment in state_before.pending_shipments:
                if shipment.id in batch_shipment_ids:
                    time_to_deadline = (shipment.delivery_deadline - estimated_delivery_time).total_seconds() / 3600
                    
                    if time_to_deadline > 0:
                        # On-time delivery expected
                        on_time_reward += self.w_sla
                        
                        # Extra bonus for delivering well before deadline
                        if time_to_deadline > 12:
                            on_time_reward += 500.0
                    else:
                        # Late delivery expected - heavy penalty
                        hours_late = abs(time_to_deadline)
                        late_penalty += self.w_late * hours_late
        
        # Component 3: Cost Penalty
        total_cost = sum(b['estimated_cost'] for b in batches)
        cost_penalty = -self.w_cost * total_cost
        
        # Component 4: Efficiency Bonus
        # Reward good route quality (low deadhead miles, compact routes)
        efficiency_bonus = 0.0
        
        for batch in batches:
            # Bonus for consolidating multiple shipments
            num_shipments = len(batch['shipments'])
            if num_shipments >= 3:
                efficiency_bonus += 200.0 * (num_shipments - 2)
            
            # Bonus for high density (many deliveries per km)
            distance_km = batch.get('estimated_distance_km', 50.0)
            deliveries_per_km = num_shipments / max(distance_km, 1)
            if deliveries_per_km > 0.1:  # More than 1 delivery per 10km
                efficiency_bonus += 300.0
        
        # Total reward
        total_reward = (
            utilization_reward +
            on_time_reward +
            cost_penalty +
            late_penalty +
            efficiency_bonus
        )
        
        reasoning = (
            f"DISPATCH: util={avg_utilization:.2%} ({utilization_reward:.0f}), "
            f"on_time_bonus={on_time_reward:.0f}, late_penalty={late_penalty:.0f}, "
            f"cost={cost_penalty:.0f}, efficiency={efficiency_bonus:.0f}"
        )
        
        return RewardComponents(
            utilization_reward=utilization_reward,
            on_time_reward=on_time_reward,
            cost_penalty=cost_penalty,
            late_penalty=late_penalty,
            efficiency_bonus=efficiency_bonus,
            total_reward=total_reward,
            reasoning=reasoning
        )
    
    def calculate_actual_reward(self, route: Route) -> float:
        """
        Calculate actual reward after route completion
        
        This is called when we have ground truth data about:
        - Actual delivery times
        - Actual costs
        - Customer satisfaction
        
        Used for learning and performance evaluation
        """
        
        if not route.completed_at:
            return 0.0  # Route not completed yet
        
        actual_reward = 0.0
        
        # Actual utilization
        # (Would need to get shipment details from route)
        
        # Actual on-time performance
        # Compare route.completed_at with shipment deadlines
        
        # Actual cost
        actual_cost = route.actual_distance * self.config.cost_per_km
        actual_reward -= self.w_cost * actual_cost
        
        # Duration accuracy bonus (reward good predictions)
        if route.actual_duration and route.estimated_duration:
            prediction_error = abs(
                (route.actual_duration - route.estimated_duration).total_seconds() / 3600
            )
            if prediction_error < 0.5:  # Within 30 minutes
                actual_reward += 200.0
        
        return actual_reward
    
    def _estimate_potential_utilization(self, state: SystemState) -> float:
        """
        Estimate the best utilization achievable with current pending shipments
        """
        if not state.pending_shipments:
            return 0.0
        
        available_vehicles = state.get_available_vehicles()
        if not available_vehicles:
            return 0.0
        
        # Simple greedy estimate: pack shipments into vehicles
        total_volume = sum(s.volume for s in state.pending_shipments)
        best_vehicle = max(available_vehicles, key=lambda v: v.capacity.volume)
        
        return min(1.0, total_volume / best_vehicle.capacity.volume)