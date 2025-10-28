# src/core/decision_engine.py
"""
Decision Engine: Main orchestration with closed learning loop
FIXES:
1. Proper PFA batch format conversion with 'id' key
2. Add missing _trigger_learning_update method
3. Fix reward calculator action format (include batches in normalized action)
4. Complete TD learning integration
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from enum import Enum
from uuid import uuid4
import numpy as np
from .state_manager import (
    StateManager, SystemState, Shipment, ShipmentStatus, VehicleState,
    Route, DecisionEvent, VehicleStatus
)
from .standard_types import StandardBatch
from .meta_controller import MetaController, MetaDecision, FunctionClass
from .vfa_neural import NeuralVFA as ValueFunctionApproximator
from .reward_calculator import RewardCalculator, RewardComponents
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class CycleResult:
    """Results from a single decision cycle"""
    cycle_number: int
    timestamp: datetime
    state_before: SystemState
    decision: MetaDecision
    reward_components: RewardComponents
    state_after: SystemState
    execution_time_ms: float
    shipments_dispatched: int
    vehicles_utilized: int
    learning_updates_applied: bool
    vfa_value_before: float
    vfa_value_after: float
    td_error: float


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    total_cycles: int
    total_shipments_processed: int
    total_dispatches: int
    avg_utilization: float
    avg_reward_per_cycle: float
    avg_cycle_time_ms: float
    function_class_usage: Dict[str, int]
    learning_convergence: float


class DecisionEngine:
    """
    Enhanced Decision Engine with Closed Learning Loop
    
    Mathematical Foundation:
    At each decision epoch t:
    1. Observe state S_t
    2. Estimate V(S_t) using VFA
    3. Make decision a_t via meta-controller
    4. Execute action, observe S_{t+1}
    5. Calculate reward r_t
    6. Update VFA: θ ← θ + α * δ_t * e_t
       where δ_t = r_t + γ*V(S_{t+1}) - V(S_t)
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.meta_controller = MetaController()
        self.vfa = ValueFunctionApproximator()
        self.reward_calculator = RewardCalculator()
        
        self.status = EngineStatus.IDLE
        self.current_cycle = 0
        self.cycle_history: List[CycleResult] = []
        
        logger.info("Decision Engine initialized")
    
    def run_cycle(self, current_time: Optional[datetime] = None) -> CycleResult:
        """
        Run a single decision cycle with complete learning loop
        
        Args:
            current_time: Current timestamp (defaults to now)
            
        Returns:
            CycleResult with decision and execution details
        """
        if current_time is None:
            current_time = datetime.now()
        
        start_time = datetime.now()
        self.status = EngineStatus.RUNNING
        
        try:
            # Step 1: Get current state
            state_before = self.state_manager.get_current_state()
            logger.info(f"Cycle {self.current_cycle}: {len(state_before.pending_shipments)} pending shipments")
            
            # Step 2: Make decision via meta-controller
            decision = self.meta_controller.decide(state_before)
            logger.info(
                f"Decision: {decision.function_class.value} -> {decision.action_type} "
                f"(confidence: {decision.confidence:.2f})"
            )
            
            # Step 3: Execute decision
            execution_result = self._execute_decision(decision, state_before)
            
            # Step 4: Get updated state
            state_after = self.state_manager.get_current_state()
            
            # Step 5: Calculate reward
            # FIX: Normalize action structure to include batches
            normalized_action = self._normalize_action(decision, execution_result)
            
            reward_dict = self.reward_calculator.calculate_reward(
                state_before,
                normalized_action,
                state_after
            )
            
            # Convert dict to RewardComponents
            reward_components = RewardComponents(
                utilization_reward=reward_dict.get('utilization_bonus', 0.0),
                on_time_reward=reward_dict.get('timeliness_bonus', 0.0),
                cost_penalty=reward_dict.get('operational_cost', 0.0),
                late_penalty=reward_dict.get('delay_penalty', 0.0),
                efficiency_bonus=0.0,
                total_reward=reward_dict.get('total', 0.0),
                reasoning=""
            )
            
            # Step 6: Trigger learning update
            learning_result = self._trigger_learning_update(
                state_before=state_before,
                action=normalized_action,
                reward=reward_components.total_reward,
                state_after=state_after
            )
            
            # Step 7: Record cycle result
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = CycleResult(
                cycle_number=self.current_cycle,
                timestamp=current_time,
                state_before=state_before,
                decision=decision,
                reward_components=reward_components,
                state_after=state_after,
                execution_time_ms=execution_time,
                shipments_dispatched=execution_result['shipments_dispatched'],
                vehicles_utilized=execution_result['vehicles_utilized'],
                learning_updates_applied=learning_result['updated'],
                vfa_value_before=learning_result['value_before'],
                vfa_value_after=learning_result['value_after'],
                td_error=learning_result['td_error']
            )
            
            self.cycle_history.append(result)
            self.current_cycle += 1
            self.status = EngineStatus.IDLE
            
            logger.info(
                f"Cycle completed in {execution_time:.1f}ms | "
                f"Reward: {reward_components.total_reward:.1f} | "
                f"TD Error: {learning_result['td_error']:.2f}"
            )
            return result
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Cycle failed: {str(e)}", exc_info=True)
            raise
    
    def _normalize_action(self, decision: MetaDecision, execution_result: Dict) -> dict:
        """
        Normalize action structure for reward calculation
        
        CRITICAL FIX: Include batches in the normalized action so reward calculator
        can properly calculate utilization bonuses
        
        Mathematical Foundation:
        Action space A must have consistent representation:
        a ∈ A where a = {type: ActionType, batches: List[Batch], ...}
        
        Args:
            decision: MetaDecision from meta-controller
            execution_result: Result from _execute_decision with batch info
            
        Returns:
            Normalized action dict with 'type' and 'batches' keys
        """
        normalized = {
            'type': decision.action_type,
            'function_class': decision.function_class.value,
            'confidence': decision.confidence
        }
        
        # CRITICAL: Include batches for reward calculation
        # Extract batches from decision.action_details OR from execution_result
        if 'batches' in decision.action_details:
            normalized['batches'] = decision.action_details['batches']
        elif 'executed_batches' in execution_result:
            normalized['batches'] = execution_result['executed_batches']
        else:
            normalized['batches'] = []
        
        return normalized
    
    def _execute_decision(self, decision: MetaDecision, state: SystemState) -> Dict:
        """
        Execute the decision and update system state
        
        FIX: Properly handle PFA format conversion and return executed batches
        
        Args:
            decision: Decision to execute
            state: Current state
            
        Returns:
            Dict with execution results and executed_batches for reward calculation
        """
        if decision.action_type == 'WAIT':
            return {
                'shipments_dispatched': 0,
                'vehicles_utilized': 0,
                'routes_created': 0,
                'executed_batches': []
            }
        
        elif decision.action_type in ['DISPATCH', 'DISPATCH_IMMEDIATE']:
            # Get batches or convert old PFA format
            batches = decision.action_details.get('batches', [])
            
            # FIX: Handle old PFA format (no 'batches' key, just 'shipments' and 'vehicle')
            if not batches and 'shipments' in decision.action_details:
                batch_id = f"pfa_batch_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                pfa_batch = {
                    'id': batch_id,  # FIX: Add missing 'id' key
                    'shipments': decision.action_details['shipments'],
                    'vehicle': decision.action_details['vehicle'],
                    'route': [],
                    'sequence': [],
                    'estimated_distance_km': 0.0,
                    'estimated_duration_hours': 3.0,
                    'estimated_cost': 5000.0,  # Add for reward calculation
                    'utilization': 0.8  # Add for reward calculation
                }
                batches = [pfa_batch]
                logger.info(f"Converted old PFA format to batch: {batch_id}")
            
            if not batches:
                logger.warning("No batches to execute")
                return {
                    'shipments_dispatched': 0,
                    'vehicles_utilized': 0,
                    'routes_created': 0,
                    'executed_batches': []
                }
            
            # Execute batches
            shipments_dispatched = 0
            vehicles_utilized = set()
            routes_created = []
            executed_batches = []
            
            for batch in batches:
                try:
                    # Validate batch has required fields
                    if 'shipments' not in batch or 'vehicle' not in batch:
                        logger.error(f"Invalid batch missing required fields: {list(batch.keys())}")
                        continue
                    
                    if 'id' not in batch:
                        batch['id'] = f"batch_{uuid4().hex[:8]}"
                    
                    # Dispatch batch
                    route_result = self._dispatch_batch(batch, state)
                    
                    shipments_dispatched += route_result['shipments']
                    if route_result['vehicle']:
                        vehicles_utilized.add(route_result['vehicle'])
                    routes_created.append(batch['id'])
                    
                    # Store batch for reward calculation
                    executed_batches.append(batch)
                    
                    logger.info(f"✓ Batch {batch['id']}: {len(batch['shipments'])} shipments dispatched")
                    
                except Exception as e:
                    logger.error(f"✗ Batch dispatch failed: {e}")
                    continue
            
            logger.info(
                f"Executed DISPATCH: {shipments_dispatched} shipments, "
                f"{len(vehicles_utilized)} vehicles, {len(routes_created)} routes"
            )
            
            return {
                'shipments_dispatched': shipments_dispatched,
                'vehicles_utilized': len(vehicles_utilized),
                'routes_created': len(routes_created),
                'executed_batches': executed_batches  # FIX: Return for reward calculation
            }
        
        else:
            logger.warning(f"Unknown action type: {decision.action_type}")
            return {
                'shipments_dispatched': 0,
                'vehicles_utilized': 0,
                'routes_created': 0,
                'executed_batches': []
            }
    
    def _trigger_learning_update(self, state_before: SystemState, action: Dict,
                             reward: float, state_after: SystemState) -> Dict:
            """
            Trigger tactical VFA learning update
            Week 5 addition for tactical learning
            """
            # Extract features from state
            features_before = self._extract_state_features(state_before)
            features_after = self._extract_state_features(state_after)
            
            # Get VFA value estimates
            value_before = self.vfa.evaluate(features_before)
            value_after = self.vfa.evaluate(features_after)
            
            # Compute TD error: δ = r + γV(s') - V(s)
            gamma = self.config.learning_config.get('discount_factor', 0.95)
            td_error = reward + gamma * value_after - value_before
            
            # FIX: Update VFA with correct signature (state, actual_outcome only)
            target = reward + gamma * value_after
            self.vfa.update(
                state=features_before,
                actual_outcome=target
            )
            
            # Log for multi-scale coordinator
            try:
                self.state_manager.log_learning_update(
                    update_type='tactical_td',
                    state_features=features_before,
                    td_error=td_error,
                    actual_reward=reward,
                    predicted_reward=value_before
                )
            except Exception as e:
                logger.warning(f"Failed to log learning update: {e}")
            
            return {
                'updated': True,
                'value_before': value_before,
                'value_after': value_after,
                'td_error': td_error
            }
    
    def _extract_state_features(self, state: SystemState) -> Dict:
        """Extract VFA features from state"""
        try:
            urgent_count = len(state.get_urgent_shipments(24))
            avg_urgency = 0
            if state.pending_shipments:
                urgencies = [s.urgency_score for s in state.pending_shipments if hasattr(s, 'urgency_score')]
                avg_urgency = np.mean(urgencies) if urgencies else 0
            
            return {
                'pending_count': len(state.pending_shipments),
                'fleet_available': len(state.get_available_vehicles()),
                'avg_urgency': avg_urgency,
                'urgent_count': urgent_count,
                'total_volume': state.total_pending_volume(),
                'total_weight': state.total_pending_weight(),
                'fleet_utilization': state.fleet_utilization()
            }
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return {
                'pending_count': len(state.pending_shipments),
                'fleet_available': 0,
                'avg_urgency': 0,
                'urgent_count': 0,
                'total_volume': 0,
                'total_weight': 0,
                'fleet_utilization': 0
            }
    def _extract_state_features(self, state: SystemState) -> Dict:
        """Extract VFA features from state"""
        return {
            'pending_count': len(state.pending_shipments),
            'fleet_available': len(state.get_available_vehicles()),
            'avg_urgency': np.mean([s.urgency_score for s in state.pending_shipments]) if state.pending_shipments else 0,
            'total_volume': state.total_pending_volume(),
            'total_weight': state.total_pending_weight(),
            'fleet_utilization': state.fleet_utilization()
        }

    def _dispatch_batch(self, batch: Dict, state: SystemState) -> Dict:
        """
        Dispatch a single batch by updating shipment statuses
        
        Args:
            batch: Batch dict with shipments and vehicle
            state: Current state
            
        Returns:
            Dict with dispatch results and state_snapshot for learning
        """
        shipments = batch.get('shipments', [])
        vehicle_id = batch.get('vehicle_id') or batch.get('vehicle')
        route = batch.get('route_stops', batch.get('route', []))
        
        if not shipments:
            return {'shipments': 0, 'vehicle': None, 'state_snapshot': {}}
        
        # Create route
        route_id = batch.get('id', f"route_{uuid4().hex[:8]}")
        
        # Update each shipment status - FIX: use correct signature
        for shipment_id in shipments:
            success = self.state_manager.update_shipment_status(
                shipment_id=shipment_id,
                status=ShipmentStatus.EN_ROUTE,  # FIX: not 'new_status'
                batch_id=batch.get('id'),
                route_id=route_id
            )
            
            if not success:
                logger.warning(f"Failed to update shipment {shipment_id}")
        
        # Create state snapshot for Week 5 learning
        state_snapshot = {
            'pending_count': len(state.pending_shipments),
            'fleet_available': len(state.get_available_vehicles()),
            'batch_id': batch.get('id'),
            'vehicle_id': vehicle_id,
            'shipment_count': len(shipments),
            'estimated_cost': batch.get('estimated_cost', 0),
            'estimated_duration_hours': batch.get('estimated_duration_hours', 0),
            'utilization_volume': batch.get('utilization_volume', 0),
            'origin_lat': route[0]['lat'] if route and len(route) > 0 else 0,
            'origin_lon': route[0]['lon'] if route and len(route) > 0 else 0,
            'dest_lat': route[-1]['lat'] if route and len(route) > 1 else 0,
            'dest_lon': route[-1]['lon'] if route and len(route) > 1 else 0,
        }
        
        logger.info(f"Dispatched batch {route_id}: {len(shipments)} shipments via {vehicle_id}")
        
        return {
            'shipments': len(shipments),
            'vehicle': vehicle_id,
            'route_id': route_id,
            'state_snapshot': state_snapshot  # Week 5: for learning
        }
        
    def _trigger_learning_update(
        self,
        state_before: SystemState,
        action: dict,
        reward: float,
        state_after: SystemState
    ) -> Dict:
        """
        Trigger VFA learning update using TD(λ)
        
        Mathematical Foundation:
        TD Error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        Weight Update: θ ← θ + α * δ_t * e_t
        
        Where:
        - r_t: immediate reward
        - γ: discount factor (typically 0.95)
        - V(s): value function estimate
        - α: learning rate
        - e_t: eligibility trace
        
        Args:
            state_before: State before action
            action: Action taken (normalized format)
            reward: Immediate reward
            state_after: State after action
            
        Returns:
            Dict with learning metrics
        """
        # Evaluate value functions
        vfa_eval_before = self.vfa.evaluate(state_before)
        vfa_eval_after = self.vfa.evaluate(state_after)
        
        value_before = vfa_eval_before.value
        value_after = vfa_eval_after.value
        
        # Calculate TD error
        gamma = self.config.get('vfa.learning.discount_factor', 0.95)
        td_error = reward + gamma * value_after - value_before
        
        # Update VFA weights using TD(λ)
        self.vfa.td_update(
            s_t=state_before,
            action=action,
            reward=reward,
            s_tp1=state_after
        )
        
        logger.debug(
            f"Learning update: V(s_t)={value_before:.1f}, "
            f"V(s_t+1)={value_after:.1f}, r={reward:.1f}, δ={td_error:.2f}"
        )
        
        return {
            'updated': True,
            'value_before': value_before,
            'value_after': value_after,
            'td_error': td_error
        }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics from cycle history"""
        if not self.cycle_history:
            return PerformanceMetrics(
                total_cycles=0,
                total_shipments_processed=0,
                total_dispatches=0,
                avg_utilization=0.0,
                avg_reward_per_cycle=0.0,
                avg_cycle_time_ms=0.0,
                function_class_usage={},
                learning_convergence=0.0
            )
        
        total_shipments = sum(r.shipments_dispatched for r in self.cycle_history)
        total_dispatches = sum(1 for r in self.cycle_history if r.shipments_dispatched > 0)
        avg_reward = sum(r.reward_components.total_reward for r in self.cycle_history) / len(self.cycle_history)
        avg_cycle_time = sum(r.execution_time_ms for r in self.cycle_history) / len(self.cycle_history)
        
        # Function class usage
        function_usage = {}
        for result in self.cycle_history:
            fc = result.decision.function_class.value
            function_usage[fc] = function_usage.get(fc, 0) + 1
        
        # Learning convergence (based on recent TD errors)
        recent_td_errors = [abs(r.td_error) for r in self.cycle_history[-20:]]
        learning_convergence = 1.0 / (1.0 + sum(recent_td_errors) / len(recent_td_errors))
        
        return PerformanceMetrics(
            total_cycles=len(self.cycle_history),
            total_shipments_processed=total_shipments,
            total_dispatches=total_dispatches,
            avg_utilization=0.0,  # TODO: Calculate from batches
            avg_reward_per_cycle=avg_reward,
            avg_cycle_time_ms=avg_cycle_time,
            function_class_usage=function_usage,
            learning_convergence=learning_convergence
        )