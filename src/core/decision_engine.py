# src/core/decision_engine.py

"""
Enhanced Decision Engine: Main orchestration with closed learning loop
Implements autonomous decision cycles and TD learning updates
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from enum import Enum
from uuid import uuid4
from .state_manager import (
    StateManager, SystemState, Shipment, ShipmentStatus, VehicleState,
    Route, DecisionEvent
)
from .meta_controller import MetaController, MetaDecision, FunctionClass
from .vfa import ValueFunctionApproximator
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
    learning_convergence: float  # Based on recent TD errors

class DecisionEngine:
    """
    Enhanced Decision Engine with Closed Learning Loop
    
    Key Enhancements:
    1. Autonomous decision cycles (runs on schedule)
    2. Post-decision state observation
    3. Reward calculation
    4. VFA TD learning updates
    5. Performance tracking
    
    Mathematical Foundation:
    At each decision epoch t:
    1. Observe state S_t
    2. Estimate V(S_t) using VFA
    3. Make decision a_t via meta-controller
    4. Execute action, observe S_{t+1}
    5. Calculate reward r_t
    6. Update VFA: θ ← θ + α * δ_t * e_t
       where δ_t = r_t + γ*V(S_{t+1}) - V(S_t)
    7. Loop to next epoch
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
        self.autonomous_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced Decision Engine initialized with learning loop")
    
    async def start_autonomous_mode(self, cycle_interval_minutes: int = 60):
        """
        Start autonomous decision-making mode
        
        The engine will make decisions automatically every cycle_interval_minutes
        This is the core of sequential decision-making - decisions happen over time
        
        Args:
            cycle_interval_minutes: Time between decision cycles (default 60 = hourly)
        """
        if self.status == EngineStatus.AUTONOMOUS:
            logger.warning("Already running in autonomous mode")
            return
        
        self.status = EngineStatus.AUTONOMOUS
        logger.info(f"Starting autonomous mode: decisions every {cycle_interval_minutes} minutes")
        
        while self.status == EngineStatus.AUTONOMOUS:
            try:
                # Run decision cycle
                cycle_result = self.run_cycle()
                
                logger.info(
                    f"Autonomous cycle {cycle_result.cycle_number} complete: "
                    f"Function={cycle_result.decision.function_class.value}, "
                    f"Action={cycle_result.decision.action_type}, "
                    f"Reward={cycle_result.reward_components.total_reward:.1f}, "
                    f"TD_error={cycle_result.td_error:.2f}"
                )
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in autonomous cycle: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_autonomous_mode(self):
        """Stop autonomous decision-making"""
        if self.status == EngineStatus.AUTONOMOUS:
            self.status = EngineStatus.IDLE
            logger.info("Stopped autonomous mode")
    
    def run_cycle(self, current_time: Optional[datetime] = None) -> CycleResult:
        """
        Run a single decision cycle with closed learning loop
        
        This is the core sequential decision process:
        1. Observe current state S_t
        2. Estimate value V(S_t)
        3. Make decision via meta-controller
        4. Execute decision
        5. Observe new state S_{t+1}
        6. Calculate reward r_t
        7. Update VFA with TD learning
        8. Record results
        
        Args:
            current_time: Current timestamp (defaults to now)
            
        Returns:
            CycleResult with complete cycle information
        """
        if current_time is None:
            current_time = datetime.now()
        
        start_time = datetime.now()
        self.status = EngineStatus.RUNNING
        self.current_cycle += 1
        
        try:
            # === STEP 1: Observe State S_t ===
            state_before = self.state_manager.get_current_state()
            logger.info(
                f"Cycle {self.current_cycle}: "
                f"{len(state_before.pending_shipments)} pending, "
                f"{len(state_before.get_available_vehicles())} vehicles available"
            )
            
            # === STEP 2: Estimate V(S_t) ===
            value_estimate_before = self.vfa.estimate_value(state_before)
            logger.debug(f"V(S_t) = {value_estimate_before.value:.2f}")
            
            # === STEP 3: Make Decision ===
            decision = self.meta_controller.decide(state_before)
            logger.info(
                f"Decision: {decision.function_class.value} -> {decision.action_type} "
                f"(confidence: {decision.confidence:.2f})"
            )
            
            # === STEP 4: Execute Decision ===
            execution_result = self._execute_decision(decision, state_before)
            
            # === STEP 5: Observe New State S_{t+1} ===
            state_after = self.state_manager.get_current_state()
            value_estimate_after = self.vfa.estimate_value(state_after)
            logger.debug(f"V(S_{{t+1}}) = {value_estimate_after.value:.2f}")
            
            # === STEP 6: Calculate Reward r_t ===
            reward_components = self.reward_calculator.calculate_reward(
                state_before,
                decision.action_details,
                state_after
            )
            logger.info(f"Reward: {reward_components.total_reward:.1f} - {reward_components.reasoning}")
            
            # === STEP 7: VFA TD Learning Update ===
            self.vfa.update(
                state_before,
                decision.action_details,
                reward_components.total_reward,
                state_after
            )
            
            td_error = (
                reward_components.total_reward + 
                self.vfa.discount_factor * value_estimate_after.value - 
                value_estimate_before.value
            )
            
            # === STEP 8: Record Results ===
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            cycle_result = CycleResult(
                cycle_number=self.current_cycle,
                timestamp=current_time,
                state_before=state_before,
                decision=decision,
                reward_components=reward_components,
                state_after=state_after,
                execution_time_ms=execution_time,
                shipments_dispatched=execution_result['shipments_dispatched'],
                vehicles_utilized=execution_result['vehicles_utilized'],
                learning_updates_applied=True,
                vfa_value_before=value_estimate_before.value,
                vfa_value_after=value_estimate_after.value,
                td_error=td_error
            )
            
            self.cycle_history.append(cycle_result)
            
            # Log decision event to database
            self._log_decision_event(cycle_result)
            
            self.status = EngineStatus.IDLE
            return cycle_result
            
        except Exception as e:
            logger.error(f"Error in decision cycle: {e}", exc_info=True)
            self.status = EngineStatus.ERROR
            raise
    
    def _execute_decision(self, decision: MetaDecision, state: SystemState) -> Dict:
        """
        Execute the decision and update system state
        
        Args:
            decision: Decision to execute
            state: Current state
            
        Returns:
            Execution results (shipments dispatched, vehicles used, etc.)
        """
        if decision.action_type == 'WAIT':
            return {
                'shipments_dispatched': 0,
                'vehicles_utilized': 0,
                'routes_created': 0
            }
        
        elif decision.action_type == 'DISPATCH':
            batches = decision.action_details.get('batches', [])
            
            shipments_dispatched = 0
            vehicles_utilized = len(batches)
            routes_created = []
            
            for batch in batches:
                # Create route
                route = Route(
                    id=batch['id'],
                    vehicle_id=batch['vehicle'],
                    shipment_ids=batch['shipments'],
                    sequence=batch.get('sequence', []),
                    estimated_duration=timedelta(hours=batch.get('estimated_duration_hours', 3)),
                    estimated_distance=batch.get('estimated_distance_km', 50.0),
                    created_at=datetime.now()
                )
                
                # Add route to state
                self.state_manager.add_route(route)
                routes_created.append(route)
                
                # Update shipment statuses
                for shipment_id in batch['shipments']:
                    self.state_manager.update_shipment_status(
                        shipment_id,
                        ShipmentStatus.IN_TRANSIT
                    )
                    shipments_dispatched += 1
                
                # Update vehicle status
                self.state_manager.update_vehicle_status(
                    batch['vehicle'],
                    'in_transit'
                )
            
            logger.info(
                f"Executed DISPATCH: {shipments_dispatched} shipments, "
                f"{vehicles_utilized} vehicles, {len(routes_created)} routes"
            )
            
            return {
                'shipments_dispatched': shipments_dispatched,
                'vehicles_utilized': vehicles_utilized,
                'routes_created': len(routes_created)
            }
        
        else:
            logger.warning(f"Unknown action type: {decision.action_type}")
            return {
                'shipments_dispatched': 0,
                'vehicles_utilized': 0,
                'routes_created': 0
            }
    
    def _log_decision_event(self, cycle_result: CycleResult):
        """Log decision event to database for analysis"""
        event = DecisionEvent(
            timestamp=cycle_result.timestamp,
            id=f"decision_{uuid4().hex[:8]}",
            function_class=cycle_result.decision.function_class.value,
            decision_type=cycle_result.decision.action_type,
            reasoning=cycle_result.decision.reasoning,
            confidence=cycle_result.decision.confidence,
            reward=cycle_result.reward_components.total_reward,
            vfa_value_before=cycle_result.vfa_value_before,
            vfa_value_after=cycle_result.vfa_value_after,
            td_error=cycle_result.td_error,
            state_snapshot=cycle_result.state_before,  # Simplified
            alternatives_considered=[],
            action_details=cycle_result.decision.action_details,
        )
        
        self.state_manager.log_decision(event)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
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
        
        total_shipments = sum(c.shipments_dispatched for c in self.cycle_history)
        total_dispatches = sum(1 for c in self.cycle_history if c.decision.action_type == 'DISPATCH')
        
        avg_reward = np.mean([c.reward_components.total_reward for c in self.cycle_history])
        avg_time = np.mean([c.execution_time_ms for c in self.cycle_history])
        
        # Function class usage
        function_usage = {}
        for c in self.cycle_history:
            fc = c.decision.function_class.value
            function_usage[fc] = function_usage.get(fc, 0) + 1
        
        # Learning convergence (based on recent TD errors)
        recent_td_errors = [abs(c.td_error) for c in self.cycle_history[-100:]]
        learning_convergence = 1.0 / (1.0 + np.mean(recent_td_errors)) if recent_td_errors else 0.0
        
        # Utilization (would need batch details)
        avg_utilization = 0.75  # Placeholder
        
        return PerformanceMetrics(
            total_cycles=len(self.cycle_history),
            total_shipments_processed=total_shipments,
            total_dispatches=total_dispatches,
            avg_utilization=avg_utilization,
            avg_reward_per_cycle=avg_reward,
            avg_cycle_time_ms=avg_time,
            function_class_usage=function_usage,
            learning_convergence=learning_convergence
        )
    
    def get_recent_cycles(self, n: int = 20) -> List[Dict]:
        """Get recent decision cycles for dashboard display"""
        recent = self.cycle_history[-n:]
        return [
            {
                'cycle_number': c.cycle_number,
                'timestamp': c.timestamp.isoformat(),
                'function_class': c.decision.function_class.value,
                'action_type': c.decision.action_type,
                'reward': round(c.reward_components.total_reward, 1),
                'td_error': round(c.td_error, 2),
                'shipments_dispatched': c.shipments_dispatched,
                'vehicles_utilized': c.vehicles_utilized,
                'confidence': round(c.decision.confidence, 2),
                'vfa_value': round(c.vfa_value_before, 1)
            }
            for c in recent
        ]