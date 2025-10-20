# src/core/decision_engine.py

"""
Decision Engine: Main orchestration loop for Senga SDA
Runs hourly consolidation cycles and coordinates all components
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from .state_manager import (
    StateManager, SystemState, Shipment, VehicleState,
    ShipmentStatus, VehicleStatus, DecisionEvent
)
from .meta_controller import MetaController, MetaDecision, FunctionClass
from .vfa import ValueFunctionApproximator
from ..config.senga_config import SengaConfigurator

logger = logging.getLogger(__name__)

class EngineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class CycleResult:
    """Results from a single decision cycle"""
    timestamp: datetime
    state_before: SystemState
    decision: MetaDecision
    state_after: SystemState
    execution_time_ms: float
    shipments_dispatched: int
    vehicles_utilized: int
    learning_updates_applied: int

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    total_cycles: int
    total_shipments_processed: int
    total_dispatches: int
    avg_utilization: float
    avg_sla_compliance: float
    avg_cycle_time_ms: float
    function_class_usage: Dict[str, int]

class DecisionEngine:
    """
    Main orchestration engine for Senga SDA
    
    Responsibilities:
    1. Run hourly consolidation cycles
    2. Coordinate state management and decision making
    3. Execute approved decisions (dispatch/wait)
    4. Trigger learning updates
    5. Monitor system performance
    6. Handle errors and recovery
    
    Mathematical Foundation:
    At each decision epoch t (hourly):
    - Observe state S_t
    - Query meta-controller for decision a_t
    - Execute action and observe outcome
    - Update value function with TD learning
    - Transition to S_{t+1}
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        self.meta_controller = MetaController()
        self.vfa = ValueFunctionApproximator()
        
        self.status = EngineStatus.IDLE
        self.current_cycle = 0
        self.cycle_history: List[CycleResult] = []
        
        logger.info("Decision Engine initialized")
    
    def run_cycle(self, current_time: Optional[datetime] = None) -> CycleResult:
        """
        Run a single decision cycle
        
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
            
            # Step 5: Trigger learning update (if dispatch occurred)
            learning_updates = 0
            if decision.action_type == 'DISPATCH':
                learning_updates = self._trigger_learning_update(
                    state_before, decision, execution_result
                )
            
            # Step 6: Record cycle result
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result = CycleResult(
                timestamp=current_time,
                state_before=state_before,
                decision=decision,
                state_after=state_after,
                execution_time_ms=execution_time,
                shipments_dispatched=execution_result['shipments_dispatched'],
                vehicles_utilized=execution_result['vehicles_utilized'],
                learning_updates_applied=learning_updates
            )
            
            self.cycle_history.append(result)
            self.current_cycle += 1
            self.status = EngineStatus.IDLE
            
            logger.info(f"Cycle completed in {execution_time:.1f}ms")
            return result
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Cycle failed: {str(e)}", exc_info=True)
            raise
    
 
    def _execute_decision(self, decision: MetaDecision, state: SystemState) -> Dict:
            """
            Execute the decision made by meta-controller
            
            Args:
                decision: Decision to execute
                state: Current state
                
            Returns:
                Execution result dictionary
            """
            from uuid import uuid4
            
            result = {
                'shipments_dispatched': 0,
                'vehicles_utilized': 0,
                'routes_created': 0
            }
            
            if decision.action_type == 'WAIT':
                logger.info("Decision: WAIT - no action taken")
                
                # Create DecisionEvent for logging
                decision_event = DecisionEvent(
                    id=f"DEC{uuid4().hex[:8].upper()}",
                    timestamp=datetime.now(),
                    state_snapshot=state,
                    decision_type='WAIT',
                    function_class=decision.function_class.value,
                    action_details={'reasoning': decision.reasoning},
                    reasoning=decision.reasoning
                )
                
                # Log the wait decision
                self.state_manager.log_decision(decision_event)
                
                return result
            
            elif decision.action_type == 'DISPATCH':
                # Extract dispatch details
                batches = decision.action_details.get('batches', [])
                
                if not batches:
                    logger.warning("DISPATCH decision but no batches provided")
                    return result
                
                # Execute each batch dispatch
                for batch in batches:
                    route_result = self._dispatch_batch(batch, state)
                    result['shipments_dispatched'] += route_result['shipments']
                    result['vehicles_utilized'] += 1
                    result['routes_created'] += 1
                
                logger.info(
                    f"Dispatched {result['shipments_dispatched']} shipments "
                    f"using {result['vehicles_utilized']} vehicles"
                )
                
                # Create DecisionEvent for logging
                decision_event = DecisionEvent(
                    id=f"DEC{uuid4().hex[:8].upper()}",
                    timestamp=datetime.now(),
                    state_snapshot=state,
                    decision_type='DISPATCH',
                    function_class=decision.function_class.value,
                    action_details={
                        **decision.action_details,
                        'outcome': result
                    },
                    reasoning=decision.reasoning
                )
                
                # Log the dispatch decision
                self.state_manager.log_decision(decision_event)
                
                return result
            
            elif decision.action_type.startswith('EMERGENCY_'):
                # Handle emergency actions
                logger.warning(f"Emergency action: {decision.action_type}")
                
                # Create DecisionEvent for logging
                decision_event = DecisionEvent(
                    id=f"DEC{uuid4().hex[:8].upper()}",
                    timestamp=datetime.now(),
                    state_snapshot=state,
                    decision_type=decision.action_type,
                    function_class=decision.function_class.value,
                    action_details=decision.action_details,
                    reasoning=decision.reasoning
                )
                
                # Log the emergency decision
                self.state_manager.log_decision(decision_event)
                
                # Emergency handling would go here (alerts, escalations, etc.)
                return result
            
            else:
                logger.error(f"Unknown action type: {decision.action_type}")
                return result
    def _dispatch_batch(self, batch: Dict, state: SystemState) -> Dict:
        """
        Dispatch a single batch (route)
        
        Args:
            batch: Batch details (shipments, vehicle, route)
            state: Current state
            
        Returns:
            Dispatch result
        """
        shipment_ids = batch.get('shipments', [])
        vehicle_id = batch.get('vehicle')
        route = batch.get('route', [])
        
        # Update shipment statuses to IN_TRANSIT
        dispatched_count = 0
        for shipment_id in shipment_ids:
            success = self.state_manager.update_shipment_status(
                shipment_id=shipment_id,
                new_status=ShipmentStatus.IN_TRANSIT,
                location=route[0] if route else None
            )
            if success:
                dispatched_count += 1
        
        # Update vehicle status to IN_TRANSIT
        if vehicle_id:
            self.state_manager.update_vehicle_status(
                vehicle_id=vehicle_id,
                new_status=VehicleStatus.IN_TRANSIT,
                current_location=route[0] if route else None,
                assigned_shipments=shipment_ids
            )
        
        return {
            'shipments': dispatched_count,
            'vehicle': vehicle_id,
            'route_stops': len(route)
        }
    
    def _trigger_learning_update(
        self,
        state_before: SystemState,
        decision: MetaDecision,
        execution_result: Dict
    ) -> int:
        """
        Trigger VFA learning update after dispatch
        
        Args:
            state_before: State before decision
            decision: Decision that was made
            execution_result: Result of execution
            
        Returns:
            Number of learning updates applied
        """
        # For now, we'll do immediate partial learning
        # Full learning happens when routes complete (handled separately)
        
        # Extract immediate reward (negative cost of dispatch)
        immediate_cost = decision.action_details.get('total_cost', 0)
        immediate_reward = -immediate_cost
        
        # Estimate next state value (simplified - would need actual next state)
        # For dispatch, next state typically has fewer pending shipments
        next_state_value = 0.0  # Conservative estimate
        
        # TD update
        self.vfa.update(
            state=state_before,
            action_value=immediate_reward + self.config.vfa_discount_factor * next_state_value,
            actual_outcome=immediate_reward
        )
        
        return 1
    
    def run_continuous(
        self,
        interval_minutes: Optional[int] = None,
        max_cycles: Optional[int] = None
    ):
        """
        Run engine in continuous mode (for production)
        
        Args:
            interval_minutes: Time between cycles (defaults to config)
            max_cycles: Maximum cycles to run (None = infinite)
        """
        if interval_minutes is None:
            interval_minutes = self.config.model_config.get('decision_cycle_minutes', 60)
        
        logger.info(f"Starting continuous mode: {interval_minutes}min cycles")
        
        cycles_run = 0
        while max_cycles is None or cycles_run < max_cycles:
            try:
                result = self.run_cycle()
                cycles_run += 1
                
                # Wait for next cycle
                if max_cycles is None or cycles_run < max_cycles:
                    import time
                    time.sleep(interval_minutes * 60)
                    
            except KeyboardInterrupt:
                logger.info("Continuous mode stopped by user")
                break
            except Exception as e:
                logger.error(f"Cycle failed: {e}", exc_info=True)
                # Could implement retry logic here
                import time
                time.sleep(60)  # Wait 1 minute before retry
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate system performance metrics
        
        Returns:
            PerformanceMetrics with aggregated statistics
        """
        if not self.cycle_history:
            return PerformanceMetrics(
                total_cycles=0,
                total_shipments_processed=0,
                total_dispatches=0,
                avg_utilization=0.0,
                avg_sla_compliance=0.0,
                avg_cycle_time_ms=0.0,
                function_class_usage={}
            )
        
        total_shipments = sum(r.shipments_dispatched for r in self.cycle_history)
        total_dispatches = sum(1 for r in self.cycle_history if r.decision.action_type == 'DISPATCH')
        avg_cycle_time = sum(r.execution_time_ms for r in self.cycle_history) / len(self.cycle_history)
        
        # Function class usage
        function_usage = {}
        for result in self.cycle_history:
            fc = result.decision.function_class.value
            function_usage[fc] = function_usage.get(fc, 0) + 1
        
        # Get utilization and SLA metrics from state manager
        analytics = self.state_manager.get_analytics()
        
        return PerformanceMetrics(
            total_cycles=len(self.cycle_history),
            total_shipments_processed=total_shipments,
            total_dispatches=total_dispatches,
            avg_utilization=analytics.get('avg_utilization', 0.0),
            avg_sla_compliance=analytics.get('sla_compliance_rate', 0.0),
            avg_cycle_time_ms=avg_cycle_time,
            function_class_usage=function_usage
        )
    
    def reset(self):
        """Reset engine state (for testing)"""
        self.current_cycle = 0
        self.cycle_history.clear()
        self.status = EngineStatus.IDLE
        logger.info("Engine reset")