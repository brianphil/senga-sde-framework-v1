# src/core/meta_controller.py

"""
Meta-Controller: Coordinates PFA, CFA, VFA, and DLA
This is the decision coordinator that Powell's framework needs
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .state_manager import StateManager, SystemState, DecisionEvent
from .pfa import PolicyFunctionApproximator, PFAAction
from .cfa import CostFunctionApproximator, CFASolution
from .vfa_neural import NeuralVFA as ValueFunctionApproximator
from .dla import DirectLookaheadApproximator, LookaheadResult
from ..config.senga_config import SengaConfigurator

class FunctionClass(Enum):
    PFA = "pfa"
    CFA = "cfa"
    DLA = "dla"

@dataclass
class ComplexityAssessment:
    """State complexity metrics"""
    num_pending: int
    num_available_vehicles: int
    time_pressure: float
    complexity_score: float
    is_simple: bool
    is_complex: bool

@dataclass
class StakesAssessment:
    """Decision stakes metrics"""
    estimated_cost: float
    potential_penalties: float
    stakes_score: float
    is_low_stakes: bool
    is_high_stakes: bool

@dataclass
class MetaDecision:
    """Final decision from meta-controller"""
    function_class: FunctionClass
    action_type: str
    action_details: dict
    reasoning: str
    confidence: float

class MetaController:
    """
    Coordinates all four function classes following Powell's framework:
    - PFA: Fast rules for simple/emergency cases
    - CFA: Optimization with VFA guidance for medium complexity
    - VFA: Value function for guidance (not standalone decision maker)
    - DLA: Lookahead for high-stakes/high-complexity decisions
    """
    
    def __init__(self):
        self.config = SengaConfigurator()
        self.state_manager = StateManager()
        
        # Initialize function approximators
        self.pfa = PolicyFunctionApproximator()
        self.cfa = CostFunctionApproximator()
        self.vfa = ValueFunctionApproximator()
        self.dla = DirectLookaheadApproximator()
        self._last_pfa_action = None
    def decide(self, state: SystemState) -> MetaDecision:
        """
        Main decision coordination logic
        
        Decision tree:
        1. Emergency? → PFA
        2. Simple state? → PFA
        3. High stakes/complexity? → DLA
        4. Default → CFA with VFA guidance
        """
        
        # Step 1: Check PFA (emergencies and simple cases)
        pfa_action = self.pfa.decide(state)
        self._last_pfa_action = pfa_action
        if pfa_action.action_type.startswith('EMERGENCY'):
            # Emergency: always use PFA
            return MetaDecision(
                function_class=FunctionClass.PFA,
                action_type=pfa_action.action_type,
                action_details={
                    'shipments': [s.id for s in pfa_action.shipments],
                    'vehicle': pfa_action.vehicle.id if pfa_action.vehicle else None
                },
                reasoning=f"PFA Emergency: {pfa_action.reasoning}",
                confidence=1.0
            )
        
        if pfa_action.action_type == 'DISPATCH_IMMEDIATE' and pfa_action.confidence > 0.9:
            # Simple dispatch with high confidence
            return MetaDecision(
                function_class=FunctionClass.PFA,
                action_type=pfa_action.action_type,
                action_details={
                    'shipments': [s.id for s in pfa_action.shipments],
                    'vehicle': pfa_action.vehicle.id
                },
                reasoning=f"PFA Simple Dispatch: {pfa_action.reasoning}",
                confidence=pfa_action.confidence
            )
        
        # Step 2: Assess complexity and stakes
        complexity = self._assess_complexity(state)
        stakes = self._assess_stakes(state)
        
        # Step 3: Route to appropriate function class
        if stakes.is_high_stakes or complexity.is_complex:
            # Use DLA for high-stakes or complex decisions
            return self._use_dla(state, complexity, stakes)
        else:
            # Use CFA with VFA guidance for medium complexity
            return self._use_cfa(state)
    
    def _assess_complexity(self, state: SystemState) -> ComplexityAssessment:
        """Assess state complexity"""
        
        num_pending = len(state.pending_shipments)
        num_available = len(state.get_available_vehicles())
        
        # Time pressure
        if state.pending_shipments:
            avg_time_pressure = sum(
                s.time_pressure(state.timestamp)
                for s in state.pending_shipments
            ) / len(state.pending_shipments)
        else:
            avg_time_pressure = 0.0
        
        # Complexity score
        complexity_score = (
            num_pending * 0.4 +
            num_available * 0.3 +
            avg_time_pressure * 10.0
        )
        
        # Thresholds from config
        low_threshold = self.config.get('dla.complexity_assessment.low_complexity_threshold.max_pending_shipments', 5)
        
        is_simple = num_pending <= low_threshold and num_available <= 2
        is_complex = num_pending > 10 or num_available > 5 or avg_time_pressure > 0.7
        
        return ComplexityAssessment(
            num_pending=num_pending,
            num_available_vehicles=num_available,
            time_pressure=avg_time_pressure,
            complexity_score=complexity_score,
            is_simple=is_simple,
            is_complex=is_complex
        )
    
    def _assess_stakes(self, state: SystemState) -> StakesAssessment:
        """Assess decision stakes"""
        
        # Estimate cost of potential dispatch
        estimated_cost = 0.0
        if state.get_available_vehicles():
            avg_vehicle_cost = sum(v.fixed_cost_per_trip for v in state.get_available_vehicles()) / len(state.get_available_vehicles())
            estimated_cost = avg_vehicle_cost * len(state.pending_shipments) / 5  # Rough estimate
        
        # Estimate potential delay penalties
        potential_penalties = 0.0
        penalty_per_hour = self.config.get('delay_penalty_per_hour', 500)
        
        for shipment in state.pending_shipments:
            if shipment.time_pressure(state.timestamp) > 0.7:
                # High risk of delay
                potential_penalties += penalty_per_hour * 2
        
        stakes_score = estimated_cost + potential_penalties
        
        # Thresholds
        low_stakes_threshold = self.config.get('dla.stakes_assessment.low_stakes_max_cost', 10000)
        high_stakes_threshold = self.config.get('dla.stakes_assessment.medium_stakes_max_cost', 50000)
        
        is_low_stakes = stakes_score < low_stakes_threshold
        is_high_stakes = stakes_score >= high_stakes_threshold
        
        return StakesAssessment(
            estimated_cost=estimated_cost,
            potential_penalties=potential_penalties,
            stakes_score=stakes_score,
            is_low_stakes=is_low_stakes,
            is_high_stakes=is_high_stakes
        )
    
    def _use_cfa(self, state: SystemState) -> MetaDecision:
        """Use CFA with VFA guidance"""
        
        solution = self.cfa.optimize(state, value_function=self.vfa)
        
        if solution.status in ['OPTIMAL', 'FEASIBLE'] and solution.batches:
            return MetaDecision(
                function_class=FunctionClass.CFA,
                action_type='DISPATCH',
                action_details={
                    'type': 'DISPATCH',  # CRITICAL: Add type key
                    'batches': [
                        {
                            'id': b.id,
                            'shipments': [s.id for s in b.shipments],
                            'vehicle': b.vehicle.id,
                            'utilization': b.utilization,
                            'estimated_cost': b.total_cost
                        }
                        for b in solution.batches
                    ],
                    'total_cost': solution.total_cost,
                    'avg_utilization': solution.avg_utilization
                },
                reasoning=solution.reasoning,
                confidence=0.8 if solution.status == 'OPTIMAL' else 0.6
            )
        else:
            # No feasible solution, wait
            return MetaDecision(
                function_class=FunctionClass.CFA,
                action_type='WAIT',
                action_details={'type': 'WAIT'},  # CRITICAL: Add type key
                reasoning=solution.reasoning,
                confidence=0.5
            )
    
    def _use_pfa(self, state: SystemState) -> MetaDecision:
        """Use PFA for simple/emergency cases"""
        
        action = self.pfa.select_action(state)
        
        if action.action_type in ['DISPATCH', 'DISPATCH_IMMEDIATE']:
            # Convert PFA's simple format to batch format expected by decision_engine
            batch = {
                'id': f"PFA_BATCH_{datetime.now().timestamp()}",
                'shipments': action.shipments if isinstance(action.shipments, list) else [action.shipments],
                'vehicle': action.vehicle,
                'sequence': [],
                'estimated_duration_hours': 3,
                'estimated_distance_km': 50.0
            }
            
            return MetaDecision(
                function_class=FunctionClass.PFA,
                action_type=action.action_type,
                action_details={
                    'type': action.action_type,
                    'batches': [batch]  # Wrap in batches array
                },
                reasoning=action.reasoning,
                confidence=1.0
            )
        else:
            return MetaDecision(
                function_class=FunctionClass.PFA,
                action_type='WAIT',
                action_details={'type': 'WAIT'},
                reasoning=action.reasoning,
                confidence=0.7
            )      
    def _use_dla(self, state: SystemState,
                complexity: ComplexityAssessment,
                stakes: StakesAssessment) -> MetaDecision:
        """Use DLA for high-stakes decisions"""
        
        # Generate candidate actions
        candidates = self._generate_candidate_actions(state)
        
        if not candidates:
            # No candidates, must wait
            return MetaDecision(
                function_class=FunctionClass.DLA,
                action_type='WAIT',
                action_details={},
                reasoning="DLA: No viable candidate actions",
                confidence=0.7
            )
        
        # Evaluate with lookahead
        result = self.dla.evaluate_with_lookahead(state, candidates)
        
        return MetaDecision(
            function_class=FunctionClass.DLA,
            action_type=result.action['type'],
            action_details=result.action,
            reasoning=result.reasoning,
            confidence=0.9  # High confidence from lookahead
        )
    
    def _generate_candidate_actions(self, state: SystemState) -> list:
        """Generate candidate actions for DLA evaluation"""
        
        candidates = []
        
        # Candidate 1: CFA solution
        cfa_solution = self.cfa.optimize(state, value_function=self.vfa)
        if cfa_solution.status in ['OPTIMAL', 'FEASIBLE'] and cfa_solution.batches:
            candidates.append({
                'type': 'DISPATCH',
                'batches': cfa_solution.batches
            })
        
        # Candidate 2: Wait
        candidates.append({
            'type': 'WAIT'
        })
        
        # Could add more candidates (partial dispatches, different batch formations, etc.)
        
        return candidates