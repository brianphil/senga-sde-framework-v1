# src/config/validation.py
from typing import List
from .senga_config import SengaConfigurator
class ConfigValidator:
    """Validates configuration consistency"""
    
    @staticmethod
    def validate_system_config(config: SengaConfigurator) -> List[str]:
        """
        Check for configuration issues
        Returns list of warnings/errors
        """
        issues = []
        
        # Check utilization threshold vs bonus
        if config.min_utilization >= 0.9:
            if config.get('utilization_bonus_per_percent') < 200:
                issues.append(
                    "WARNING: High utilization threshold (90%) but low bonus. "
                    "Consider increasing utilization_bonus_per_percent."
                )
        
        # Check emergency threshold vs SLA
        emergency_hours = config.emergency_threshold_hours
        sla_hours = config.sla_hours
        
        if emergency_hours > sla_hours * 0.5:
            issues.append(
                f"WARNING: Emergency threshold ({emergency_hours}h) is > 50% of SLA ({sla_hours}h). "
                "Emergency dispatches may trigger too late."
            )
        
        # Check fleet capacity vs typical shipment sizes
        fleet = config.fleet
        if not fleet:
            issues.append("ERROR: No active vehicles in fleet configuration!")
        
        # Check solver time limit vs decision cycle
        decision_cycle_min = config.get('decision_cycle_minutes', 60)
        solver_limit_sec = config.cfa_solver_time_limit
        
        if solver_limit_sec > decision_cycle_min * 60 * 0.8:
            issues.append(
                f"WARNING: Solver time limit ({solver_limit_sec}s) is > 80% of decision cycle "
                f"({decision_cycle_min}min). May cause delays."
            )
        
        return issues