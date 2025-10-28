# src/config/senga_config.py

from typing import Any, Optional, Dict
from .business_config import BusinessConfigurator
from .model_config import ModelConfigurator
from datetime import datetime
class SengaConfigurator:
    """
    Unified interface for all Senga configuration
    Transparently routes to business or model config
    """
    
    _instance = None  # Singleton
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 db_path: str = "data/senga_config.db",
                 model_config_path: str = "config/model_parameters.yaml",
                 env: str = "production"):
        
        # Only initialize once (singleton pattern)
        if hasattr(self, '_initialized'):
            return
        
        self.business = BusinessConfigurator(db_path)
        self.model = ModelConfigurator(model_config_path, env)
        self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Smart getter that routes to correct config source
        
        Usage:
            config.get('min_utilization_threshold')  → business config
            config.get('vfa.learning.learning_rate') → model config
        """
        # If key contains dots, it's a model config path
        if '.' in key:
            return self.model.get(key, default)
        
        # Otherwise, try business config first
        value = self.business.get(key, None)
        if value is not None:
            return value
        
        return default
    
    def set_business_param(self, key: str, value: Any, updated_by: str = "system") -> bool:
        """Update business parameter (UI/DB)"""
        return self.business.set(key, value, updated_by)
    
    def reload_all(self):
        """Reload both business and model configs"""
        self.business.reload()
        self.model.reload()
    
    # Convenience accessors for common operations
    
    @property
    def min_utilization(self) -> float:
        """Minimum utilization threshold (0.75)"""
        return self.business.get('min_utilization_threshold', 0.75)
    
    @property
    def sla_hours(self) -> int:
        """Standard SLA in hours (48)"""
        return self.business.get('sla_hours', 48)
    
    @property
    def cost_per_km(self) -> float:
        """Variable cost per kilometer"""
        return self.business.get('cost_per_km', 50.0)
    
    @property
    def emergency_threshold_hours(self) -> int:
        """Hours before deadline to trigger emergency dispatch"""
        return self.business.get('emergency_threshold_hours', 4)
    
    @property
    def vfa_learning_rate(self) -> float:
        """Current VFA learning rate"""
        return self.model.get('vfa.learning.initial_learning_rate', 0.01)
    
    @property
    def cfa_solver_time_limit(self) -> int:
        """CFA solver time limit in seconds"""
        return self.model.get('cfa.solver.time_limit_seconds', 30)
    
    @property
    def dla_lookahead_horizon(self) -> int:
        """DLA lookahead horizon in hours"""
        return self.model.get('dla.lookahead.horizon_hours', 3)
    
    @property
    def fleet(self) -> list:
        """Get active fleet configuration"""
        return self.business.get_fleet()
    @property
    def learning_config(self) -> Dict:
        """Learning configuration parameters"""
        return {
            'discount_factor': self.get('vfa.learning.discount_factor', 0.95),
            'learning_rate': self.get('vfa.learning.initial_learning_rate', 0.01),
            'lambda_trace': self.get('vfa.learning.lambda_trace', 0.7),
            'epsilon': self.get('vfa.exploration.initial_epsilon', 0.1)
        }
    def get_customer_priority(self, customer_id: str) -> float:
        """Get customer priority multiplier"""
        customer_config = self.business.get_customer_config(customer_id)
        if customer_config:
            return customer_config['priority_multiplier']
        return 1.0  # Default priority
    
    def get_zone_difficulty(self, zone_id: str) -> float:
        """Get zone delivery difficulty factor"""
        zone_config = self.business.get_zone_config(zone_id)
        if zone_config:
            return zone_config['difficulty_factor']
        return 1.0  # Default difficulty
    
    # Validation methods
    
    def validate_shipment_params(self, volume: float, weight: float) -> bool:
        """Validate shipment against fleet capacity"""
        fleet = self.fleet
        max_volume = max(v['capacity']['volume'] for v in fleet)
        max_weight = max(v['capacity']['weight'] for v in fleet)
        
        return volume <= max_volume and weight <= max_weight
    
    def get_computed_deadline(self, booking_time, customer_id: str = None) -> datetime:
        """Compute deadline based on SLA and customer priority"""
        from datetime import timedelta
        
        # Check for custom customer SLA
        if customer_id:
            customer_config = self.business.get_customer_config(customer_id)
            if customer_config and customer_config['custom_sla_hours']:
                sla_hours = customer_config['custom_sla_hours']
            else:
                sla_hours = self.sla_hours
        else:
            sla_hours = self.sla_hours
        
        return booking_time + timedelta(hours=sla_hours)