# src/config/business_config.py

from typing import Any, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from enum import Enum

class ValueType(Enum):
    FLOAT = "float"
    INT = "int"
    MONEY = "money"
    DURATION = "duration"
    BOOLEAN = "boolean"

@dataclass
class ConfigParameter:
    key: str
    value: Any
    value_type: ValueType
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

class BusinessConfigurator:
    """
    Manages business parameters stored in database
    Provides UI-friendly interface for operations team
    """
    
    def __init__(self, db_path: str = "data/senga_config.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
        self._cache: Dict[str, ConfigParameter] = {}
        self._load_cache()
    
    def _init_tables(self):
        """Create config tables if they don't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS business_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_key TEXT UNIQUE NOT NULL,
                parameter_value TEXT NOT NULL,
                value_type TEXT NOT NULL,
                description TEXT,
                min_value REAL,
                max_value REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fleet_config (
                vehicle_id TEXT PRIMARY KEY,
                vehicle_type TEXT NOT NULL,
                capacity_volume_m3 REAL NOT NULL,
                capacity_weight_kg REAL NOT NULL,
                cost_per_km REAL NOT NULL,
                fixed_cost_per_trip REAL NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                home_location_place_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_config (
                customer_id TEXT PRIMARY KEY,
                priority_tier INTEGER DEFAULT 1,
                priority_multiplier REAL DEFAULT 1.0,
                custom_sla_hours INTEGER,
                preferred_delivery_windows TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS zone_config (
                zone_id TEXT PRIMARY KEY,
                zone_name TEXT NOT NULL,
                delivery_difficulty_factor REAL DEFAULT 1.0,
                traffic_multiplier_peak REAL DEFAULT 1.5,
                traffic_multiplier_offpeak REAL DEFAULT 1.0,
                peak_hours_start TIME DEFAULT '07:00',
                peak_hours_end TIME DEFAULT '10:00',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def _load_cache(self):
        """Load all business config into memory cache"""
        cursor = self.conn.execute("""
            SELECT parameter_key, parameter_value, value_type, description,
                   min_value, max_value, updated_at, updated_by
            FROM business_config
        """)
        
        for row in cursor.fetchall():
            key, value_str, type_str, desc, min_val, max_val, updated, updater = row
            value_type = ValueType(type_str)
            
            # Parse value based on type
            value = self._parse_value(value_str, value_type)
            
            self._cache[key] = ConfigParameter(
                key=key,
                value=value,
                value_type=value_type,
                description=desc,
                min_value=min_val,
                max_value=max_val,
                updated_at=updated,
                updated_by=updater
            )
    
    def _parse_value(self, value_str: str, value_type: ValueType) -> Any:
        """Convert string value to appropriate type"""
        if value_type == ValueType.FLOAT or value_type == ValueType.MONEY:
            return float(value_str)
        elif value_type == ValueType.INT or value_type == ValueType.DURATION:
            return int(value_str)
        elif value_type == ValueType.BOOLEAN:
            return value_str.lower() in ('true', '1', 'yes')
        return value_str
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get business parameter value
        Uses cache for performance, falls back to default
        """
        if key in self._cache:
            return self._cache[key].value
        return default
    
    def get_parameter(self, key: str) -> Optional[ConfigParameter]:
        """Get full parameter object with metadata"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, updated_by: str = "system") -> bool:
        """
        Update business parameter with validation
        Returns True if successful, False otherwise
        """
        # Get existing parameter for validation
        param = self._cache.get(key)
        if not param:
            raise ValueError(f"Unknown parameter: {key}")
        
        # Validate value
        if not self._validate_value(value, param):
            raise ValueError(
                f"Invalid value {value} for {key}. "
                f"Must be between {param.min_value} and {param.max_value}"
            )
        
        # Update database
        try:
            self.conn.execute("""
                UPDATE business_config
                SET parameter_value = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    updated_by = ?
                WHERE parameter_key = ?
            """, (str(value), updated_by, key))
            self.conn.commit()
            
            # Update cache
            param.value = value
            param.updated_at = datetime.now()
            param.updated_by = updated_by
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update {key}: {e}")
    
    def _validate_value(self, value: Any, param: ConfigParameter) -> bool:
        """Validate value against constraints"""
        if param.min_value is not None and value < param.min_value:
            return False
        if param.max_value is not None and value > param.max_value:
            return False
        return True
    
    def get_all_parameters(self) -> Dict[str, ConfigParameter]:
        """Get all business parameters for UI display"""
        return self._cache.copy()
    
    def reload(self):
        """Hot reload from database"""
        self._cache.clear()
        self._load_cache()
    
    # Fleet Configuration Methods
    
    def get_fleet(self) -> list:
        """Get all active vehicles"""
        cursor = self.conn.execute("""
            SELECT vehicle_id, vehicle_type, capacity_volume_m3, capacity_weight_kg,
                   cost_per_km, fixed_cost_per_trip, home_location_place_id
            FROM fleet_config
            WHERE is_active = TRUE
        """)
        
        return [
            {
                'vehicle_id': row[0],
                'vehicle_type': row[1],
                'capacity': {'volume': row[2], 'weight': row[3]},
                'cost_per_km': row[4],
                'fixed_cost': row[5],
                'home_location': row[6]
            }
            for row in cursor.fetchall()
        ]
    
    def add_vehicle(self, vehicle_id: str, vehicle_type: str,
                    capacity_volume: float, capacity_weight: float,
                    cost_per_km: float, fixed_cost: float,
                    home_location: Optional[str] = None) -> bool:
        """Add new vehicle to fleet"""
        try:
            self.conn.execute("""
                INSERT INTO fleet_config 
                (vehicle_id, vehicle_type, capacity_volume_m3, capacity_weight_kg,
                 cost_per_km, fixed_cost_per_trip, home_location_place_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (vehicle_id, vehicle_type, capacity_volume, capacity_weight,
                  cost_per_km, fixed_cost, home_location))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add vehicle: {e}")
    
    def update_vehicle(self, vehicle_id: str, **kwargs) -> bool:
        """Update vehicle parameters"""
        allowed_fields = {
            'vehicle_type', 'capacity_volume_m3', 'capacity_weight_kg',
            'cost_per_km', 'fixed_cost_per_trip', 'is_active', 'home_location_place_id'
        }
        
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return False
        
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [vehicle_id]
        
        try:
            self.conn.execute(f"""
                UPDATE fleet_config
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE vehicle_id = ?
            """, values)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update vehicle: {e}")
    
    # Customer Configuration Methods
    
    def get_customer_config(self, customer_id: str) -> Optional[Dict]:
        """Get customer-specific configuration"""
        cursor = self.conn.execute("""
            SELECT priority_tier, priority_multiplier, custom_sla_hours,
                   preferred_delivery_windows, notes
            FROM customer_config
            WHERE customer_id = ?
        """, (customer_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'priority_tier': row[0],
            'priority_multiplier': row[1],
            'custom_sla_hours': row[2],
            'delivery_windows': row[3],
            'notes': row[4]
        }
    
    def set_customer_priority(self, customer_id: str, tier: int, 
                             multiplier: float = None) -> bool:
        """Set customer priority tier"""
        if multiplier is None:
            multiplier = tier  # Default multiplier equals tier
        
        try:
            self.conn.execute("""
                INSERT INTO customer_config (customer_id, priority_tier, priority_multiplier)
                VALUES (?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    priority_tier = excluded.priority_tier,
                    priority_multiplier = excluded.priority_multiplier,
                    updated_at = CURRENT_TIMESTAMP
            """, (customer_id, tier, multiplier))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to set customer priority: {e}")
    
    # Zone Configuration Methods
    
    def get_zone_config(self, zone_id: str) -> Optional[Dict]:
        """Get zone-specific parameters"""
        cursor = self.conn.execute("""
            SELECT zone_name, delivery_difficulty_factor,
                   traffic_multiplier_peak, traffic_multiplier_offpeak,
                   peak_hours_start, peak_hours_end
            FROM zone_config
            WHERE zone_id = ?
        """, (zone_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'zone_name': row[0],
            'difficulty_factor': row[1],
            'traffic_peak': row[2],
            'traffic_offpeak': row[3],
            'peak_start': row[4],
            'peak_end': row[5]
        }