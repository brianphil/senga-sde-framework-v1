# src/core/state_manager.py

"""
State Manager: Central state tracking and event logging
Provides the mathematical state space S_t for all decision functions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import json
import uuid
from pathlib import Path

from ..config.senga_config import SengaConfigurator

# ============= State Space Definitions =============

class ShipmentStatus(Enum):
    PENDING = "pending"
    BATCHED = "batched"
    EN_ROUTE = "en_route"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class VehicleStatus(Enum):
    IDLE = "idle"
    EN_ROUTE = "en_route"
    LOADING = "loading"
    UNLOADING = "unloading"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DISPATCHED = "dispatched"

@dataclass
class Location:
    """Location with Google Places integration"""
    place_id: str
    lat: float
    lng: float
    formatted_address: str
    zone_id: Optional[str] = None

@dataclass
class Shipment:
    """Individual shipment in the system"""
    id: str
    customer_id: str
    origin: Location
    destinations: List[Location]  # Multi-destination support
    volume: float  # m³
    weight: float  # kg
    creation_time: datetime
    deadline: datetime
    status: ShipmentStatus
    priority: float = 1.0
    batch_id: Optional[str] = None
    route_id: Optional[str] = None
    
    # Additional fields needed for proper SDE functionality:
    declared_value: float = 0.0  # For cost calculations and prioritization
    
    # Legacy aliases for backwards compatibility (can remove later)
    @property
    def delivery_deadline(self):
        """Alias for deadline"""
        return self.deadline
    
    @property
    def weight_kg(self):
        """Alias for weight"""
        return self.weight
    
    @property
    def volume_m3(self):
        """Alias for volume"""
        return self.volume
    
    def time_to_deadline(self, current_time: datetime) -> timedelta:
        """Time remaining until deadline"""
        return self.deadline - current_time
    
    def time_pressure(self, current_time: datetime) -> float:
        """Normalized time pressure [0, 1], 1 = very urgent"""
        config = SengaConfigurator()
        sla_hours = config.sla_hours
        
        remaining = self.time_to_deadline(current_time)
        total_sla = timedelta(hours=sla_hours)
        
        if remaining <= timedelta(0):
            return 1.0
        
        return 1.0 - (remaining.total_seconds() / total_sla.total_seconds())
   
    def priority_name(self) -> str:
        """Convert priority value to readable name"""
        if self.priority >= 3.0:
            return "EMERGENCY"
        elif self.priority >= 2.0:
            return "URGENT"
        else:
            return "STANDARD"
    def to_dict(self) -> dict:
        """Serialize for storage"""
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'origin': {
                'place_id': self.origin.place_id,
                'lat': self.origin.lat,
                'lng': self.origin.lng,
                'formatted_address': self.origin.formatted_address,
                'zone_id': self.origin.zone_id
            },
            'destinations': [
                {
                    'place_id': d.place_id,
                    'lat': d.lat,
                    'lng': d.lng,
                    'formatted_address': d.formatted_address,
                    'zone_id': d.zone_id
                }
                for d in self.destinations
            ],
            'volume': self.volume,
            'weight': self.weight,
            'creation_time': self.creation_time.isoformat(),
            'deadline': self.deadline.isoformat(),
            'status': self.status.value,
            'priority': self.priority,
            'declared_value': self.declared_value,  # ← ADD THIS
            'batch_id': self.batch_id,
            'route_id': self.route_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Shipment':
        """Deserialize from storage"""
        return cls(
            id=data['id'],
            customer_id=data['customer_id'],
            origin=Location(**data['origin']),
            destinations=[Location(**d) for d in data['destinations']],
            volume=data['volume'],
            weight=data['weight'],
            creation_time=datetime.fromisoformat(data['creation_time']),
            deadline=datetime.fromisoformat(data['deadline']),
            status=ShipmentStatus(data['status']),
            priority=data['priority'],
            declared_value=data.get('declared_value', 0.0),  # ← ADD THIS (with default for old records)
            batch_id=data.get('batch_id'),
            route_id=data.get('route_id')
        )
@dataclass
class VehicleCapacity:
    volume: float  # m³
    weight: float  # kg

@dataclass
class VehicleState:
    """Current state of a vehicle"""
    id: str
    vehicle_type: str
    capacity: VehicleCapacity
    current_location: Location
    status: VehicleStatus
    cost_per_km: float
    fixed_cost_per_trip: float
    current_route_id: Optional[str] = None
    availability_time: Optional[datetime] = None
    home_location: Optional[Location] = None
    
    def is_available(self, current_time: datetime) -> bool:
        """Check if vehicle is available for dispatch"""
        if self.status not in [VehicleStatus.IDLE]:
            return False
        
        if self.availability_time and self.availability_time > current_time:
            return False
        
        return True
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'vehicle_type': self.vehicle_type,
            'capacity': {'volume': self.capacity.volume, 'weight': self.capacity.weight},
            'current_location': {
                'place_id': self.current_location.place_id,
                'lat': self.current_location.lat,
                'lng': self.current_location.lng,
                'formatted_address': self.current_location.formatted_address,
                'zone_id': self.current_location.zone_id
            },
            'status': self.status.value,
            'cost_per_km': self.cost_per_km,
            'fixed_cost_per_trip': self.fixed_cost_per_trip,
            'current_route_id': self.current_route_id,
            'availability_time': self.availability_time.isoformat() if self.availability_time else None,
            'home_location': {
                'place_id': self.home_location.place_id,
                'lat': self.home_location.lat,
                'lng': self.home_location.lng,
                'formatted_address': self.home_location.formatted_address,
                'zone_id': self.home_location.zone_id
            } if self.home_location else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VehicleState':
        return cls(
            id=data['id'],
            vehicle_type=data['vehicle_type'],
            capacity=VehicleCapacity(**data['capacity']),
            current_location=Location(**data['current_location']),
            status=VehicleStatus(data['status']),
            cost_per_km=data['cost_per_km'],
            fixed_cost_per_trip=data['fixed_cost_per_trip'],
            current_route_id=data.get('current_route_id'),
            availability_time=datetime.fromisoformat(data['availability_time']) if data.get('availability_time') else None,
            home_location=Location(**data['home_location']) if data.get('home_location') else None
        )

@dataclass
class Route:
    """Active route being executed"""
    id: str
    vehicle_id: str
    shipment_ids: List[str]
    sequence: List[Location]  # Ordered stops
    estimated_duration: timedelta
    estimated_distance: float  # km
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[timedelta] = None
    actual_distance: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'shipment_ids': self.shipment_ids,
            'sequence': [
                {
                    'place_id': loc.place_id,
                    'lat': loc.lat,
                    'lng': loc.lng,
                    'formatted_address': loc.formatted_address,
                    'zone_id': loc.zone_id
                }
                for loc in self.sequence
            ],
            'estimated_duration': self.estimated_duration.total_seconds(),
            'estimated_distance': self.estimated_distance,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'actual_duration': self.actual_duration.total_seconds() if self.actual_duration else None,
            'actual_distance': self.actual_distance
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Route':
        return cls(
            id=data['id'],
            vehicle_id=data['vehicle_id'],
            shipment_ids=data['shipment_ids'],
            sequence=[Location(**loc) for loc in data['sequence']],
            estimated_duration=timedelta(seconds=data['estimated_duration']),
            estimated_distance=data['estimated_distance'],
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            actual_duration=timedelta(seconds=data['actual_duration']) if data.get('actual_duration') else None,
            actual_distance=data.get('actual_distance')
        )

@dataclass
class SystemState:
    """
    Complete system state S_t
    This is the mathematical state space for Powell's framework
    """
    timestamp: datetime
    pending_shipments: List[Shipment]
    active_routes: List[Route]
    fleet_state: List[VehicleState]
    
    def get_available_vehicles(self) -> List[VehicleState]:
        """Get vehicles available for dispatch"""
        return [v for v in self.fleet_state if v.is_available(self.timestamp)]
    
    def get_urgent_shipments(self, threshold_hours: int) -> List[Shipment]:
        """Get shipments approaching deadline"""
        threshold = timedelta(hours=threshold_hours)
        return [
            s for s in self.pending_shipments
            if s.time_to_deadline(self.timestamp) <= threshold
        ]
    
    def total_pending_volume(self) -> float:
        """Total volume waiting for dispatch"""
        return sum(s.volume for s in self.pending_shipments)
    
    def total_pending_weight(self) -> float:
        """Total weight waiting for dispatch"""
        return sum(s.weight for s in self.pending_shipments)
    
    def fleet_utilization(self) -> float:
        """Current fleet utilization rate"""
        busy_vehicles = sum(1 for v in self.fleet_state if v.status not in [VehicleStatus.IDLE])
        return busy_vehicles / len(self.fleet_state) if self.fleet_state else 0.0
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'pending_shipments': [s.to_dict() for s in self.pending_shipments],
            'active_routes': [r.to_dict() for r in self.active_routes],
            'fleet_state': [v.to_dict() for v in self.fleet_state]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SystemState':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            pending_shipments=[Shipment.from_dict(s) for s in data['pending_shipments']],
            active_routes=[Route.from_dict(r) for r in data['active_routes']],
            fleet_state=[VehicleState.from_dict(v) for v in data['fleet_state']]
        )

@dataclass
class DecisionEvent:
    """Record of a decision made by the system."""
    id: str
    timestamp: datetime
    state_snapshot: 'SystemState'
    decision_type: str  # 'DISPATCH', 'WAIT', 'REOPTIMIZE', etc.
    function_class: str  # 'PFA', 'CFA', 'VFA', 'DLA', etc.
    action_details: Dict
    reasoning: str
    confidence: Optional[float] = None
    reward: Optional[float] = None
    vfa_value_before: Optional[float] = None
    vfa_value_after: Optional[float] = None
    td_error: Optional[float] = None
    alternatives_considered: List[Dict] = field(default_factory=list)
    
    @classmethod
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'state_snapshot': (
                self.state_snapshot.to_dict()
                if hasattr(self.state_snapshot, "to_dict")
                else str(self.state_snapshot)
            ),
            'decision_type': self.decision_type,
            'function_class': self.function_class,
            'action_details': self.action_details,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'reward': self.reward,
            'vfa_value_before': self.vfa_value_before,
            'vfa_value_after': self.vfa_value_after,
            'td_error': self.td_error,
            'alternatives_considered': self.alternatives_considered,
        }
    @classmethod
    def from_dict(cls, data: dict) -> 'DecisionEvent':
        """Deserialize from storage"""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            state_snapshot=SystemState.from_dict(data['state_snapshot']),
            decision_type=data['decision_type'],
            function_class=data['function_class'],
            action_details=data['action_details'],
            reasoning=data['reasoning'],
            confidence=data.get('confidence'),
            reward=data.get('reward'),
            vfa_value_before=data.get('vfa_value_before'),
            vfa_value_after=data.get('vfa_value_after'),
            td_error=data.get('td_error'),
            alternatives_considered=data.get('alternatives_considered', [])
        )

# ============= State Manager Implementation =============

class StateManager:
    """
    Central state management and event logging
    Provides offline-first storage with sync capability
    """
    
    def __init__(self, db_path: str = "data/senga_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.config = SengaConfigurator()
        self._init_tables()
        
        # In-memory cache for performance
        self._current_state: Optional[SystemState] = None
        self._state_cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=30)
    
    def _init_tables(self):
        """Initialize database schema"""
        
        # Shipments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS shipments (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                data TEXT NOT NULL,  -- JSON serialized shipment
                status TEXT NOT NULL,
                creation_time TIMESTAMP NOT NULL,
                deadline TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on status for quick filtering
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shipments_status 
            ON shipments(status)
        """)
        
        # Routes table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                id TEXT PRIMARY KEY,
                vehicle_id TEXT NOT NULL,
                data TEXT NOT NULL,  -- JSON serialized route
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Vehicle state table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_states (
                vehicle_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,  -- JSON serialized vehicle state
                status TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Decision log table (append-only for learning)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_log (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                decision_type TEXT NOT NULL,
                function_class TEXT NOT NULL,
                state_snapshot TEXT NOT NULL,  -- JSON
                action_details TEXT NOT NULL,  -- JSON
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # State transitions for TD learning
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT NOT NULL,
                state_before TEXT NOT NULL,  -- JSON
                action TEXT NOT NULL,  -- JSON
                reward REAL,
                state_after TEXT,  -- JSON (null until outcome observed)
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (decision_id) REFERENCES decision_log(id)
            )
        """)
        
        # Outcome tracking for learning
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS route_outcomes (
                route_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                estimated_cost REAL,
                actual_cost REAL,
                estimated_duration REAL,
                actual_duration REAL,
                utilization REAL,
                on_time_deliveries INTEGER,
                total_deliveries INTEGER,
                delay_penalties REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (route_id) REFERENCES routes(id),
                FOREIGN KEY (decision_id) REFERENCES decision_log(id)
            )
        """)
        
        self.conn.commit()
    
    # ============= Shipment Management =============
    
    def add_shipment(self, shipment: Shipment) -> bool:
        """Add new shipment to pending queue"""
        try:
            self.conn.execute("""
                INSERT INTO shipments (id, customer_id, data, status, creation_time, deadline)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                shipment.id,
                shipment.customer_id,
                json.dumps(shipment.to_dict()),
                shipment.status.value,
                shipment.creation_time,
                shipment.deadline
            ))
            self.conn.commit()
            self._invalidate_cache()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add shipment: {e}")
    
    def get_shipment(self, shipment_id: str) -> Optional[Shipment]:
        """Get shipment by ID"""
        cursor = self.conn.execute("""
            SELECT data FROM shipments WHERE id = ?
        """, (shipment_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return Shipment.from_dict(json.loads(row[0]))
    
    def update_shipment_status(self, shipment_id: str, status: ShipmentStatus,
                              batch_id: Optional[str] = None,
                              route_id: Optional[str] = None) -> bool:
        """Update shipment status"""
        shipment = self.get_shipment(shipment_id)
        if not shipment:
            return False
        
        shipment.status = status
        if batch_id:
            shipment.batch_id = batch_id
        if route_id:
            shipment.route_id = route_id
        
        try:
            self.conn.execute("""
                UPDATE shipments
                SET data = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(shipment.to_dict()), status.value, shipment_id))
            self.conn.commit()
            self._invalidate_cache()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update shipment: {e}")
    
    def get_pending_shipments(self) -> List[Shipment]:
        """Get all pending shipments"""
        cursor = self.conn.execute("""
            SELECT data FROM shipments WHERE status = ?
        """, (ShipmentStatus.PENDING.value,))
        
        return [Shipment.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
    
    # ============= Route Management =============
    
    def add_route(self, route: Route) -> bool:
        """Add new active route"""
        try:
            self.conn.execute("""
                INSERT INTO routes (id, vehicle_id, data, status)
                VALUES (?, ?, ?, ?)
            """, (
                route.id,
                route.vehicle_id,
                json.dumps(route.to_dict()),
                'active'
            ))
            self.conn.commit()
            self._invalidate_cache()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add route: {e}")
    
    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID"""
        cursor = self.conn.execute("""
            SELECT data FROM routes WHERE id = ?
        """, (route_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return Route.from_dict(json.loads(row[0]))
    
    def update_route(self, route: Route) -> bool:
        """Update route information"""
        try:
            self.conn.execute("""
                UPDATE routes
                SET data = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(route.to_dict()), route.id))
            self.conn.commit()
            self._invalidate_cache()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update route: {e}")
    
    def complete_route(self, route_id: str) -> bool:
        """Mark route as completed"""
        route = self.get_route(route_id)
        if not route:
            return False
        
        route.completed_at = datetime.now()
        
        try:
            self.conn.execute("""
                UPDATE routes
                SET data = ?, status = ?, completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (json.dumps(route.to_dict()), 'completed', route_id))
            self.conn.commit()
            self._invalidate_cache()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to complete route: {e}")
    
    def get_active_routes(self) -> List[Route]:
        """Get all active routes"""
        cursor = self.conn.execute("""
            SELECT data FROM routes WHERE status = 'active'
        """)
        
        return [Route.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
    
    # ============= Vehicle Management =============
    
    def sync_fleet_from_config(self):
        """Sync vehicle states from configuration"""
        fleet_config = self.config.fleet
        
        for vehicle_config in fleet_config:
            vehicle_id = vehicle_config['vehicle_id']
            
            # Check if vehicle already exists
            existing = self.get_vehicle_state(vehicle_id)
            
            if not existing:
                # Create new vehicle state
                vehicle_state = VehicleState(
                    id=vehicle_id,
                    vehicle_type=vehicle_config['vehicle_type'],
                    capacity=VehicleCapacity(
                        volume=vehicle_config['capacity']['volume'],
                        weight=vehicle_config['capacity']['weight']
                    ),
                    current_location=self._get_default_location(vehicle_config.get('home_location')),
                    status=VehicleStatus.IDLE,
                    cost_per_km=vehicle_config['cost_per_km'],
                    fixed_cost_per_trip=vehicle_config['fixed_cost'],
                    home_location=self._get_default_location(vehicle_config.get('home_location'))
                )
                
                self.update_vehicle_state(vehicle_state)
    
    def _get_default_location(self, place_id: Optional[str]) -> Location:
        """Get default location (Nairobi CBD if none specified)"""
        if place_id:
            # In production, resolve via Google Places API
            # For now, return placeholder
            return Location(
                place_id=place_id,
                lat=-1.2864,
                lng=36.8172,
                formatted_address="Nairobi, Kenya",
                zone_id="NAIROBI_CBD"
            )
        else:
            # Default to Nairobi CBD
            return Location(
                place_id="ChIJp0lN2HIRLxgRTJKXslQCz4o",
                lat=-1.2864,
                lng=36.8172,
                formatted_address="Nairobi CBD, Kenya",
                zone_id="NAIROBI_CBD"
            )
    
    def update_vehicle_state(self, vehicle_state: VehicleState) -> bool:
        """Update or insert vehicle state"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO vehicle_states (vehicle_id, data, status)
                VALUES (?, ?, ?)
            """, (
                vehicle_state.id,
                json.dumps(vehicle_state.to_dict()),
                vehicle_state.status.value
            ))
            self.conn.commit()
            self._invalidate_cache()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update vehicle state: {e}")
    
    def get_vehicle_state(self, vehicle_id: str) -> Optional[VehicleState]:
        """Get current vehicle state"""
        cursor = self.conn.execute("""
            SELECT data FROM vehicle_states WHERE vehicle_id = ?
        """, (vehicle_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return VehicleState.from_dict(json.loads(row[0]))
    
    def get_fleet_state(self) -> List[VehicleState]:
        """Get all vehicle states"""
        cursor = self.conn.execute("""
            SELECT data FROM vehicle_states
        """)
        
        return [VehicleState.from_dict(json.loads(row[0])) for row in cursor.fetchall()]
    
    # ============= System State =============
    
    def get_current_state(self, force_refresh: bool = False) -> SystemState:
        """
        Get current system state S_t
        Uses cache for performance unless force_refresh=True
        """
        now = datetime.now()
        
        # Check cache
        if not force_refresh and self._current_state and self._state_cache_time:
            if now - self._state_cache_time < self._cache_ttl:
                return self._current_state
        
        # Build fresh state
        state = SystemState(
            timestamp=now,
            pending_shipments=self.get_pending_shipments(),
            active_routes=self.get_active_routes(),
            fleet_state=self.get_fleet_state()
        )
        
        # Update cache
        self._current_state = state
        self._state_cache_time = now
        
        return state
    
    def _invalidate_cache(self):
        """Invalidate state cache after updates"""
        self._current_state = None
        self._state_cache_time = None
    
    # ============= Decision Logging =============
    
    def log_decision(self, decision: DecisionEvent) -> bool:
        """Log a decision for audit and learning"""
        try:
            self.conn.execute("""
                INSERT INTO decision_log (id, timestamp, decision_type, function_class, 
                                         state_snapshot, action_details, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.id,
                decision.timestamp,
                decision.decision_type,
                decision.function_class,
                json.dumps(decision.state_snapshot.to_dict()),
                json.dumps(decision.action_details),
                decision.reasoning
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to log decision: {e}")
    
    def log_state_transition(self, decision_id: str, state_before: SystemState,
                           action: dict) -> int:
        """
        Log state transition for TD learning
        Returns transition_id for later reward update
        """
        try:
            cursor = self.conn.execute("""
                INSERT INTO state_transitions (decision_id, state_before, action)
                VALUES (?, ?, ?)
            """, (
                decision_id,
                json.dumps(state_before.to_dict()),
                json.dumps(action)
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to log transition: {e}")
    
    def update_transition_outcome(self, transition_id: int, reward: float,
                                 state_after: SystemState) -> bool:
        """Update transition with observed reward and next state"""
        try:
            self.conn.execute("""
                UPDATE state_transitions
                SET reward = ?, state_after = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                reward,
                json.dumps(state_after.to_dict()),
                transition_id
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to update transition: {e}")
    
    def log_route_outcome(self, route_id: str, decision_id: str,
                         estimated_cost: float, actual_cost: float,
                         estimated_duration: float, actual_duration: float,
                         utilization: float, on_time_deliveries: int,
                         total_deliveries: int, delay_penalties: float) -> bool:
        """Log route outcome for performance analysis"""
        try:
            self.conn.execute("""
                INSERT INTO route_outcomes 
                (route_id, decision_id, estimated_cost, actual_cost, estimated_duration,
                 actual_duration, utilization, on_time_deliveries, total_deliveries, delay_penalties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                route_id, decision_id, estimated_cost, actual_cost, estimated_duration,
                actual_duration, utilization, on_time_deliveries, total_deliveries, delay_penalties
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to log route outcome: {e}")
    
    # ============= Query Methods for Learning =============
    
    def get_completed_routes(self, since: Optional[datetime] = None,
                            limit: Optional[int] = None) -> List[tuple]:
        """
        Get completed routes with outcomes for learning
        Returns: List of (route, outcome) tuples
        """
        query = """
            SELECT r.data, o.estimated_cost, o.actual_cost, o.estimated_duration,
                   o.actual_duration, o.utilization, o.on_time_deliveries,
                   o.total_deliveries, o.delay_penalties
            FROM routes r
            JOIN route_outcomes o ON r.id = o.route_id
            WHERE r.status = 'completed'
        """
        
        params = []
        if since:
            query += " AND r.completed_at >= ?"
            params.append(since)
        
        query += " ORDER BY r.completed_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            route = Route.from_dict(json.loads(row[0]))
            outcome = {
                'estimated_cost': row[1],
                'actual_cost': row[2],
                'estimated_duration': row[3],
                'actual_duration': row[4],
                'utilization': row[5],
                'on_time_deliveries': row[6],
                'total_deliveries': row[7],
                'delay_penalties': row[8]
            }
            results.append((route, outcome))
        
        return results
    
    def get_state_transitions(self, since: Optional[datetime] = None,
                             completed_only: bool = True,
                             limit: Optional[int] = None) -> List[dict]:
        """
        Get state transitions for TD learning
        Returns: List of {state_before, action, reward, state_after} dicts
        """
        query = """
            SELECT state_before, action, reward, state_after, completed_at
            FROM state_transitions
            WHERE 1=1
        """
        
        params = []
        
        if completed_only:
            query += " AND reward IS NOT NULL AND state_after IS NOT NULL"
        
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        transitions = []
        for row in cursor.fetchall():
            transitions.append({
                'state_before': SystemState.from_dict(json.loads(row[0])),
                'action': json.loads(row[1]),
                'reward': row[2],
                'state_after': SystemState.from_dict(json.loads(row[3])) if row[3] else None,
                'completed_at': datetime.fromisoformat(row[4]) if row[4] else None
            })
        
        return transitions
    
    def get_decision_history(self, limit: int = 100) -> List[DecisionEvent]:
        """Get recent decisions for analysis"""
        cursor = self.conn.execute("""
            SELECT id, timestamp, state_snapshot, decision_type, function_class,
                   action_details, reasoning
            FROM decision_log
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        decisions = []
        for row in cursor.fetchall():
            decision = DecisionEvent(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                state_snapshot=SystemState.from_dict(json.loads(row[2])),
                decision_type=row[3],
                function_class=row[4],
                action_details=json.loads(row[5]),
                reasoning=row[6]
            )
            decisions.append(decision)
        
        return decisions
    
    # ============= Analytics Queries =============
    
    def get_utilization_stats(self, days: int = 7) -> dict:
        """Get fleet utilization statistics"""
        since = datetime.now() - timedelta(days=days)
        
        cursor = self.conn.execute("""
            SELECT AVG(utilization), MIN(utilization), MAX(utilization)
            FROM route_outcomes
            WHERE created_at >= ?
        """, (since,))
        
        row = cursor.fetchone()
        return {
            'avg_utilization': row[0] if row[0] else 0.0,
            'min_utilization': row[1] if row[1] else 0.0,
            'max_utilization': row[2] if row[2] else 0.0
        }
    
    def get_on_time_performance(self, days: int = 7) -> float:
        """Get on-time delivery rate"""
        since = datetime.now() - timedelta(days=days)
        
        cursor = self.conn.execute("""
            SELECT SUM(on_time_deliveries), SUM(total_deliveries)
            FROM route_outcomes
            WHERE created_at >= ?
        """, (since,))
        
        row = cursor.fetchone()
        if row[1] and row[1] > 0:
            return row[0] / row[1]
        return 0.0
    
    def get_cost_efficiency(self, days: int = 7) -> dict:
        """Get cost efficiency metrics"""
        since = datetime.now() - timedelta(days=days)
        
        cursor = self.conn.execute("""
            SELECT AVG(actual_cost), AVG(estimated_cost), 
                   SUM(delay_penalties), COUNT(*)
            FROM route_outcomes
            WHERE created_at >= ?
        """, (since,))
        
        row = cursor.fetchone()
        return {
            'avg_actual_cost': row[0] if row[0] else 0.0,
            'avg_estimated_cost': row[1] if row[1] else 0.0,
            'total_delay_penalties': row[2] if row[2] else 0.0,
            'num_routes': row[3]
        }