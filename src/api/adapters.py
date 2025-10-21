# src/api/adapters.py

"""
Data Adapters: Translation layer between API formats and Core dataclasses
Handles conversion between external API dictionaries and internal Shipment/Location objects
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

from ..core.state_manager import (
    Shipment, Location, ShipmentStatus, VehicleState, 
    VehicleStatus, VehicleCapacity
)


class OrderAdapter:
    """
    Converts between API order format and core Shipment dataclass
    
    API Format (from demo/frontend):
    {
        "order_id": "ORD123",
        "customer_name": "John Mwangi",
        "customer_phone": "+254712345678",
        "pickup_location": {
            "address": "Nairobi CBD",
            "latitude": -1.2864,
            "longitude": 36.8172
        },
        "delivery_location": {
            "address": "Nakuru Town",
            "latitude": -0.3031,
            "longitude": 36.0800
        },
        "package_weight": 10.5,
        "priority": "standard",
        "created_at": "2025-10-21T10:00:00",
        "status": "pending"
    }
    
    Core Format (Shipment dataclass):
    Shipment(
        id="ORD123",
        customer_id="CUST_hash",
        origin=Location(...),
        destinations=[Location(...)],
        volume=0.5,
        weight=10.5,
        creation_time=datetime,
        deadline=datetime,
        status=ShipmentStatus.PENDING,
        priority=1.0
    )
    """
    
    @staticmethod
    def from_api_to_shipment(order_dict: Dict[str, Any]) -> Shipment:
        """
        Convert API order format to Shipment dataclass
        
        Args:
            order_dict: Order data from API/frontend
            
        Returns:
            Shipment object ready for StateManager
        """
        # Extract order ID or generate if missing
        order_id = order_dict.get('order_id')
        if not order_id:
            order_id = f"ORD{uuid.uuid4().hex[:8].upper()}"
        
        # Convert customer info
        # If customer_id exists use it, otherwise derive from customer_name
        customer_id = order_dict.get('customer_id')
        if not customer_id:
            customer_name = order_dict.get('customer_name', 'Unknown')
            # Create stable customer ID from name (for ops team tracking)
            customer_id = f"CUST_{abs(hash(customer_name)) % 10000:04d}"
        
        # Convert pickup location to origin
        pickup_loc = order_dict.get('pickup_location', {})
        origin = Location(
            place_id=pickup_loc.get('place_id', f"place_{uuid.uuid4().hex[:8]}"),
            lat=float(pickup_loc.get('latitude', 0.0)),
            lng=float(pickup_loc.get('longitude', 0.0)),
            formatted_address=pickup_loc.get('address', 'Unknown Location'),
            zone_id=pickup_loc.get('zone_id')
        )
        
        # Convert delivery location(s) to destinations
        # Support both single delivery_location and multiple delivery_locations
        destinations = []
        
        # Check for single delivery location
        if 'delivery_location' in order_dict:
            delivery_loc = order_dict['delivery_location']
            dest = Location(
                place_id=delivery_loc.get('place_id', f"place_{uuid.uuid4().hex[:8]}"),
                lat=float(delivery_loc.get('latitude', 0.0)),
                lng=float(delivery_loc.get('longitude', 0.0)),
                formatted_address=delivery_loc.get('address', 'Unknown Destination'),
                zone_id=delivery_loc.get('zone_id')
            )
            destinations.append(dest)
        
        # Check for multiple delivery locations
        elif 'delivery_locations' in order_dict:
            for delivery_loc in order_dict['delivery_locations']:
                dest = Location(
                    place_id=delivery_loc.get('place_id', f"place_{uuid.uuid4().hex[:8]}"),
                    lat=float(delivery_loc.get('latitude', 0.0)),
                    lng=float(delivery_loc.get('longitude', 0.0)),
                    formatted_address=delivery_loc.get('address', 'Unknown Destination'),
                    zone_id=delivery_loc.get('zone_id')
                )
                destinations.append(dest)
        
        # Default to origin if no destination specified (shouldn't happen)
        if not destinations:
            destinations = [origin]
        
        # Convert weight (kg)
        weight = float(order_dict.get('package_weight', 0.0))
        if weight <= 0:
            weight = 1.0  # Default minimum weight
        
        # Estimate volume from weight if not provided
        # Typical freight density: ~200 kg/m³
        volume = order_dict.get('volume_m3', order_dict.get('package_volume'))
        if volume is None:
            volume = weight / 200.0  # Estimate from weight
        else:
            volume = float(volume)
        
        # Parse creation time
        created_at = order_dict.get('created_at')
        if isinstance(created_at, str):
            try:
                creation_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                creation_time = datetime.now()
        elif isinstance(created_at, datetime):
            creation_time = created_at
        else:
            creation_time = datetime.now()
        
        # Calculate deadline based on priority
        priority_str = order_dict.get('priority', 'standard').lower()
        
        # Import config to get SLA hours
        from ..config.senga_config import SengaConfigurator
        config = SengaConfigurator()
        
        # Get SLA hours based on priority
        if priority_str == 'emergency':
            sla_hours = config.business.get('emergency_delivery_sla_hours', 2.0)
            priority_value = 3.0
        elif priority_str == 'urgent':
            sla_hours = config.business.get('urgent_delivery_sla_hours', 6.0)
            priority_value = 2.0
        else:  # standard
            sla_hours = config.business.get('standard_delivery_sla_hours', 24.0)
            priority_value = 1.0
        
        deadline = creation_time + timedelta(hours=sla_hours)
        
        # Convert status
        status_str = order_dict.get('status', 'pending').lower()
        status_map = {
            'pending': ShipmentStatus.PENDING,
            'batched': ShipmentStatus.BATCHED,
            'en_route': ShipmentStatus.EN_ROUTE,
            'in_transit': ShipmentStatus.EN_ROUTE,
            'delivered': ShipmentStatus.DELIVERED,
            'cancelled': ShipmentStatus.CANCELLED
        }
        status = status_map.get(status_str, ShipmentStatus.PENDING)
        
        # Create Shipment object
        return Shipment(
            id=order_id,
            customer_id=customer_id,
            origin=origin,
            destinations=destinations,
            volume=volume,
            weight=weight,
            creation_time=creation_time,
            deadline=deadline,
            status=status,
            priority=priority_value,
            batch_id=order_dict.get('batch_id'),
            route_id=order_dict.get('route_id')
        )
    
    @staticmethod
    def from_shipment_to_api(shipment: Shipment) -> Dict[str, Any]:
        """
        Convert Shipment dataclass to API-friendly dictionary
        
        Args:
            shipment: Core Shipment object
            
        Returns:
            Dictionary suitable for API responses and UI display
        """
        # Convert priority value back to string
        if shipment.priority >= 2.5:
            priority_str = 'emergency'
        elif shipment.priority >= 1.5:
            priority_str = 'urgent'
        else:
            priority_str = 'standard'
        
        # Use first destination as primary delivery location
        primary_destination = shipment.destinations[0] if shipment.destinations else shipment.origin
        
        return {
            'order_id': shipment.id,
            'customer_id': shipment.customer_id,
            'customer_name': f"Customer {shipment.customer_id.split('_')[-1]}",  # Display name
            'pickup_location': {
                'place_id': shipment.origin.place_id,
                'address': shipment.origin.formatted_address,
                'latitude': shipment.origin.lat,
                'longitude': shipment.origin.lng,
                'zone_id': shipment.origin.zone_id
            },
            'delivery_location': {
                'place_id': primary_destination.place_id,
                'address': primary_destination.formatted_address,
                'latitude': primary_destination.lat,
                'longitude': primary_destination.lng,
                'zone_id': primary_destination.zone_id
            },
            'delivery_locations': [
                {
                    'place_id': dest.place_id,
                    'address': dest.formatted_address,
                    'latitude': dest.lat,
                    'longitude': dest.lng,
                    'zone_id': dest.zone_id
                }
                for dest in shipment.destinations
            ],
            'package_weight': shipment.weight,
            'volume_m3': shipment.volume,
            'priority': priority_str,
            'created_at': shipment.creation_time.isoformat(),
            'deadline': shipment.deadline.isoformat(),
            'status': shipment.status.value,
            'batch_id': shipment.batch_id,
            'route_id': shipment.route_id,
            'time_to_deadline_hours': (shipment.deadline - datetime.now()).total_seconds() / 3600
        }
    
    @staticmethod
    def batch_from_shipments_to_api(shipments: List[Shipment]) -> List[Dict[str, Any]]:
        """Convert list of Shipments to API format"""
        return [OrderAdapter.from_shipment_to_api(s) for s in shipments]


class VehicleAdapter:
    """
    Converts between API vehicle format and core VehicleState dataclass
    """
    
    @staticmethod
    def from_api_to_vehicle_state(vehicle_dict: Dict[str, Any]) -> VehicleState:
        """Convert API vehicle format to VehicleState dataclass"""
        
        # Parse current location
        current_loc = vehicle_dict.get('current_location', {})
        current_location = Location(
            place_id=current_loc.get('place_id', ''),
            lat=float(current_loc.get('latitude', 0.0)),
            lng=float(current_loc.get('longitude', 0.0)),
            formatted_address=current_loc.get('address', 'Unknown'),
            zone_id=current_loc.get('zone_id')
        )
        
        # Parse home location if exists
        home_location = None
        if 'home_location' in vehicle_dict:
            home_loc = vehicle_dict['home_location']
            home_location = Location(
                place_id=home_loc.get('place_id', ''),
                lat=float(home_loc.get('latitude', 0.0)),
                lng=float(home_loc.get('longitude', 0.0)),
                formatted_address=home_loc.get('address', 'Unknown'),
                zone_id=home_loc.get('zone_id')
            )
        
        # Parse capacity
        capacity = VehicleCapacity(
            volume=float(vehicle_dict.get('capacity_volume_m3', 10.0)),
            weight=float(vehicle_dict.get('capacity_weight_kg', 1000.0))
        )
        
        # Parse status
        status_str = vehicle_dict.get('status', 'idle').lower()
        status_map = {
            'idle': VehicleStatus.IDLE,
            'available': VehicleStatus.IDLE,
            'en_route': VehicleStatus.EN_ROUTE,
            'in_transit': VehicleStatus.EN_ROUTE,
            'loading': VehicleStatus.LOADING,
            'unloading': VehicleStatus.UNLOADING,
            'offline': VehicleStatus.OFFLINE,
            'maintenance': VehicleStatus.MAINTENANCE
        }
        status = status_map.get(status_str, VehicleStatus.IDLE)
        
        # Parse availability time
        availability_time = None
        if 'availability_time' in vehicle_dict:
            avail_str = vehicle_dict['availability_time']
            if isinstance(avail_str, str):
                try:
                    availability_time = datetime.fromisoformat(avail_str.replace('Z', '+00:00'))
                except:
                    pass
            elif isinstance(avail_str, datetime):
                availability_time = avail_str
        
        return VehicleState(
            id=vehicle_dict.get('vehicle_id', vehicle_dict.get('id', f"VEH{uuid.uuid4().hex[:8]}")),
            vehicle_type=vehicle_dict.get('vehicle_type', 'truck'),
            capacity=capacity,
            current_location=current_location,
            status=status,
            cost_per_km=float(vehicle_dict.get('cost_per_km', 15.0)),
            fixed_cost_per_trip=float(vehicle_dict.get('fixed_cost_per_trip', 500.0)),
            current_route_id=vehicle_dict.get('current_route_id'),
            availability_time=availability_time,
            home_location=home_location
        )
    
    @staticmethod
    def from_vehicle_state_to_api(vehicle: VehicleState) -> Dict[str, Any]:
        """Convert VehicleState to API-friendly dictionary"""
        return {
            'vehicle_id': vehicle.id,
            'vehicle_type': vehicle.vehicle_type,
            'capacity_volume_m3': vehicle.capacity.volume,
            'capacity_weight_kg': vehicle.capacity.weight,
            'current_location': {
                'place_id': vehicle.current_location.place_id,
                'address': vehicle.current_location.formatted_address,
                'latitude': vehicle.current_location.lat,
                'longitude': vehicle.current_location.lng,
                'zone_id': vehicle.current_location.zone_id
            },
            'status': vehicle.status.value,
            'cost_per_km': vehicle.cost_per_km,
            'fixed_cost_per_trip': vehicle.fixed_cost_per_trip,
            'current_route_id': vehicle.current_route_id,
            'availability_time': vehicle.availability_time.isoformat() if vehicle.availability_time else None,
            'is_available': vehicle.is_available(datetime.now()),
            'home_location': {
                'place_id': vehicle.home_location.place_id,
                'address': vehicle.home_location.formatted_address,
                'latitude': vehicle.home_location.lat,
                'longitude': vehicle.home_location.lng,
                'zone_id': vehicle.home_location.zone_id
            } if vehicle.home_location else None
        }