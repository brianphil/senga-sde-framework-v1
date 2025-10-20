# src/integrations/external_systems.py

"""
Integration Layer: Connects Senga SDA to external systems
- Order Management System (OMS) API
- Google Places API for geocoding
- Driver Mobile App API
- Customer Platform API
- M-Pesa Payment Gateway
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import requests
from enum import Enum

logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"

@dataclass
class GeocodingResult:
    """Result from geocoding service"""
    place_id: str
    formatted_address: str
    latitude: float
    longitude: float
    confidence: float  # 0-1
    address_components: Dict

@dataclass
class OMSOrder:
    """Order from OMS"""
    order_id: str
    customer_id: str
    pickup_location: str
    delivery_location: str
    items: List[Dict]
    total_weight_kg: float
    total_volume_m3: float
    declared_value: float
    payment_method: str
    delivery_window_start: datetime
    delivery_window_end: datetime
    priority: str
    special_instructions: Optional[str]

@dataclass
class DriverUpdate:
    """Update from driver app"""
    driver_id: str
    vehicle_id: str
    timestamp: datetime
    location: Optional[Tuple[float, float]]  # (lat, lon)
    status: str  # 'available', 'in_transit', 'at_pickup', 'at_delivery'
    current_shipments: List[str]
    fuel_level: Optional[float]
    notes: Optional[str]

class GooglePlacesIntegration:
    """
    Integration with Google Places API for address geocoding
    
    Reality: Senga uses Google Places for accurate geocoding
    No informal address resolution needed - this is handled properly
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        self.status = IntegrationStatus.OFFLINE
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test Google Places API connectivity"""
        try:
            # Simple test query
            response = requests.get(
                f"{self.base_url}/textsearch/json",
                params={
                    'query': 'Nairobi',
                    'key': self.api_key
                },
                timeout=5
            )
            if response.status_code == 200:
                self.status = IntegrationStatus.ONLINE
                logger.info("Google Places API: Online")
            else:
                self.status = IntegrationStatus.DEGRADED
                logger.warning(f"Google Places API: Degraded (status {response.status_code})")
        except Exception as e:
            self.status = IntegrationStatus.OFFLINE
            logger.error(f"Google Places API: Offline - {e}")
    
    def geocode_address(self, address: str, bias_location: Optional[Tuple[float, float]] = None) -> GeocodingResult:
        """
        Geocode an address using Google Places
        
        Args:
            address: Address string to geocode
            bias_location: Optional (lat, lon) to bias results
            
        Returns:
            GeocodingResult with coordinates and confidence
        """
        try:
            params = {
                'query': address,
                'key': self.api_key
            }
            
            # Add location bias if provided (prefer Kenyan results)
            if bias_location:
                params['location'] = f"{bias_location[0]},{bias_location[1]}"
                params['radius'] = 50000  # 50km bias radius
            
            response = requests.get(
                f"{self.base_url}/textsearch/json",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
            
            data = response.json()
            
            if not data.get('results'):
                raise Exception("No results found")
            
            # Take first result (highest confidence)
            result = data['results'][0]
            
            # Calculate confidence based on result properties
            confidence = self._calculate_confidence(result)
            
            return GeocodingResult(
                place_id=result['place_id'],
                formatted_address=result['formatted_address'],
                latitude=result['geometry']['location']['lat'],
                longitude=result['geometry']['location']['lng'],
                confidence=confidence,
                address_components=result.get('address_components', {})
            )
            
        except Exception as e:
            logger.error(f"Geocoding failed for '{address}': {e}")
            # Return low-confidence result (fallback)
            return GeocodingResult(
                place_id="",
                formatted_address=address,
                latitude=0.0,
                longitude=0.0,
                confidence=0.0,
                address_components={}
            )
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate geocoding confidence from result properties"""
        confidence = 0.5  # Base confidence
        
        # Boost for specific location types
        if 'street_address' in result.get('types', []):
            confidence += 0.3
        elif 'premise' in result.get('types', []):
            confidence += 0.25
        elif 'route' in result.get('types', []):
            confidence += 0.15
        
        # Boost for complete address components
        components = result.get('address_components', [])
        if any(c for c in components if 'street_number' in c.get('types', [])):
            confidence += 0.1
        
        # Boost for geometry viewport (specific location)
        if 'viewport' in result.get('geometry', {}):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def batch_geocode(self, addresses: List[str]) -> List[GeocodingResult]:
        """Geocode multiple addresses (with rate limiting)"""
        results = []
        for address in addresses:
            result = self.geocode_address(address)
            results.append(result)
            # Simple rate limiting (Google allows 50 requests/sec)
            import time
            time.sleep(0.05)
        return results

class OMSIntegration:
    """
    Integration with Order Management System
    Hooks to existing OMS via API - does not rebuild order management
    """
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.status = IntegrationStatus.OFFLINE
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        self._test_connection()
    
    def _test_connection(self):
        """Test OMS API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.status = IntegrationStatus.ONLINE
                logger.info("OMS API: Online")
            else:
                self.status = IntegrationStatus.DEGRADED
                logger.warning(f"OMS API: Degraded")
        except Exception as e:
            self.status = IntegrationStatus.OFFLINE
            logger.error(f"OMS API: Offline - {e}")
    
    def fetch_pending_orders(self, since: Optional[datetime] = None) -> List[OMSOrder]:
        """
        Fetch pending orders from OMS
        
        Args:
            since: Only fetch orders created after this time
            
        Returns:
            List of pending orders
        """
        try:
            params = {'status': 'pending'}
            if since:
                params['created_after'] = since.isoformat()
            
            response = self.session.get(
                f"{self.base_url}/orders",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
            
            data = response.json()
            orders = []
            
            for order_data in data.get('orders', []):
                order = self._parse_order(order_data)
                if order:
                    orders.append(order)
            
            logger.info(f"Fetched {len(orders)} pending orders from OMS")
            return orders
            
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []
    
    def _parse_order(self, data: Dict) -> Optional[OMSOrder]:
        """Parse order data from OMS format"""
        try:
            return OMSOrder(
                order_id=data['order_id'],
                customer_id=data['customer_id'],
                pickup_location=data['pickup_address'],
                delivery_location=data['delivery_address'],
                items=data['items'],
                total_weight_kg=data['total_weight'],
                total_volume_m3=data['total_volume'],
                declared_value=data['declared_value'],
                payment_method=data['payment_method'],
                delivery_window_start=datetime.fromisoformat(data['delivery_window_start']),
                delivery_window_end=datetime.fromisoformat(data['delivery_window_end']),
                priority=data.get('priority', 'standard'),
                special_instructions=data.get('special_instructions')
            )
        except Exception as e:
            logger.error(f"Failed to parse order {data.get('order_id')}: {e}")
            return None
    
    def update_order_status(self, order_id: str, status: str, details: Optional[Dict] = None) -> bool:
        """
        Update order status in OMS
        
        Args:
            order_id: Order to update
            status: New status ('dispatched', 'in_transit', 'delivered', etc.)
            details: Additional details (vehicle_id, driver_id, ETA, etc.)
            
        Returns:
            Success boolean
        """
        try:
            payload = {
                'status': status,
                'updated_at': datetime.now().isoformat()
            }
            if details:
                payload.update(details)
            
            response = self.session.patch(
                f"{self.base_url}/orders/{order_id}",
                json=payload,
                timeout=10
            )
            
            return response.status_code in [200, 204]
            
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            return False

class DriverAppIntegration:
    """
    Integration with Driver Mobile App
    
    Responsibilities:
    - Send route manifests to drivers
    - Receive location and status updates
    - Handle offline sync when drivers reconnect
    """
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.status = IntegrationStatus.OFFLINE
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        self._test_connection()
    
    def _test_connection(self):
        """Test Driver App API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.status = IntegrationStatus.ONLINE
                logger.info("Driver App API: Online")
            else:
                self.status = IntegrationStatus.DEGRADED
        except Exception as e:
            self.status = IntegrationStatus.OFFLINE
            logger.error(f"Driver App API: Offline - {e}")
    
    def send_route_manifest(self, driver_id: str, route_data: Dict) -> bool:
        """
        Send route manifest to driver's app
        
        Args:
            driver_id: Driver to send to
            route_data: Complete route information (offline-compatible)
            
        Returns:
            Success boolean
        """
        try:
            # Prepare offline-compatible manifest
            manifest = {
                'route_id': route_data['route_id'],
                'driver_id': driver_id,
                'vehicle_id': route_data['vehicle_id'],
                'created_at': datetime.now().isoformat(),
                'stops': route_data['stops'],
                'shipments': route_data['shipments'],
                'total_distance_km': route_data['total_distance'],
                'estimated_duration_hours': route_data['estimated_duration'],
                'offline_mode': True  # Support offline operation
            }
            
            response = self.session.post(
                f"{self.base_url}/drivers/{driver_id}/route",
                json=manifest,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Route manifest sent to driver {driver_id}")
                return True
            else:
                logger.warning(f"Failed to send manifest: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send route manifest: {e}")
            return False
    
    def get_driver_updates(self, since: Optional[datetime] = None) -> List[DriverUpdate]:
        """
        Fetch driver status updates
        
        Args:
            since: Only fetch updates after this time
            
        Returns:
            List of driver updates
        """
        try:
            params = {}
            if since:
                params['since'] = since.isoformat()
            
            response = self.session.get(
                f"{self.base_url}/driver_updates",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            updates = []
            
            for update_data in data.get('updates', []):
                update = self._parse_driver_update(update_data)
                if update:
                    updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Failed to fetch driver updates: {e}")
            return []
    
    def _parse_driver_update(self, data: Dict) -> Optional[DriverUpdate]:
        """Parse driver update from API format"""
        try:
            location = None
            if data.get('latitude') and data.get('longitude'):
                location = (data['latitude'], data['longitude'])
            
            return DriverUpdate(
                driver_id=data['driver_id'],
                vehicle_id=data['vehicle_id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                location=location,
                status=data['status'],
                current_shipments=data.get('current_shipments', []),
                fuel_level=data.get('fuel_level'),
                notes=data.get('notes')
            )
        except Exception as e:
            logger.error(f"Failed to parse driver update: {e}")
            return None

class IntegrationManager:
    """
    Central manager for all external integrations
    Handles connection pooling, fallback strategies, offline operation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize all integrations
        
        Args:
            config: Integration configuration (API keys, URLs, etc.)
        """
        # Initialize integrations
        self.google_places = GooglePlacesIntegration(
            api_key=config.get('google_places_api_key', '')
        )
        
        self.oms = OMSIntegration(
            base_url=config.get('oms_base_url', ''),
            api_key=config.get('oms_api_key', '')
        )
        
        self.driver_app = DriverAppIntegration(
            base_url=config.get('driver_app_base_url', ''),
            api_key=config.get('driver_app_api_key', '')
        )
        
        logger.info("Integration Manager initialized")
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status of all integrations"""
        return {
            'google_places': self.google_places.status.value,
            'oms': self.oms.status.value,
            'driver_app': self.driver_app.status.value
        }
    
    def is_online(self) -> bool:
        """Check if critical integrations are online"""
        return (
            self.oms.status == IntegrationStatus.ONLINE and
            self.driver_app.status == IntegrationStatus.ONLINE
        )
    
    def refresh_connections(self):
        """Refresh all integration connections"""
        self.google_places._test_connection()
        self.oms._test_connection()
        self.driver_app._test_connection()
        logger.info("Integration connections refreshed")