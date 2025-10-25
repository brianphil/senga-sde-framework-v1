# src/utils/distance_calculator.py
"""
Real distance and time calculation with Nairobi-specific traffic patterns.

Mathematical Foundation:
- Distance: Haversine formula for great circle distance
- Time: Distance / speed, where speed varies by time of day based on Nairobi patterns
"""

import numpy as np
from typing import Tuple, Dict, Optional
from functools import lru_cache
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DistanceTimeCalculator:
    """
    Calculate real distances and travel times for Nairobi/Kenya operations.
    
    Features:
    - Haversine distance calculation (great circle)
    - Nairobi traffic pattern modeling
    - LRU cache for performance
    - Google Maps API integration (optional, for online mode)
    
    Traffic Patterns (Nairobi-specific):
    - Morning Rush (6-9am): 20 km/h
    - Midday (9am-4pm): 40 km/h
    - Evening Rush (4-7pm): 25 km/h
    - Night (7pm-6am): 50 km/h
    """
    
    # Nairobi traffic speed patterns (km/h) by hour of day
    NAIROBI_SPEEDS = {
        0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50,  # Night
        6: 20, 7: 20, 8: 20,                        # Morning rush
        9: 40, 10: 40, 11: 40, 12: 40, 13: 40, 14: 40, 15: 40,  # Midday
        16: 25, 17: 25, 18: 25,                     # Evening rush
        19: 40, 20: 40, 21: 40, 22: 50, 23: 50     # Evening/night
    }
    
    # Road quality adjustment factors for different regions
    ROAD_QUALITY = {
        'nairobi_cbd': 0.8,      # Heavy traffic, good roads
        'nairobi_suburbs': 0.9,  # Moderate traffic
        'highway': 1.2,          # Fast highways
        'rural': 0.7,            # Poor road conditions
        'default': 1.0
    }
    
    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize calculator
        
        Args:
            use_api: If True, use Google Maps API for online routing
            api_key: Google Maps API key (required if use_api=True)
        """
        self.use_api = use_api
        self.api_key = api_key
        self.cache_hits = 0
        self.cache_misses = 0
        
        if use_api and not api_key:
            logger.warning("API mode enabled but no API key provided. Falling back to Haversine.")
            self.use_api = False
    
    @lru_cache(maxsize=1000)
    def calculate_distance_km(self, lat1: float, lon1: float, 
                              lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance using Haversine formula.
        
        Mathematical Formula:
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2(√a, √(1−a))
        d = R ⋅ c
        
        where:
        - φ is latitude in radians
        - λ is longitude in radians  
        - R = 6371 km (Earth's radius)
        - Δφ = φ2 - φ1
        - Δλ = λ2 - λ1
        
        Args:
            lat1, lon1: Origin coordinates
            lat2, lon2: Destination coordinates
            
        Returns:
            Distance in kilometers
        """
        # Earth's radius in km
        R = 6371.0
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distance = R * c
        
        return distance
    
    def estimate_travel_time_hours(self, distance_km: float, 
                                   time_of_day: Optional[int] = None,
                                   region: str = 'default') -> float:
        """
        Estimate travel time based on distance and Nairobi traffic patterns.
        
        Formula:
        time = distance / (base_speed * road_quality_factor)
        
        Args:
            distance_km: Distance in kilometers
            time_of_day: Hour of day (0-23). If None, uses current time.
            region: Road quality region ('nairobi_cbd', 'highway', etc.)
            
        Returns:
            Estimated travel time in hours
        """
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Get base speed for time of day
        base_speed = self.NAIROBI_SPEEDS.get(time_of_day, 35)  # Default 35 km/h
        
        # Apply road quality adjustment
        quality_factor = self.ROAD_QUALITY.get(region, 1.0)
        effective_speed = base_speed * quality_factor
        
        # Calculate time
        time_hours = distance_km / effective_speed
        
        return time_hours
    
    def calculate_route_metrics(self, stops: list) -> Tuple[float, float]:
        """
        Calculate total distance and time for a multi-stop route.
        
        Args:
            stops: List of (lat, lon) tuples representing route stops
            
        Returns:
            Tuple of (total_distance_km, total_time_hours)
        """
        if len(stops) < 2:
            return 0.0, 0.0
        
        total_distance = 0.0
        total_time = 0.0
        current_hour = datetime.now().hour
        
        for i in range(len(stops) - 1):
            lat1, lon1 = stops[i]
            lat2, lon2 = stops[i + 1]
            
            # Calculate segment distance
            segment_distance = self.calculate_distance_km(lat1, lon1, lat2, lon2)
            total_distance += segment_distance
            
            # Calculate segment time (time progresses through day)
            segment_time = self.estimate_travel_time_hours(
                segment_distance, 
                time_of_day=current_hour
            )
            total_time += segment_time
            
            # Update hour for next segment (assuming continuous travel)
            current_hour = (current_hour + int(segment_time)) % 24
        
        return total_distance, total_time
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics"""
        cache_info = self.calculate_distance_km.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'size': cache_info.currsize,
            'max_size': cache_info.maxsize
        }
    
    def clear_cache(self):
        """Clear distance calculation cache"""
        self.calculate_distance_km.cache_clear()
        logger.info("Distance calculator cache cleared")


# Nairobi-specific helper functions

def classify_region(lat: float, lon: float) -> str:
    """
    Classify region for road quality estimation.
    
    Nairobi CBD: lat ≈ -1.286, lon ≈ 36.817
    Ranges are approximate for demonstration.
    """
    # Nairobi CBD bounds (approximate)
    if (-1.32 < lat < -1.26) and (36.79 < lon < 36.85):
        return 'nairobi_cbd'
    
    # Nairobi suburbs (wider area)
    if (-1.40 < lat < -1.20) and (36.70 < lon < 36.95):
        return 'nairobi_suburbs'
    
    # Check if on major highway (simplified - would need real road data)
    # Major highways: Thika Road, Mombasa Road, Nairobi-Nakuru
    if abs(lat - (-1.286)) > 0.5 or abs(lon - 36.817) > 0.5:
        return 'highway'
    
    return 'default'


def get_traffic_multiplier(hour: int) -> float:
    """
    Get traffic congestion multiplier for given hour.
    
    Returns value > 1 for slower traffic, < 1 for faster.
    """
    if hour in [7, 8, 17, 18]:  # Peak hours
        return 2.0  # Traffic doubles travel time
    elif hour in [6, 9, 16, 19]:  # Shoulder hours
        return 1.5
    elif 22 <= hour or hour <= 5:  # Night
        return 0.7  # Faster at night
    else:
        return 1.0  # Normal


# Testing helpers

def validate_calculator():
    """
    Validate calculator with known distances.
    
    Test cases:
    1. Nairobi to Mombasa: ~480 km
    2. Nairobi to Nakuru: ~160 km
    3. Nairobi to Eldoret: ~310 km
    """
    calc = DistanceTimeCalculator(use_api=False)
    
    test_cases = [
        # (lat1, lon1, lat2, lon2, expected_distance_km, name)
        (-1.286389, 36.817223, -4.043477, 39.668206, 480, "Nairobi-Mombasa"),
        (-1.286389, 36.817223, -0.283333, 36.066667, 160, "Nairobi-Nakuru"),
        (-1.286389, 36.817223, 0.514277, 35.269779, 310, "Nairobi-Eldoret"),
        (-1.286389, 36.817223, -1.286389, 36.817223, 0, "Same point"),
    ]
    
    print("\n=== Distance Calculator Validation ===")
    all_passed = True
    
    for lat1, lon1, lat2, lon2, expected, name in test_cases:
        calculated = calc.calculate_distance_km(lat1, lon1, lat2, lon2)
        error_pct = abs(calculated - expected) / max(expected, 1) * 100
        
        passed = error_pct < 5  # Within 5%
        status = "✓" if passed else "✗"
        
        print(f"{status} {name}: {calculated:.1f} km (expected {expected} km, error: {error_pct:.1f}%)")
        
        if not passed:
            all_passed = False
    
    # Test traffic patterns
    print("\n=== Traffic Pattern Validation ===")
    distance = 50  # km
    
    for hour, description in [(8, "Morning rush"), (12, "Midday"), (17, "Evening rush"), (2, "Night")]:
        time = calc.estimate_travel_time_hours(distance, hour)
        speed = distance / time
        print(f"{hour:02d}:00 ({description}): {time:.2f} hours = {speed:.1f} km/h")
    
    # Cache stats
    print("\n=== Cache Performance ===")
    stats = calc.get_cache_stats()
    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}, Size: {stats['size']}/{stats['max_size']}")
    
    return all_passed


if __name__ == "__main__":
    # Run validation
    validate_calculator()