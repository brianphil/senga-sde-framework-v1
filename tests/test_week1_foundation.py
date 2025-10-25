# tests/test_week1_foundation.py
"""
Week 1 Validation Tests: Data Structures + Distance Calculator

Tests verify:
1. StandardBatch validation and conversion
2. StandardAction validation
3. Haversine distance accuracy
4. Nairobi traffic pattern correctness
5. Integration between components
"""

import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.standard_types import (
    StandardBatch, StandardAction, RouteStop, ActionType,
    create_standard_batch_from_pfa, create_wait_action
)
from src.utils.distance_calculator import DistanceTimeCalculator, classify_region


class TestStandardBatch:
    """Test StandardBatch creation and validation"""
    
    def test_valid_batch_creation(self):
        """Test creating a valid batch"""
        route_stops = [
            RouteStop(-1.286389, 36.817223, "Nairobi CBD", []),
            RouteStop(-0.283333, 36.066667, "Nakuru", ["S1", "S2"], 120, 15)
        ]
        
        batch = StandardBatch(
            batch_id="test_batch_001",
            shipment_ids=["S1", "S2"],
            vehicle_id="V1",
            route_stops=route_stops,
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=0.75,
            utilization_weight=0.80
        )
        
        assert batch.validate() == True
        assert len(batch.shipment_ids) == 2
        assert batch.estimated_distance_km > 0
    
    def test_invalid_batch_no_shipments(self):
        """Test that batch with no shipments is invalid"""
        route_stops = [
            RouteStop(-1.286389, 36.817223, "Nairobi", []),
            RouteStop(-0.283333, 36.066667, "Nakuru", [], 120, 15)
        ]
        
        batch = StandardBatch(
            batch_id="invalid_001",
            shipment_ids=[],  # No shipments
            vehicle_id="V1",
            route_stops=route_stops,
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=0.75,
            utilization_weight=0.80
        )
        
        assert batch.validate() == False
    
    def test_invalid_batch_no_route(self):
        """Test that batch with insufficient route stops is invalid"""
        batch = StandardBatch(
            batch_id="invalid_002",
            shipment_ids=["S1"],
            vehicle_id="V1",
            route_stops=[],  # No route
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=0.75,
            utilization_weight=0.80
        )
        
        assert batch.validate() == False
    
    def test_invalid_utilization(self):
        """Test that invalid utilization values are caught"""
        route_stops = [
            RouteStop(-1.286389, 36.817223, "Nairobi", []),
            RouteStop(-0.283333, 36.066667, "Nakuru", ["S1"], 120, 15)
        ]
        
        batch = StandardBatch(
            batch_id="invalid_003",
            shipment_ids=["S1"],
            vehicle_id="V1",
            route_stops=route_stops,
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=1.5,  # > 1.0 is invalid
            utilization_weight=0.80
        )
        
        assert batch.validate() == False
    
    def test_batch_to_dict_conversion(self):
        """Test converting batch to dict for backward compatibility"""
        route_stops = [
            RouteStop(-1.286389, 36.817223, "Nairobi", []),
            RouteStop(-0.283333, 36.066667, "Nakuru", ["S1"], 120, 15)
        ]
        
        batch = StandardBatch(
            batch_id="test_batch_002",
            shipment_ids=["S1"],
            vehicle_id="V1",
            route_stops=route_stops,
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=0.75,
            utilization_weight=0.80
        )
        
        batch_dict = batch.to_dict()
        
        assert batch_dict['id'] == "test_batch_002"
        assert batch_dict['shipments'] == ["S1"]
        assert batch_dict['vehicle'] == "V1"
        assert len(batch_dict['route']) == 2
        assert batch_dict['estimated_distance'] == 160
    
    def test_batch_from_dict_conversion(self):
        """Test creating batch from legacy dict format"""
        legacy_dict = {
            'id': 'legacy_batch_001',
            'shipments': ['S1', 'S2'],
            'vehicle': 'V1',
            'route': [
                {'lat': -1.286389, 'lng': 36.817223, 'formatted_address': 'Nairobi'},
                {'lat': -0.283333, 'lng': 36.066667, 'formatted_address': 'Nakuru'}
            ],
            'estimated_distance': 160,
            'estimated_duration': 4.0,
            'estimated_cost': 5000,
            'utilization': 0.75
        }
        
        batch = StandardBatch.from_dict(legacy_dict)
        
        assert batch.batch_id == 'legacy_batch_001'
        assert len(batch.shipment_ids) == 2
        assert batch.vehicle_id == 'V1'
        assert len(batch.route_stops) == 2


class TestStandardAction:
    """Test StandardAction creation and validation"""
    
    def test_valid_dispatch_action(self):
        """Test creating a valid dispatch action"""
        route_stops = [
            RouteStop(-1.286389, 36.817223, "Nairobi", []),
            RouteStop(-0.283333, 36.066667, "Nakuru", ["S1"], 120, 15)
        ]
        
        batch = StandardBatch(
            batch_id="batch_001",
            shipment_ids=["S1"],
            vehicle_id="V1",
            route_stops=route_stops,
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=0.75,
            utilization_weight=0.80
        )
        
        action = StandardAction(
            action_type=ActionType.DISPATCH,
            batches=[batch],
            function_class='pfa',
            reasoning="Simple dispatch for urgent shipment",
            confidence=0.95,
            total_estimated_cost=5000
        )
        
        assert action.validate() == True
        assert len(action.batches) == 1
    
    def test_valid_wait_action(self):
        """Test creating a valid wait action"""
        action = create_wait_action('cfa', 'Waiting for better consolidation', 0.7)
        
        assert action.validate() == True
        assert action.action_type == ActionType.WAIT
        assert len(action.batches) == 0
    
    def test_invalid_dispatch_no_batches(self):
        """Test that dispatch without batches is invalid"""
        action = StandardAction(
            action_type=ActionType.DISPATCH,
            batches=[],  # No batches
            function_class='cfa',
            reasoning="Invalid dispatch",
            confidence=0.5,
            total_estimated_cost=0
        )
        
        assert action.validate() == False
    
    def test_invalid_wait_with_batches(self):
        """Test that wait with batches is invalid"""
        route_stops = [
            RouteStop(-1.286389, 36.817223, "Nairobi", []),
            RouteStop(-0.283333, 36.066667, "Nakuru", ["S1"], 120, 15)
        ]
        
        batch = StandardBatch(
            batch_id="batch_002",
            shipment_ids=["S1"],
            vehicle_id="V1",
            route_stops=route_stops,
            estimated_distance_km=160,
            estimated_duration_hours=4.0,
            estimated_cost=5000,
            utilization_volume=0.75,
            utilization_weight=0.80
        )
        
        action = StandardAction(
            action_type=ActionType.WAIT,
            batches=[batch],  # Should be empty for WAIT
            function_class='pfa',
            reasoning="Invalid wait",
            confidence=0.5,
            total_estimated_cost=0
        )
        
        assert action.validate() == False


class TestDistanceCalculator:
    """Test DistanceTimeCalculator accuracy and performance"""
    
    def test_haversine_nairobi_mombasa(self):
        """Test distance calculation: Nairobi to Mombasa (~480 km)"""
        calc = DistanceTimeCalculator(use_api=False)
        
        nairobi = (-1.286389, 36.817223)
        mombasa = (-4.043477, 39.668206)
        
        distance = calc.calculate_distance_km(*nairobi, *mombasa)
        
        # Should be approximately 480 km (within 5%)
        assert 456 < distance < 504, f"Distance {distance} km not within 5% of 480 km"
    
    def test_haversine_nairobi_nakuru(self):
        """Test distance calculation: Nairobi to Nakuru (~160 km)"""
        calc = DistanceTimeCalculator(use_api=False)
        
        nairobi = (-1.286389, 36.817223)
        nakuru = (-0.283333, 36.066667)
        
        distance = calc.calculate_distance_km(*nairobi, *nakuru)
        
        # Should be approximately 160 km (within 5%)
        assert 152 < distance < 168, f"Distance {distance} km not within 5% of 160 km"
    
    def test_same_point_distance_zero(self):
        """Test that distance from point to itself is zero"""
        calc = DistanceTimeCalculator(use_api=False)
        
        nairobi = (-1.286389, 36.817223)
        distance = calc.calculate_distance_km(*nairobi, *nairobi)
        
        assert distance < 0.001, f"Same point distance should be ~0, got {distance}"
    
    def test_traffic_pattern_morning_rush(self):
        """Test Nairobi morning rush traffic (6-9am): 20 km/h"""
        calc = DistanceTimeCalculator(use_api=False)
        
        distance = 50  # km
        time_8am = calc.estimate_travel_time_hours(distance, time_of_day=8)
        speed_8am = distance / time_8am
        
        # Morning rush should be ~20 km/h
        assert 18 < speed_8am < 22, f"Morning rush speed {speed_8am} km/h not ~20 km/h"
    
    def test_traffic_pattern_midday(self):
        """Test Nairobi midday traffic (9am-4pm): 40 km/h"""
        calc = DistanceTimeCalculator(use_api=False)
        
        distance = 50  # km
        time_12pm = calc.estimate_travel_time_hours(distance, time_of_day=12)
        speed_12pm = distance / time_12pm
        
        # Midday should be ~40 km/h
        assert 38 < speed_12pm < 42, f"Midday speed {speed_12pm} km/h not ~40 km/h"
    
    def test_traffic_pattern_night(self):
        """Test Nairobi night traffic (7pm-6am): 50 km/h"""
        calc = DistanceTimeCalculator(use_api=False)
        
        distance = 50  # km
        time_2am = calc.estimate_travel_time_hours(distance, time_of_day=2)
        speed_2am = distance / time_2am
        
        # Night should be ~50 km/h
        assert 48 < speed_2am < 52, f"Night speed {speed_2am} km/h not ~50 km/h"
    
    def test_multi_stop_route(self):
        """Test calculating metrics for multi-stop route"""
        calc = DistanceTimeCalculator(use_api=False)
        
        stops = [
            (-1.286389, 36.817223),  # Nairobi
            (-0.283333, 36.066667),  # Nakuru
            (0.514277, 35.269779),   # Eldoret
        ]
        
        total_distance, total_time = calc.calculate_route_metrics(stops)
        
        assert total_distance > 0
        assert total_time > 0
        assert total_distance > 300  # At least 300 km for this route
    
    def test_cache_performance(self):
        """Test that caching improves performance"""
        calc = DistanceTimeCalculator(use_api=False)
        
        nairobi = (-1.286389, 36.817223)
        nakuru = (-0.283333, 36.066667)
        
        # First call - cache miss
        calc.calculate_distance_km(*nairobi, *nakuru)
        
        # Second call - cache hit
        calc.calculate_distance_km(*nairobi, *nakuru)
        
        stats = calc.get_cache_stats()
        assert stats['hits'] > 0, "Should have cache hits"


class TestHelperFunctions:
    """Test helper functions for PFA integration"""
    
    def test_create_standard_batch_from_pfa(self):
        """Test PFA helper for creating standard batches"""
        batch = create_standard_batch_from_pfa(
            shipment_ids=["S1"],
            vehicle_id="V1",
            origin_lat=-1.286389,
            origin_lon=36.817223,
            origin_addr="Nairobi CBD",
            dest_lat=-0.283333,
            dest_lon=36.066667,
            dest_addr="Nakuru",
            distance_km=160,
            duration_hours=4.0,
            cost=5000,
            util_volume=0.75,
            util_weight=0.80
        )
        
        assert batch.validate() == True
        assert len(batch.route_stops) == 2
        assert batch.shipment_ids == ["S1"]
    
    def test_region_classification_nairobi_cbd(self):
        """Test that Nairobi CBD is correctly classified"""
        region = classify_region(-1.286, 36.817)
        assert region == 'nairobi_cbd'
    
    def test_region_classification_nairobi_suburbs(self):
        """Test that Nairobi suburbs are correctly classified"""
        region = classify_region(-1.35, 36.75)
        assert region == 'nairobi_suburbs'
    
    def test_region_classification_highway(self):
        """Test that highway/far locations are classified"""
        region = classify_region(-4.0, 39.0)  # Near Mombasa
        assert region == 'highway'


class TestIntegration:
    """Integration tests combining data structures and calculator"""
    
    def test_complete_batch_with_real_distances(self):
        """Test creating batch with real distance calculations"""
        calc = DistanceTimeCalculator(use_api=False)
        
        # Calculate real metrics
        nairobi = (-1.286389, 36.817223)
        nakuru = (-0.283333, 36.066667)
        distance = calc.calculate_distance_km(*nairobi, *nakuru)
        duration = calc.estimate_travel_time_hours(distance, time_of_day=12)
        
        # Create batch using real calculations
        batch = create_standard_batch_from_pfa(
            shipment_ids=["S1"],
            vehicle_id="V1",
            origin_lat=nairobi[0],
            origin_lon=nairobi[1],
            origin_addr="Nairobi CBD",
            dest_lat=nakuru[0],
            dest_lon=nakuru[1],
            dest_addr="Nakuru",
            distance_km=distance,
            duration_hours=duration,
            cost=distance * 30,  # 30 KES per km
            util_volume=0.75,
            util_weight=0.80
        )
        
        assert batch.validate() == True
        assert 152 < batch.estimated_distance_km < 168
        assert batch.estimated_duration_hours > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])