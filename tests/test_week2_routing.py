# tests/test_week2_routing.py
"""
Week 2 Validation Tests: Route Optimizer + PFA Adapter

Tests verify:
1. Route optimizer with OR-Tools TSP
2. Greedy fallback when solver fails
3. PFA adapter converts PFAAction → StandardBatch
4. Route quality metrics
5. Integration with distance calculator
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.route_optimizer import (
    RouteSequenceOptimizer, Location, RouteMetrics, compare_routes
)
from src.utils.distance_calculator import DistanceTimeCalculator
from src.core.pfa_adapter import PFABatchAdapter, create_pfa_adapter
from src.core.standard_types import StandardBatch, ActionType


class TestRouteOptimizer:
    """Test RouteSequenceOptimizer functionality"""
    
    @pytest.fixture
    def distance_calc(self):
        """Create distance calculator"""
        return DistanceTimeCalculator(use_api=False)
    
    @pytest.fixture
    def optimizer(self, distance_calc):
        """Create route optimizer"""
        return RouteSequenceOptimizer(distance_calc, solver_time_limit_seconds=5)
    
    def test_single_destination_route(self, optimizer):
        """Test simple two-stop route"""
        origin = Location(-1.286389, 36.817223, "Nairobi CBD")
        dest = Location(-0.283333, 36.066667, "Nakuru", ["S1"])
        
        result = optimizer.optimize_route_sequence(origin, [dest])
        
        assert result.optimization_status == 'TRIVIAL'
        assert len(result.route_stops) == 2
        assert result.route_stops[0].location_address == "Nairobi CBD"
        assert result.route_stops[1].location_address == "Nakuru"
        assert result.total_distance_km > 0
        assert result.total_duration_hours > 0
        
        # Distance should be ~160 km (Nairobi to Nakuru)
        assert 150 < result.total_distance_km < 170
    
    def test_multi_destination_route(self, optimizer):
        """Test TSP with multiple destinations"""
        origin = Location(-1.286389, 36.817223, "Nairobi")
        destinations = [
            Location(-0.283333, 36.066667, "Nakuru", ["S1"]),
            Location(0.514277, 35.269779, "Eldoret", ["S2"]),
            Location(-0.091702, 34.767956, "Kisumu", ["S3"])
        ]
        
        result = optimizer.optimize_route_sequence(origin, destinations)
        
        assert result.optimization_status in ['OPTIMAL', 'FEASIBLE', 'GREEDY_FALLBACK']
        assert len(result.route_stops) == 4  # Origin + 3 destinations
        assert result.total_distance_km > 300  # At least 300 km for this route
        assert result.total_duration_hours > 5  # At least 5 hours
        
        # Verify all shipments included
        all_shipments = []
        for stop in result.route_stops:
            all_shipments.extend(stop.shipment_ids)
        assert set(all_shipments) == {'S1', 'S2', 'S3'}
    
    def test_empty_destinations(self, optimizer):
        """Test with no destinations"""
        origin = Location(-1.286389, 36.817223, "Nairobi")
        
        result = optimizer.optimize_route_sequence(origin, [])
        
        assert result.optimization_status == 'TRIVIAL'
        assert len(result.route_stops) == 1
        assert result.total_distance_km == 0
        assert result.total_duration_hours == 0
    
    def test_greedy_fallback(self, optimizer):
        """Test greedy algorithm fallback"""
        origin = Location(-1.286389, 36.817223, "Nairobi")
        destinations = [
            Location(-0.283333, 36.066667, "Nakuru", ["S1"]),
            Location(0.514277, 35.269779, "Eldoret", ["S2"])
        ]
        
        # Force greedy by using internal method
        result = optimizer._solve_with_greedy(origin, destinations)
        
        assert result.optimization_status == 'GREEDY_FALLBACK'
        assert len(result.route_stops) == 3
        assert result.total_distance_km > 0
    
    def test_route_quality_metrics(self, optimizer):
        """Test route quality calculation"""
        origin = Location(-1.286389, 36.817223, "Nairobi")
        dest = Location(-0.283333, 36.066667, "Nakuru", ["S1"])
        
        result = optimizer.optimize_route_sequence(origin, [dest])
        quality = optimizer.calculate_route_quality(result.route_stops)
        
        assert 'total_distance_km' in quality
        assert 'total_duration_hours' in quality
        assert 'avg_segment_distance_km' in quality
        assert 'num_stops' in quality
        
        assert quality['num_stops'] == 2
        assert quality['total_distance_km'] > 0
        assert quality['avg_segment_distance_km'] > 0


class TestPFAAdapter:
    """Test PFA → StandardBatch adapter"""
    
    @pytest.fixture
    def adapter(self):
        """Create PFA adapter"""
        return create_pfa_adapter({'use_maps_api': False})
    
    @pytest.fixture
    def mock_shipment(self):
        """Create mock shipment"""
        shipment = Mock()
        shipment.id = "S001"
        shipment.volume = 5.0
        shipment.weight = 500.0
        shipment.origin = Mock(lat=-1.286389, lng=36.817223, formatted_address="Nairobi")
        shipment.destination = Mock(lat=-0.283333, lng=36.066667, formatted_address="Nakuru")
        return shipment
    
    @pytest.fixture
    def mock_vehicle(self):
        """Create mock vehicle"""
        vehicle = Mock()
        vehicle.id = "V001"
        vehicle.capacity = Mock(volume=15.0, weight=2000.0)
        vehicle.cost_per_km = 30
        vehicle.fixed_cost = 1000
        vehicle.current_location = Mock(lat=-1.286389, lng=36.817223, 
                                       formatted_address="Nairobi Depot")
        return vehicle
    
    @pytest.fixture
    def mock_state(self):
        """Create mock system state"""
        state = Mock()
        state.timestamp = datetime.now()
        state.pending_shipments = []
        state.active_routes = []
        return state
    
    def test_convert_dispatch_action(self, adapter, mock_shipment, mock_vehicle, mock_state):
        """Test converting PFA DISPATCH action to StandardAction"""
        # Create mock PFAAction
        pfa_action = Mock()
        pfa_action.action_type = 'DISPATCH_IMMEDIATE'
        pfa_action.shipments = [mock_shipment]
        pfa_action.vehicle = mock_vehicle
        pfa_action.reasoning = "Test dispatch"
        pfa_action.confidence = 0.95
        
        # Convert
        standard_action = adapter.convert_to_standard_action(pfa_action, mock_state)
        
        assert standard_action.action_type == ActionType.DISPATCH_IMMEDIATE
        assert standard_action.function_class == 'pfa'
        assert len(standard_action.batches) == 1
        assert standard_action.confidence == 0.95
        assert standard_action.total_estimated_cost > 0
        
        # Check batch
        batch = standard_action.batches[0]
        assert batch.validate() == True
        assert batch.shipment_ids == ["S001"]
        assert batch.vehicle_id == "V001"
        assert len(batch.route_stops) >= 2
    
    def test_convert_wait_action(self, adapter, mock_state):
        """Test converting PFA WAIT action"""
        pfa_action = Mock()
        pfa_action.action_type = 'WAIT'
        pfa_action.shipments = []
        pfa_action.vehicle = None
        pfa_action.reasoning = "Waiting for consolidation"
        pfa_action.confidence = 0.7
        
        standard_action = adapter.convert_to_standard_action(pfa_action, mock_state)
        
        assert standard_action.action_type == ActionType.WAIT
        assert len(standard_action.batches) == 0
        assert standard_action.total_estimated_cost == 0
        assert standard_action.validate() == True
    
    def test_convert_emergency_no_vehicle(self, adapter, mock_shipment, mock_state):
        """Test converting EMERGENCY_NO_VEHICLE action"""
        pfa_action = Mock()
        pfa_action.action_type = 'EMERGENCY_NO_VEHICLE'
        pfa_action.shipments = [mock_shipment]
        pfa_action.vehicle = None
        pfa_action.reasoning = "No vehicles available"
        pfa_action.confidence = 1.0
        
        standard_action = adapter.convert_to_standard_action(pfa_action, mock_state)
        
        assert standard_action.action_type == ActionType.EMERGENCY_NO_VEHICLE
        assert len(standard_action.batches) == 0  # Can't dispatch
        assert standard_action.validate() == True
    
    def test_batch_utilization_calculation(self, adapter, mock_shipment, mock_vehicle, mock_state):
        """Test that utilization is calculated correctly"""
        pfa_action = Mock()
        pfa_action.action_type = 'DISPATCH_IMMEDIATE'
        pfa_action.shipments = [mock_shipment]  # 5 m3, 500 kg
        pfa_action.vehicle = mock_vehicle  # 15 m3, 2000 kg capacity
        pfa_action.reasoning = "Test"
        pfa_action.confidence = 0.9
        
        standard_action = adapter.convert_to_standard_action(pfa_action, mock_state)
        batch = standard_action.batches[0]
        
        # Check utilization
        expected_vol_util = 5.0 / 15.0  # ~0.33
        expected_weight_util = 500.0 / 2000.0  # 0.25
        
        assert 0.30 < batch.utilization_volume < 0.35
        assert 0.24 < batch.utilization_weight < 0.26
    
    def test_multiple_shipments_batch(self, adapter, mock_vehicle, mock_state):
        """Test batch with multiple shipments"""
        # Create two shipments
        ship1 = Mock()
        ship1.id = "S001"
        ship1.volume = 5.0
        ship1.weight = 500.0
        ship1.origin = Mock(lat=-1.286389, lng=36.817223, formatted_address="Nairobi")
        ship1.destination = Mock(lat=-0.283333, lng=36.066667, formatted_address="Nakuru")
        
        ship2 = Mock()
        ship2.id = "S002"
        ship2.volume = 4.0
        ship2.weight = 400.0
        ship2.origin = Mock(lat=-1.286389, lng=36.817223, formatted_address="Nairobi")
        ship2.destination = Mock(lat=0.514277, lng=35.269779, formatted_address="Eldoret")
        
        pfa_action = Mock()
        pfa_action.action_type = 'DISPATCH_IMMEDIATE'
        pfa_action.shipments = [ship1, ship2]
        pfa_action.vehicle = mock_vehicle
        pfa_action.reasoning = "Batch dispatch"
        pfa_action.confidence = 0.85
        
        standard_action = adapter.convert_to_standard_action(pfa_action, mock_state)
        batch = standard_action.batches[0]
        
        assert len(batch.shipment_ids) == 2
        assert set(batch.shipment_ids) == {"S001", "S002"}
        assert len(batch.route_stops) >= 3  # Origin + 2 destinations


class TestRouteComparison:
    """Test route comparison utilities"""
    
    def test_compare_routes(self):
        """Test comparing two routes"""
        route1 = RouteMetrics(
            total_distance_km=100,
            total_duration_hours=3,
            route_stops=[],
            optimization_status='OPTIMAL'
        )
        
        route2 = RouteMetrics(
            total_distance_km=120,
            total_duration_hours=3.5,
            route_stops=[],
            optimization_status='GREEDY_FALLBACK'
        )
        
        comparison = compare_routes(route1, route2)
        
        assert comparison['better_route'] == 1
        assert comparison['distance_improvement_pct'] == -20.0  # route2 is 20% worse
        assert comparison['route1_status'] == 'OPTIMAL'
        assert comparison['route2_status'] == 'GREEDY_FALLBACK'


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_pfa_to_batch(self):
        """Test complete flow: PFA → Adapter → StandardBatch"""
        # Setup
        adapter = create_pfa_adapter({'use_maps_api': False})
        
        # Mock PFAAction
        shipment = Mock()
        shipment.id = "S_TEST_001"
        shipment.volume = 8.0
        shipment.weight = 800.0
        shipment.origin = Mock(lat=-1.286389, lng=36.817223, formatted_address="Nairobi")
        shipment.destination = Mock(lat=-0.283333, lng=36.066667, formatted_address="Nakuru")
        
        vehicle = Mock()
        vehicle.id = "V_TEST_001"
        vehicle.capacity = Mock(volume=15.0, weight=2000.0)
        vehicle.cost_per_km = 30
        vehicle.fixed_cost = 1000
        vehicle.current_location = Mock(lat=-1.286389, lng=36.817223)
        
        pfa_action = Mock()
        pfa_action.action_type = 'DISPATCH_IMMEDIATE'
        pfa_action.shipments = [shipment]
        pfa_action.vehicle = vehicle
        pfa_action.reasoning = "Integration test"
        pfa_action.confidence = 0.9
        
        state = Mock()
        state.timestamp = datetime.now()
        
        # Execute
        standard_action = adapter.convert_to_standard_action(pfa_action, state)
        
        # Verify complete chain
        assert standard_action.validate() == True
        assert len(standard_action.batches) == 1
        
        batch = standard_action.batches[0]
        assert batch.validate() == True
        assert batch.estimated_distance_km > 0
        assert batch.estimated_duration_hours > 0
        assert batch.estimated_cost > 0
        
        # Convert to dict for executor
        batch_dict = batch.to_dict()
        assert 'id' in batch_dict
        assert 'shipments' in batch_dict
        assert 'vehicle' in batch_dict
        assert 'route' in batch_dict
        
        print("✓ End-to-end PFA → StandardBatch → dict works")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])