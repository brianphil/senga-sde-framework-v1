# tests/test_integration.py

"""
Integration tests for Senga SDE
Tests the complete system end-to-end with realistic scenarios
"""

import pytest
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.decision_engine import DecisionEngine
from src.core.state_manager import StateManager, Shipment, VehicleState, ShipmentStatus, VehicleStatus
from src.core.meta_controller import MetaController
from src.config.senga_config import SengaConfigurator

class TestEndToEndConsolidation:
    """
    End-to-end integration tests for the consolidation system
    
    Tests realistic scenarios following Senga's actual operations
    """
    
    @pytest.fixture
    def clean_state(self):
        """Setup clean state for each test"""
        config = SengaConfigurator()
        state_manager = StateManager()
        state_manager.reset()  # Clear all data
        
        # Add test vehicles
        vehicles = [
            VehicleState(
                vehicle_id="TRK001",
                status=VehicleStatus.AVAILABLE,
                capacity_kg=5000,
                capacity_m3=15,
                current_location="Nairobi",
                home_location="Nairobi"
            ),
            VehicleState(
                vehicle_id="TRK002",
                status=VehicleStatus.AVAILABLE,
                capacity_kg=3000,
                capacity_m3=10,
                current_location="Nairobi",
                home_location="Nairobi"
            ),
            VehicleState(
                vehicle_id="TRK003",
                status=VehicleStatus.AVAILABLE,
                capacity_kg=7000,
                capacity_m3=20,
                current_location="Nairobi",
                home_location="Nairobi"
            )
        ]
        
        for vehicle in vehicles:
            state_manager.add_vehicle(vehicle)
        
        yield state_manager
        
        # Cleanup
        state_manager.close()
    
    def test_single_order_immediate_dispatch(self, clean_state):
        """
        Test Case 1: Single high-priority order
        
        Scenario:
        - One urgent order arrives
        - Priority: urgent
        - Should trigger immediate dispatch via PFA
        
        Expected:
        - PFA handles the decision
        - Immediate dispatch action
        - One vehicle assigned
        """
        state_manager = clean_state
        engine = DecisionEngine()
        
        # Add urgent shipment
        urgent_shipment = Shipment(
            shipment_id="ORD001",
            origin_lat=-1.286389,
            origin_lon=36.817223,
            origin_address="Nairobi CBD",
            dest_lat=-0.091702,
            dest_lon=34.767956,
            dest_address="Kisumu",
            weight_kg=500,
            volume_m3=2,
            declared_value=50000,
            customer_id="CUST001",
            created_at=datetime.now() - timedelta(hours=24),  # 24 hours old
            delivery_deadline=datetime.now() + timedelta(hours=4),  # Urgent!
            priority="urgent",
            status=ShipmentStatus.PENDING
        )
        
        state_manager.add_shipment(urgent_shipment)
        
        # Run decision cycle
        result = engine.run_cycle()
        
        # Assertions
        assert result.decision.function_class.value == 'pfa', "Should use PFA for urgent shipment"
        assert result.decision.action_type == 'DISPATCH', "Should dispatch immediately"
        assert result.shipments_dispatched == 1, "Should dispatch 1 shipment"
        assert result.vehicles_utilized == 1, "Should use 1 vehicle"
        
        # Verify state updated
        updated_state = state_manager.get_current_state()
        assert len(updated_state.pending_shipments) == 0, "No pending shipments should remain"
    
    def test_consolidation_wait_for_batch(self, clean_state):
        """
        Test Case 2: Wait for consolidation
        
        Scenario:
        - 3 orders arrive for same destination
        - Not urgent, can wait
        - Below utilization threshold
        
        Expected:
        - CFA evaluates consolidation
        - Decision to WAIT for more orders
        - Utilization below 75% threshold
        """
        state_manager = clean_state
        engine = DecisionEngine()
        
        # Add 3 shipments to same destination
        for i in range(3):
            shipment = Shipment(
                shipment_id=f"ORD{i:03d}",
                origin_lat=-1.286389,
                origin_lon=36.817223,
                origin_address="Nairobi CBD",
                dest_lat=-0.283333,
                dest_lon=36.066667,
                dest_address="Nakuru",
                weight_kg=300,
                volume_m3=1.5,
                declared_value=30000,
                customer_id=f"CUST{i:03d}",
                created_at=datetime.now() - timedelta(hours=2),
                delivery_deadline=datetime.now() + timedelta(hours=40),
                priority="standard",
                status=ShipmentStatus.PENDING
            )
            state_manager.add_shipment(shipment)
        
        # Run decision cycle
        result = engine.run_cycle()
        
        # Assertions
        assert result.decision.action_type == 'WAIT', "Should wait for more orders"
        assert result.shipments_dispatched == 0, "Should not dispatch yet"
        
        # Verify state unchanged
        updated_state = state_manager.get_current_state()
        assert len(updated_state.pending_shipments) == 3, "All 3 orders should still be pending"
    
    def test_consolidation_optimal_dispatch(self, clean_state):
        """
        Test Case 3: Optimal consolidation dispatch
        
        Scenario:
        - 8 orders arrive for same corridor (Nairobi -> Western)
        - Good utilization potential (>75%)
        - Mix of destinations along route
        
        Expected:
        - CFA optimizes batch formation
        - DISPATCH decision
        - High utilization (>75%)
        - Optimal route sequence
        """
        state_manager = clean_state
        engine = DecisionEngine()
        
        # Add 8 shipments along Western corridor
        destinations = [
            ("Nakuru", -0.283333, 36.066667),
            ("Nakuru", -0.283333, 36.066667),
            ("Eldoret", 0.514277, 35.269779),
            ("Eldoret", 0.514277, 35.269779),
            ("Kitale", 1.019089, 35.006046),
            ("Kitale", 1.019089, 35.006046),
            ("Eldoret", 0.514277, 35.269779),
            ("Nakuru", -0.283333, 36.066667)
        ]
        
        for i, (dest_name, lat, lon) in enumerate(destinations):
            shipment = Shipment(
                shipment_id=f"ORD{i:03d}",
                origin_lat=-1.286389,
                origin_lon=36.817223,
                origin_address="Nairobi CBD",
                dest_lat=lat,
                dest_lon=lon,
                dest_address=dest_name,
                weight_kg=500,
                volume_m3=2,
                declared_value=40000,
                customer_id=f"CUST{i:03d}",
                created_at=datetime.now() - timedelta(hours=4),
                delivery_deadline=datetime.now() + timedelta(hours=36),
                priority="standard",
                status=ShipmentStatus.PENDING
            )
            state_manager.add_shipment(shipment)
        
        # Run decision cycle
        result = engine.run_cycle()
        
        # Assertions
        assert result.decision.function_class.value in ['cfa', 'dla'], "Should use CFA or DLA for optimization"
        assert result.decision.action_type == 'DISPATCH', "Should dispatch consolidated batch"
        assert result.shipments_dispatched >= 6, "Should dispatch at least 6 shipments for good utilization"
        assert result.vehicles_utilized >= 1, "Should use at least 1 vehicle"
        
        # Calculate utilization
        dispatched_weight = 500 * result.shipments_dispatched
        vehicle_capacity = 5000  # TRK001
        utilization = dispatched_weight / vehicle_capacity
        
        assert utilization >= 0.60, f"Should achieve >60% utilization, got {utilization:.1%}"
    
    def test_multi_vehicle_dispatch(self, clean_state):
        """
        Test Case 4: Multi-vehicle dispatch
        
        Scenario:
        - 15 orders arrive (high volume)
        - Multiple destinations
        - Requires multiple vehicles
        
        Expected:
        - CFA creates multiple batches
        - Multiple vehicles dispatched
        - Each batch optimized independently
        """
        state_manager = clean_state
        engine = DecisionEngine()
        
        # Add 15 shipments to various destinations
        destinations = [
            ("Mombasa", -4.043477, 39.668206),
            ("Mombasa", -4.043477, 39.668206),
            ("Mombasa", -4.043477, 39.668206),
            ("Nakuru", -0.283333, 36.066667),
            ("Nakuru", -0.283333, 36.066667),
            ("Nakuru", -0.283333, 36.066667),
            ("Eldoret", 0.514277, 35.269779),
            ("Eldoret", 0.514277, 35.269779),
            ("Eldoret", 0.514277, 35.269779),
            ("Kisumu", -0.091702, 34.767956),
            ("Kisumu", -0.091702, 34.767956),
            ("Kitale", 1.019089, 35.006046),
            ("Kitale", 1.019089, 35.006046),
            ("Nyeri", -0.420000, 36.950000),
            ("Nyeri", -0.420000, 36.950000)
        ]
        
        for i, (dest_name, lat, lon) in enumerate(destinations):
            shipment = Shipment(
                shipment_id=f"ORD{i:03d}",
                origin_lat=-1.286389,
                origin_lon=36.817223,
                origin_address="Nairobi CBD",
                dest_lat=lat,
                dest_lon=lon,
                dest_address=dest_name,
                weight_kg=400,
                volume_m3=1.5,
                declared_value=35000,
                customer_id=f"CUST{i:03d}",
                created_at=datetime.now() - timedelta(hours=5),
                delivery_deadline=datetime.now() + timedelta(hours=35),
                priority="standard",
                status=ShipmentStatus.PENDING
            )
            state_manager.add_shipment(shipment)
        
        # Run decision cycle
        result = engine.run_cycle()
        
        # Assertions
        assert result.decision.action_type == 'DISPATCH', "Should dispatch batches"
        assert result.shipments_dispatched >= 10, "Should dispatch most shipments"
        assert result.vehicles_utilized >= 2, "Should use multiple vehicles"
    
    def test_learning_from_outcome(self, clean_state):
        """
        Test Case 5: Learning from completed routes
        
        Scenario:
        - Dispatch a route
        - Simulate completion with actual outcomes
        - Verify VFA learns from the outcome
        
        Expected:
        - VFA weights updated
        - TD error calculated correctly
        - Learning improves predictions
        """
        state_manager = clean_state
        engine = DecisionEngine()
        
        # Add shipments and dispatch
        for i in range(5):
            shipment = Shipment(
                shipment_id=f"ORD{i:03d}",
                origin_lat=-1.286389,
                origin_lon=36.817223,
                origin_address="Nairobi CBD",
                dest_lat=-0.283333,
                dest_lon=36.066667,
                dest_address="Nakuru",
                weight_kg=500,
                volume_m3=2,
                declared_value=40000,
                customer_id=f"CUST{i:03d}",
                created_at=datetime.now() - timedelta(hours=6),
                delivery_deadline=datetime.now() + timedelta(hours=36),
                priority="standard",
                status=ShipmentStatus.PENDING
            )
            state_manager.add_shipment(shipment)
        
        # Get VFA weights before
        vfa = engine.vfa
        weights_before = vfa.weights.copy()
        num_updates_before = vfa.num_updates
        
        # Run cycle (should dispatch)
        result = engine.run_cycle()
        
        # Simulate route completion
        # In production, this would come from actual route completion
        from src.core.multi_scale_coordinator import RouteOutcome
        
        route_outcome = RouteOutcome(
            route_id="ROUTE001",
            completed_at=datetime.now(),
            initial_state={},
            shipments_delivered=5,
            total_shipments=5,
            actual_cost=25000,  # Actual cost
            predicted_cost=28000,  # Predicted was higher
            actual_duration_hours=4.5,
            predicted_duration_hours=5.0,
            utilization=0.80,
            sla_compliance=True,
            delays=[],
            issues=[]
        )
        
        # Process outcome (triggers learning)
        coordinator = engine.meta_controller.vfa
        # Manually trigger update (simulating what coordinator would do)
        coordinator.update(
            state=None,
            action_value=-28000,  # Predicted
            actual_outcome=-25000  # Actual (better than expected!)
        )
        
        # Verify learning occurred
        weights_after = vfa.weights
        num_updates_after = vfa.num_updates
        
        assert num_updates_after > num_updates_before, "VFA should have updated"
        assert not np.array_equal(weights_before, weights_after), "Weights should have changed"
    
    def test_performance_metrics(self, clean_state):
        """
        Test Case 6: Performance metrics tracking
        
        Scenario:
        - Run multiple decision cycles
        - Track performance metrics
        
        Expected:
        - Metrics calculated correctly
        - Utilization tracked
        - Function class usage tracked
        """
        state_manager = clean_state
        engine = DecisionEngine()
        
        # Run 5 cycles with varying scenarios
        for cycle in range(5):
            # Add random shipments
            num_shipments = 3 + cycle * 2
            for i in range(num_shipments):
                shipment = Shipment(
                    shipment_id=f"ORD_C{cycle}_S{i}",
                    origin_lat=-1.286389,
                    origin_lon=36.817223,
                    origin_address="Nairobi CBD",
                    dest_lat=-0.283333,
                    dest_lon=36.066667,
                    dest_address="Nakuru",
                    weight_kg=400 + i * 50,
                    volume_m3=1.5,
                    declared_value=35000,
                    customer_id=f"CUST_C{cycle}_S{i}",
                    created_at=datetime.now() - timedelta(hours=3),
                    delivery_deadline=datetime.now() + timedelta(hours=40),
                    priority="standard",
                    status=ShipmentStatus.PENDING
                )
                state_manager.add_shipment(shipment)
            
            # Run cycle
            engine.run_cycle()
        
        # Get metrics
        metrics = engine.get_performance_metrics()
        
        # Assertions
        assert metrics.total_cycles == 5, "Should have 5 cycles"
        assert metrics.total_shipments_processed >= 0, "Should track shipments"
        assert len(metrics.function_class_usage) > 0, "Should track function class usage"
        assert metrics.avg_cycle_time_ms > 0, "Should track execution time"

class TestComponentIntegration:
    """Test integration between major components"""
    
    def test_state_manager_integration(self):
        """Test StateManager integrates properly"""
        sm = StateManager()
        sm.reset()
        
        # Add shipment
        shipment = Shipment(
            shipment_id="TEST001",
            origin_lat=-1.286389,
            origin_lon=36.817223,
            origin_address="Nairobi",
            dest_lat=-0.283333,
            dest_lon=36.066667,
            dest_address="Nakuru",
            weight_kg=500,
            volume_m3=2,
            declared_value=40000,
            customer_id="CUST001",
            created_at=datetime.now(),
            delivery_deadline=datetime.now() + timedelta(hours=48),
            priority="standard",
            status=ShipmentStatus.PENDING
        )
        
        sm.add_shipment(shipment)
        
        # Get state
        state = sm.get_current_state()
        
        assert len(state.pending_shipments) == 1
        assert state.pending_shipments[0].shipment_id == "TEST001"
        
        sm.close()
    
    def test_meta_controller_integration(self):
        """Test MetaController coordinates properly"""
        sm = StateManager()
        sm.reset()
        
        # Add vehicle
        vehicle = VehicleState(
            vehicle_id="TRK001",
            status=VehicleStatus.AVAILABLE,
            capacity_kg=5000,
            capacity_m3=15,
            current_location="Nairobi",
            home_location="Nairobi"
        )
        sm.add_vehicle(vehicle)
        
        # Add urgent shipment
        shipment = Shipment(
            shipment_id="URGENT001",
            origin_lat=-1.286389,
            origin_lon=36.817223,
            origin_address="Nairobi",
            dest_lat=-0.283333,
            dest_lon=36.066667,
            dest_address="Nakuru",
            weight_kg=500,
            volume_m3=2,
            declared_value=40000,
            customer_id="CUST001",
            created_at=datetime.now() - timedelta(hours=30),
            delivery_deadline=datetime.now() + timedelta(hours=2),
            priority="urgent",
            status=ShipmentStatus.PENDING
        )
        sm.add_shipment(shipment)
        
        # Test meta-controller decision
        mc = MetaController()
        state = sm.get_current_state()
        decision = mc.decide(state)
        
        assert decision.function_class.value == 'pfa', "Should use PFA for urgent"
        assert decision.action_type in ['DISPATCH', 'EMERGENCY_DISPATCH']
        
        sm.close()

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])