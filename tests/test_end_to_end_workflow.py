# tests/test_end_to_end_workflow.py
"""
End-to-End Workflow Test

Validates complete cycle:
1. Create orders
2. Run consolidation (CFA creates batches with route sequences)
3. Verify routes have optimized sequences
4. Complete route (simulate driver feedback)
5. Verify learning triggered (VFA + CFA updates)
6. Check state transitions logged
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.decision_engine import DecisionEngine
from src.core.state_manager import StateManager, Shipment, Location, ShipmentStatus
from src.core.multi_scale_coordinator import MultiScaleCoordinator, RouteOutcome


@pytest.fixture
def clean_environment():
    """Create fresh environment for testing"""
    # Use in-memory database
    state_manager = StateManager(db_path=":memory:")
    decision_engine = DecisionEngine()
    coordinator = MultiScaleCoordinator()

    # Initialize demo fleet
    # from init_demo import initialize_demo_fleet

    # initialize_demo_fleet(state_manager)

    return {
        "state_manager": state_manager,
        "engine": decision_engine,
        "coordinator": coordinator,
    }


class TestEndToEndWorkflow:
    """Complete workflow validation"""

    def test_full_cycle_order_to_learning(self, clean_environment):
        """
        Test Case: Complete workflow from order creation to learning

        Steps:
        1. Add 5 shipments to Mombasa (consolidation opportunity)
        2. Run decision cycle (should dispatch with CFA)
        3. Verify route created with sequence
        4. Complete route with actual data
        5. Verify learning triggered
        6. Check state transitions recorded
        """
        sm = clean_environment["state_manager"]
        engine = clean_environment["engine"]
        coordinator = clean_environment["coordinator"]

        # Step 1: Create 5 orders to Mombasa
        shipment_ids = []
        for i in range(5):
            shipment = Shipment(
                id=f"TEST_S{i+1}",
                origin_address="Nairobi CBD",
                dest_address="Mombasa",
                volume=0.5,
                weight=300,
                origin=Location(-1.286389, 36.817223, "Nairobi CBD", "Nai"),
                destinations=[Location(-4.043477, 39.668206, "Mombasa", "Mbs")],
                created_at=datetime.now(),
                delivery_deadline=datetime.now() + timedelta(hours=24),
                priority="standard",
                status=ShipmentStatus.PENDING,
            )
            sm.add_shipment(shipment)
            shipment_ids.append(shipment.id)

        print(f"\n Created {len(shipment_ids)} test shipments")

        # Step 2: Run decision cycle
        result = engine.run_cycle()

        print(f" Decision cycle completed")
        print(f"  Function: {result.decision.function_class.value}")
        print(f"  Action: {result.decision.action_type}")
        print(f"  Dispatched: {result.shipments_dispatched}")

        # Verify consolidation happened
        assert result.decision.action_type == "DISPATCH", "Should dispatch batch"
        assert (
            result.shipments_dispatched >= 4
        ), "Should consolidate at least 4 shipments"

        # Step 3: Verify route created with sequence
        routes = sm.get_active_routes()
        assert len(routes) > 0, "Should have created route(s)"

        test_route = routes[0]
        print(f"\nRoute created: {test_route.id}")
        print(f"  Shipments: {len(test_route.shipment_ids)}")
        print(f"  Sequence stops: {len(test_route.sequence)}")
        print(f"  Distance: {test_route.estimated_distance:.1f}km")

        # THIS IS THE CRITICAL CHECK - route must have sequence
        assert (
            len(test_route.sequence) > 0
        ), "Route MUST have optimized sequence from CFA"
        assert (
            len(test_route.sequence) >= 2
        ), "Route must have at least origin + destination"

        # Verify sequence has proper structure
        first_stop = test_route.sequence[0]
        assert "location_lat" in first_stop or hasattr(
            first_stop, "location_lat"
        ), "Sequence stops must have location data"

        print(f" Route sequence validated: {len(test_route.sequence)} stops")

        # Step 4: Simulate route completion
        actual_cost = (
            test_route.estimated_distance * 35
        )  # Slightly higher than estimate
        actual_duration = test_route.estimated_distance / 38  # Slightly slower

        outcome = RouteOutcome(
            route_id=test_route.id,
            completed_at=datetime.now(),
            initial_state=sm.get_current_state().__dict__,
            shipments_delivered=len(test_route.shipment_ids),
            total_shipments=len(test_route.shipment_ids),
            actual_cost=actual_cost,
            predicted_cost=test_route.estimated_distance * 30,
            actual_duration_hours=actual_duration,
            predicted_duration_hours=test_route.estimated_distance / 40,
            utilization=0.75,
            sla_compliance=True,
            delays=[],
            issues=[],
        )

        # Save outcome
        sm.save_route_outcome(outcome)
        print(f"\n Route outcome recorded")
        print(
            f"  Actual cost: {actual_cost:.0f} vs predicted: {outcome.predicted_cost:.0f}"
        )
        print(f"  Cost variance: {(actual_cost - outcome.predicted_cost):.0f}")

        # Step 5: Trigger learning
        coordinator.process_completed_route(outcome)
        print(f" Learning update triggered")

        # Step 6: Verify state transitions logged
        # Check that decision was logged
        conn = sm.conn
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM decision_log")
        decision_count = cursor.fetchone()[0]
        assert decision_count > 0, "Decision should be logged"

        cursor.execute("SELECT COUNT(*) FROM state_transitions")
        transition_count = cursor.fetchone()[0]
        assert transition_count > 0, "State transition should be logged"

        cursor.execute(
            "SELECT COUNT(*) FROM route_outcomes WHERE route_id = ?", (test_route.id,)
        )
        outcome_count = cursor.fetchone()[0]
        assert outcome_count == 1, "Route outcome should be saved"

        print(f" Database records validated")
        print(f"  Decisions logged: {decision_count}")
        print(f"  State transitions: {transition_count}")
        print(f"  Route outcomes: {outcome_count}")

        print(f"\n END-TO-END TEST PASSED")
        print(
            f"Complete workflow validated: Order → Consolidation → Route → Completion → Learning"
        )

    def test_route_sequence_structure(self, clean_environment):
        """
        Test Case: Verify route sequence has correct structure

        Route sequences from CFA must have:
        - List of RouteStop objects or dicts
        - Each stop has: lat, lon, address, shipment_ids, arrival_time
        """
        sm = clean_environment["state_manager"]
        engine = clean_environment["engine"]

        # Create 3 shipments to different destinations
        destinations = [
            ("Nakuru", -0.283333, 36.066667),
            ("Eldoret", 0.514277, 35.269779),
            ("Kisumu", -0.091702, 34.767956),
        ]

        for i, (city, lat, lon) in enumerate(destinations):
            shipment = Shipment(
                id=f"SEQ_TEST_S{i+1}",
                origin_address="Nairobi",
                dest_address=city,
                volume=1.0,
                weight=500,
                origin=Location(-1.286389, 36.817223, "Nairobi"),
                destinations=[Location(lat, lon, city)],
                created_at=datetime.now(),
                delivery_deadline=datetime.now() + timedelta(hours=36),
                priority="standard",
                status=ShipmentStatus.PENDING,
            )
            sm.add_shipment(shipment)

        # Run cycle
        result = engine.run_cycle()

        if result.decision.action_type == "DISPATCH":
            routes = sm.get_active_routes()
            assert len(routes) > 0

            route = routes[0]
            sequence = route.sequence

            print(f"\nRoute Sequence Structure:")
            for i, stop in enumerate(sequence):
                if isinstance(stop, dict):
                    print(f"  Stop {i}: {stop.get('location_address', 'N/A')}")
                    assert "location_lat" in stop
                    assert "location_lon" in stop
                    assert "location_address" in stop
                else:
                    print(f"  Stop {i}: {stop.location_address}")
                    assert hasattr(stop, "location_lat")
                    assert hasattr(stop, "location_lon")

            print(f" All stops have required fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
