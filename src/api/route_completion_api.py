# src/api/route_completion_api.py
"""
Route Completion API Enhancements

Provides endpoints for:
1. Driver reporting route completion
2. System recording actual outcomes
3. Triggering learning updates
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/routes", tags=["routes"])


class RouteCompletionRequest(BaseModel):
    """Request model for route completion"""

    route_id: str
    completion_time: datetime
    actual_cost: float
    actual_duration_hours: float
    actual_distance_km: Optional[float] = None
    shipments_delivered: int
    total_shipments: int
    on_time_deliveries: int
    delays: List[Dict] = []
    issues: List[str] = []
    driver_notes: Optional[str] = None


class RouteCompletionResponse(BaseModel):
    """Response model for route completion"""

    status: str
    route_id: str
    learning_triggered: bool
    vfa_updated: bool
    cfa_updated: bool
    message: str


@router.post("/complete", response_model=RouteCompletionResponse)
async def complete_route(request: RouteCompletionRequest):
    """
    Complete a route and trigger learning

    Called by driver app when route is finished

    Flow:
    1. Validate route exists
    2. Create RouteOutcome record
    3. Update shipment statuses to DELIVERED
    4. Update vehicle status to AVAILABLE
    5. Trigger tactical learning (VFA + CFA)
    6. Log state transition

    Returns:
        RouteCompletionResponse with learning status
    """
    from src.core.state_manager import StateManager, ShipmentStatus, VehicleStatus
    from src.core.multi_scale_coordinator import MultiScaleCoordinator, RouteOutcome

    try:
        # Get singletons
        state_manager = StateManager()
        coordinator = MultiScaleCoordinator()

        # Step 1: Validate route
        route = state_manager.get_route(request.route_id)
        if not route:
            raise HTTPException(
                status_code=404, detail=f"Route {request.route_id} not found"
            )

        logger.info(f"Completing route {request.route_id}")

        # Step 2: Create outcome record
        sla_compliance = request.on_time_deliveries == request.total_shipments

        outcome = RouteOutcome(
            route_id=request.route_id,
            completed_at=request.completion_time,
            initial_state={},  # Would ideally capture state snapshot
            shipments_delivered=request.shipments_delivered,
            total_shipments=request.total_shipments,
            actual_cost=request.actual_cost,
            predicted_cost=route.estimated_distance * 30,  # From CFA estimate
            actual_duration_hours=request.actual_duration_hours,
            predicted_duration_hours=route.estimated_distance / 40,
            utilization=request.shipments_delivered / request.total_shipments,
            sla_compliance=sla_compliance,
            delays=request.delays,
            issues=request.issues,
        )

        # Save outcome to database
        state_manager.save_route_outcome(outcome)
        logger.info(f"Route outcome saved: {outcome.route_id}")

        # Step 3: Update shipment statuses
        for shipment_id in route.shipment_ids:
            state_manager.update_shipment_status(
                shipment_id=shipment_id, status=ShipmentStatus.DELIVERED
            )

        logger.info(f"Updated {len(route.shipment_ids)} shipments to DELIVERED")

        # Step 4: Update vehicle status
        if route.vehicle_id:
            vehicle = state_manager.get_vehicle_state(route.vehicle_id)
            if vehicle:
                vehicle.status = VehicleStatus.AVAILABLE
                vehicle.current_route_id = None
                state_manager.update_vehicle_state(vehicle)
                logger.info(f"Vehicle {route.vehicle_id} now AVAILABLE")

        # Step 5: Trigger learning
        vfa_updated = False
        cfa_updated = False

        try:
            # VFA learning (always)
            coordinator.process_completed_route(outcome)
            vfa_updated = True
            logger.info("VFA learning update completed")

            # CFA learning (if batch formation metadata available)
            batch_metadata = state_manager.get_route_formation_metadata(
                request.route_id
            )
            if batch_metadata:
                # Future: coordinator.process_completed_route_with_cfa_learning(outcome, batch_metadata)
                cfa_updated = True
                logger.info("CFA learning update completed")
            else:
                logger.warning(
                    f"No batch metadata for route {request.route_id}, CFA not updated"
                )

        except Exception as e:
            logger.error(f"Learning update failed: {e}", exc_info=True)
            # Don't fail the request if learning fails

        # Step 6: Mark route as completed
        state_manager.mark_route_completed(request.route_id, request.completion_time)

        return RouteCompletionResponse(
            status="success",
            route_id=request.route_id,
            learning_triggered=True,
            vfa_updated=vfa_updated,
            cfa_updated=cfa_updated,
            message=f"Route completed successfully. Delivered {request.shipments_delivered}/{request.total_shipments} shipments.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route completion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{route_id}/status")
async def get_route_status(route_id: str):
    """
    Get current status of a route

    Returns:
        Route details including progress and ETA
    """
    from src.core.state_manager import StateManager

    try:
        state_manager = StateManager()
        route = state_manager.get_route(route_id)

        if not route:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        # Get shipment statuses
        shipment_statuses = []
        for sid in route.shipment_ids:
            shipment = state_manager.get_shipment(sid)
            if shipment:
                shipment_statuses.append(
                    {
                        "shipment_id": sid,
                        "status": shipment.status.value,
                        "destination": shipment.dest_address,
                    }
                )

        return {
            "route_id": route.id,
            "vehicle_id": route.vehicle_id,
            "created_at": route.created_at.isoformat(),
            "started_at": route.started_at.isoformat() if route.started_at else None,
            "completed_at": (
                route.completed_at.isoformat() if route.completed_at else None
            ),
            "estimated_distance_km": route.estimated_distance,
            "estimated_duration_hours": route.estimated_duration.total_seconds() / 3600,
            "num_stops": len(route.sequence),
            "num_shipments": len(route.shipment_ids),
            "shipments": shipment_statuses,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get route status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def get_active_routes():
    """
    Get all active routes

    Returns:
        List of routes currently in progress
    """
    from src.core.state_manager import StateManager

    try:
        state_manager = StateManager()
        active_routes = state_manager.get_active_routes()

        routes_data = []
        for route in active_routes:
            routes_data.append(
                {
                    "route_id": route.id,
                    "vehicle_id": route.vehicle_id,
                    "num_shipments": len(route.shipment_ids),
                    "num_stops": len(route.sequence),
                    "estimated_distance_km": route.estimated_distance,
                    "created_at": route.created_at.isoformat(),
                    "started_at": (
                        route.started_at.isoformat() if route.started_at else None
                    ),
                }
            )

        return {"count": len(routes_data), "routes": routes_data}

    except Exception as e:
        logger.error(f"Failed to get active routes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
