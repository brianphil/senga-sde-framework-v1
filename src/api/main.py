# src/api/main.py

"""
Senga Sequential Decision Engine - Main API
FastAPI application with proper data persistence and adapter integration
"""

import logging
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Core components
from ..core.decision_engine import DecisionEngine, EngineStatus
from src.core.state_manager import (
    StateManager,
    Shipment,
    ShipmentStatus,
    VehicleState,
    VehicleStatus,
)
from ..config.senga_config import SengaConfigurator
from ..core.multi_scale_coordinator import MultiScaleCoordinator
from ..integrations.external_systems import IntegrationManager

# Import the new adapter
from .adapters import OrderAdapter, VehicleAdapter
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "senga_sde.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Senga Sequential Decision Engine",
    description="AI-powered freight consolidation and route optimization for African logistics",
    version="1.0.0",
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
decision_engine: Optional[DecisionEngine] = None
state_manager: Optional[StateManager] = None
config: Optional[SengaConfigurator] = None

# ============= Request/Response Models =============


class OrderCreateRequest(BaseModel):
    """Request model for creating orders - matches frontend format"""

    customer_name: str = Field(..., description="Customer name")
    customer_phone: Optional[str] = Field(None, description="Customer phone number")
    pickup_location: Dict[str, Any] = Field(
        ..., description="Pickup location with address, latitude, longitude"
    )
    delivery_location: Dict[str, Any] = Field(
        ..., description="Delivery location with address, latitude, longitude"
    )
    package_weight: float = Field(..., gt=0, description="Package weight in kg")
    volume_m3: Optional[float] = Field(
        None, description="Package volume in cubic meters"
    )
    priority: str = Field(
        "standard", description="Priority: standard, urgent, or emergency"
    )
    created_at: Optional[str] = Field(
        None, description="Creation timestamp (ISO format)"
    )
    customer_id: Optional[str] = Field(None, description="Existing customer ID")
    order_id: Optional[str] = Field(
        None, description="Specific order ID (auto-generated if not provided)"
    )


class OrderResponse(BaseModel):
    """Response model for order operations"""

    order_id: str
    customer_id: str
    customer_name: str
    pickup_location: Dict[str, Any]
    delivery_location: Dict[str, Any]
    package_weight: float
    volume_m3: float
    priority: str
    status: str
    created_at: str
    deadline: str
    time_to_deadline_hours: float


class ConsolidationCycleRequest(BaseModel):
    """Request to trigger consolidation cycle"""

    force_dispatch: bool = Field(
        False, description="Force dispatch even if utilization is low"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context (traffic, weather, etc.)"
    )


class ConsolidationCycleResponse(BaseModel):
    """Response from consolidation cycle"""

    timestamp: str
    total_pending_orders: int
    orders_dispatched: int
    orders_waiting: int
    batches_created: int
    function_class_used: str
    reasoning: str
    dispatched_batches: List[Dict[str, Any]]
    waiting_orders: List[Dict[str, Any]]


class SystemStatusResponse(BaseModel):
    """System status response"""

    status: str
    pending_orders: int
    available_vehicles: int
    active_routes: int
    timestamp: str


# ============= Startup and Shutdown =============


@app.on_event("startup")
async def startup_event():
    """Initialize all system components"""
    global decision_engine, state_manager, config

    logger.info("=" * 60)
    logger.info("Starting Senga Sequential Decision Engine")
    logger.info("=" * 60)

    try:
        # Initialize configuration
        config = SengaConfigurator()
        logger.info(" Configuration loaded")

        # Initialize state manager (with database persistence)
        state_manager = StateManager()
        logger.info(" State Manager initialized with persistent storage")

        # Initialize decision engine
        decision_engine = DecisionEngine()
        logger.info(" Decision Engine initialized")

        # Load existing pending orders count
        current_state = state_manager.get_current_state()
        logger.info(
            f" Loaded {len(current_state.pending_shipments)} pending orders from database"
        )

        logger.info("=" * 60)
        logger.info("Senga SDE Ready")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to start Senga SDE: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Senga Sequential Decision Engine")
    logger.info("Shutdown complete")


# ============= Health and Status Endpoints =============
@app.get("/shipments/pending")
async def get_pending_shipments():
    """
    Get all shipments with status 'pending' (awaiting dispatch)
    """
    try:
        # Get current system state
        current_state = state_manager.get_current_state()

        # Return pending shipments with calculated time_to_deadline
        pending = []
        for shipment in current_state.pending_shipments:
            shipment_dict = shipment.to_dict()

            # Calculate time to deadline in hours
            time_remaining = shipment.time_to_deadline(current_state.timestamp)
            shipment_dict["time_to_deadline_hours"] = (
                time_remaining.total_seconds() / 3600
            )

            pending.append(shipment_dict)

        return pending

    except Exception as e:
        logger.error(f"Failed to get pending shipments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINT 2: GET /fleet
# ============================================================================


@app.get("/fleet")
async def get_fleet():
    """
    Get all vehicles in the fleet with current status
    """
    try:
        # Get current system state
        current_state = state_manager.get_current_state()

        # Return fleet with enriched data
        fleet = []
        for vehicle in current_state.fleet_state:
            vehicle_dict = vehicle.to_dict()

            # Calculate current utilization if assigned to route
            if vehicle.status == VehicleStatus.EN_ROUTE and vehicle.assigned_route_id:
                # Find the route
                route = next(
                    (
                        r
                        for r in current_state.active_routes
                        if r.route_id == vehicle.assigned_route_id
                    ),
                    None,
                )
                if route:
                    # Calculate utilization based on route shipments
                    total_weight = sum(
                        s.weight
                        for s in current_state.pending_shipments
                        + [s for r in current_state.active_routes for s in r.shipments]
                        if s.id in [sid for sid in route.shipment_ids]
                    )
                    total_volume = sum(
                        s.volume
                        for s in current_state.pending_shipments
                        + [s for r in current_state.active_routes for s in r.shipments]
                        if s.id in [sid for sid in route.shipment_ids]
                    )

                    weight_util = (
                        (total_weight / vehicle.capacity_weight_kg) * 100
                        if vehicle.capacity_weight_kg > 0
                        else 0
                    )
                    volume_util = (
                        (total_volume / vehicle.capacity_volume_m3) * 100
                        if vehicle.capacity_volume_m3 > 0
                        else 0
                    )

                    vehicle_dict["current_utilization"] = max(weight_util, volume_util)
                else:
                    vehicle_dict["current_utilization"] = 0
            else:
                vehicle_dict["current_utilization"] = 0

            fleet.append(vehicle_dict)

        return fleet

    except Exception as e:
        logger.error(f"Failed to get fleet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OPTIONAL: GET /routes/active (if not already present)
# ============================================================================


@app.get("/routes/active")
async def get_active_routes():
    """
    Get all routes currently in progress
    """
    try:
        current_state = state_manager.get_current_state()
        return [route.to_dict() for route in current_state.active_routes]

    except Exception as e:
        logger.error(f"Failed to get active routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "state_manager": state_manager is not None,
            "decision_engine": decision_engine is not None,
            "config": config is not None,
        },
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status"""
    if not state_manager or not decision_engine:
        raise HTTPException(status_code=503, detail="System not fully initialized")

    current_state = state_manager.get_current_state()

    # Count available vehicles
    available_vehicles = sum(
        1 for v in current_state.fleet_state if v.is_available(datetime.now())
    )

    return SystemStatusResponse(
        status="operational",
        pending_orders=len(current_state.pending_shipments),
        available_vehicles=available_vehicles,
        active_routes=len(current_state.active_routes),
        timestamp=datetime.now().isoformat(),
    )


# ============= Autonomous Mode Control =============


@app.post("/autonomous/start")
async def start_autonomous_mode(
    background_tasks: BackgroundTasks, cycle_interval_minutes: int = 60
):
    """
    Start autonomous decision-making mode

    Args:
        cycle_interval_minutes: Time between decision cycles (default: 60)

    Returns:
        Status message
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if decision_engine.status == EngineStatus.AUTONOMOUS:
        return {
            "status": "already_running",
            "message": "Autonomous mode is already active",
        }

    # Start autonomous mode in background
    async def run_autonomous():
        await decision_engine.start_autonomous_mode(cycle_interval_minutes)

    background_tasks.add_task(run_autonomous)

    return {
        "status": "started",
        "cycle_interval_minutes": cycle_interval_minutes,
        "message": f"Autonomous mode started - decisions every {cycle_interval_minutes} minutes",
    }


@app.post("/autonomous/stop")
async def stop_autonomous_mode():
    """Stop autonomous decision-making"""
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    decision_engine.stop_autonomous_mode()

    return {"status": "stopped", "message": "Autonomous mode stopped"}


# ============= Decision Cycle Endpoints =============


@app.get("/cycles/recent")
async def get_recent_cycles(n: int = 20):
    """
    Get recent decision cycles

    Args:
        n: Number of recent cycles to return (default: 20, max: 100)

    Returns:
        List of recent decision cycles with details
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    n = min(n, 100)  # Cap at 100

    return decision_engine.get_recent_cycles(n)


@app.post("/trigger-cycle")
async def trigger_single_cycle():
    """
    Manually trigger a single decision cycle

    Returns:
        Cycle result details
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if decision_engine.status == EngineStatus.AUTONOMOUS:
        raise HTTPException(
            status_code=400,
            detail="Cannot trigger manual cycle while in autonomous mode",
        )

    try:
        cycle_result = decision_engine.run_cycle()

        return {
            "status": "success",
            "cycle_number": cycle_result.cycle_number,
            "function_class": cycle_result.decision.function_class.value,
            "action_type": cycle_result.decision.action_type,
            "reward": cycle_result.reward_components.total_reward,
            "td_error": cycle_result.td_error,
            "shipments_dispatched": cycle_result.shipments_dispatched,
            "vehicles_utilized": cycle_result.vehicles_utilized,
            "execution_time_ms": cycle_result.execution_time_ms,
        }

    except Exception as e:
        logger.error(f"Error in manual cycle trigger: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= Metrics Endpoints =============


@app.get("/metrics/performance")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics

    Returns:
        Performance statistics across all cycles
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        metrics = decision_engine.get_performance_metrics()

        return {
            "total_cycles": metrics.total_cycles,
            "total_shipments_processed": metrics.total_shipments_processed,
            "total_dispatches": metrics.total_dispatches,
            "avg_utilization": metrics.avg_utilization,
            "avg_reward_per_cycle": metrics.avg_reward_per_cycle,
            "avg_cycle_time_ms": metrics.avg_cycle_time_ms,
            "function_class_usage": metrics.function_class_usage,
            "learning_convergence": metrics.learning_convergence,
            "on_time_delivery_rate": 0.89,  # TODO: Calculate from actual data
            "cycles_last_hour": 1,  # TODO: Calculate from timestamps
            "reward_trend": 0.0,  # TODO: Calculate trend
            "sla_trend": 0.0,  # TODO: Calculate trend
            "utilization_trend": 0.0,  # TODO: Calculate trend
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learning/vfa-metrics")
async def get_vfa_metrics():
    """
    Get VFA learning metrics and statistics

    Returns:
        VFA learning progress, feature importance, convergence info
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        vfa_metrics = decision_engine.vfa.get_learning_metrics()

        # Calculate convergence score
        avg_td_error = vfa_metrics.get("avg_td_error", 100)
        convergence_score = 1.0 / (1.0 + avg_td_error / 100)

        return {
            "num_updates": vfa_metrics["num_updates"],
            "learning_rate": vfa_metrics["learning_rate"],
            "epsilon": vfa_metrics["epsilon"],
            "avg_td_error": vfa_metrics["avg_td_error"],
            "max_td_error": vfa_metrics["max_td_error"],
            "weight_norm": vfa_metrics["weight_norm"],
            "feature_importance": vfa_metrics["feature_importance"],
            "convergence_score": convergence_score,
            "initial_learning_rate": decision_engine.config.get(
                "vfa.learning.initial_learning_rate", 0.01
            ),
            "learning_rate_decay": decision_engine.config.get(
                "vfa.learning.learning_rate_decay", 0.9995
            ),
            "min_learning_rate": decision_engine.config.get(
                "vfa.learning.min_learning_rate", 0.0001
            ),
            "initial_epsilon": decision_engine.config.get(
                "vfa.exploration.initial_epsilon", 0.1
            ),
            "epsilon_decay": decision_engine.config.get(
                "vfa.exploration.epsilon_decay", 0.995
            ),
            "min_epsilon": decision_engine.config.get(
                "vfa.exploration.final_epsilon", 0.01
            ),
        }

    except Exception as e:
        logger.error(f"Error getting VFA metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Learning Feedback =======================
@app.post("/route/complete")
async def complete_route(
    route_id: str,
    actual_cost: float,
    actual_duration_hours: float,
    shipments_delivered: int,
    total_shipments: int,
    sla_compliant: bool,
    delays: List[Dict] = [],
    issues: List[str] = [],
):
    """
    Driver app reports route completion - triggers Week 5 tactical learning

    This is the CRITICAL Week 5 integration point
    """
    global multi_scale_coordinator

    if not state_manager or not multi_scale_coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        # Get the dispatched batch info
        # Note: You'll need to add a method to StateManager to retrieve this
        batch_info = state_manager.get_route_info(route_id)

        if not batch_info:
            raise HTTPException(status_code=404, detail=f"Route {route_id} not found")

        # Create RouteOutcome from actual data
        from src.core.multi_scale_coordinator import RouteOutcome

        outcome = RouteOutcome(
            route_id=route_id,
            completed_at=datetime.now(),
            initial_state=batch_info.get("state_snapshot", {}),
            shipments_delivered=shipments_delivered,
            total_shipments=total_shipments,
            actual_cost=actual_cost,
            predicted_cost=batch_info.get("estimated_cost", actual_cost),
            actual_duration_hours=actual_duration_hours,
            predicted_duration_hours=batch_info.get(
                "estimated_duration_hours", actual_duration_hours
            ),
            utilization=batch_info.get("utilization_volume", 0.0),
            sla_compliance=sla_compliant,
            delays=delays,
            issues=issues,
        )

        # Save outcome to database
        state_manager.save_route_outcome(outcome)

        # Trigger tactical learning (Week 5)
        batch_formation = state_manager.get_route_formation_metadata(route_id)

        if batch_formation:
            # NEW: Use enhanced method that also trains CFA
            multi_scale_coordinator.process_completed_route_with_cfa_learning(
                outcome, batch_formation
            )
        else:
            # Fallback to VFA-only learning
            logger.warning(
                f"No batch formation data for route {route_id}, CFA won't learn"
            )
            multi_scale_coordinator.process_completed_route(outcome)

        logger.info(f"Route {route_id} learning complete (VFA + CFA)")

        return {
            "status": "success",
            "route_id": route_id,
            "learning_triggered": True,
            "message": "Route outcome recorded and VFA updated",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route completion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= Analytics Endpoints =============


@app.get("/analytics/function-class-distribution")
async def get_function_class_distribution(last_n_cycles: int = 100):
    """
    Get distribution of function class usage

    Args:
        last_n_cycles: Number of recent cycles to analyze

    Returns:
        Breakdown of PFA/CFA/DLA usage
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    recent_cycles = decision_engine.get_recent_cycles(last_n_cycles)

    distribution = {"PFA": 0, "CFA": 0, "DLA": 0}

    for cycle in recent_cycles:
        fc = cycle["function_class"]
        distribution[fc] = distribution.get(fc, 0) + 1

    return distribution


@app.get("/analytics/reward-timeline")
async def get_reward_timeline(last_n_cycles: int = 100):
    """
    Get reward values over time

    Args:
        last_n_cycles: Number of recent cycles

    Returns:
        Timeline of rewards for plotting
    """
    global decision_engine

    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    recent_cycles = decision_engine.get_recent_cycles(last_n_cycles)

    return {
        "cycle_numbers": [c["cycle_number"] for c in recent_cycles],
        "rewards": [c["reward"] for c in recent_cycles],
        "vfa_values": [c["vfa_value"] for c in recent_cycles],
        "td_errors": [c["td_error"] for c in recent_cycles],
    }


# ============= Health Check Enhancement =============


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with all component status

    Returns:
        Status of engine, integrations, database, etc.
    """
    global decision_engine, integration_manager, state_manager

    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {},
    }

    # Check decision engine
    if decision_engine:
        health["components"]["decision_engine"] = {
            "status": "online",
            "mode": decision_engine.status.value,
            "cycles": decision_engine.current_cycle,
        }
    else:
        health["components"]["decision_engine"] = {
            "status": "offline",
            "error": "Not initialized",
        }

    # Check integrations
    if integration_manager:
        health["components"]["integrations"] = integration_manager.get_health_status()
    else:
        health["components"]["integrations"] = {
            "status": "offline",
            "error": "Not initialized",
        }

    # Check state manager
    if state_manager:
        try:
            current_state = state_manager.get_current_state()
            health["components"]["state_manager"] = {
                "status": "online",
                "pending_shipments": len(current_state.pending_shipments),
                "active_routes": len(current_state.active_routes),
            }
        except Exception as e:
            health["components"]["state_manager"] = {"status": "error", "error": str(e)}
    else:
        health["components"]["state_manager"] = {
            "status": "offline",
            "error": "Not initialized",
        }

    # Overall status
    if any(c.get("status") == "offline" for c in health["components"].values()):
        health["status"] = "degraded"

    return health


# ============= Error Handlers =============


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error messages"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============= CORS Configuration (if needed for web UI) =============

from fastapi.middleware.cors import CORSMiddleware

# Add this after app = FastAPI() if you need CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Startup Event Enhancement =============

# REPLACE the existing startup_event with this enhanced version:


@app.on_event("startup")
async def startup_event():
    """Initialize all system components with enhanced error handling"""
    global decision_engine, multi_scale_coordinator, integration_manager, state_manager, config

    logger.info("=" * 60)
    logger.info("Starting Senga Sequential Decision Engine")
    logger.info("=" * 60)

    try:
        # Load configuration
        config = SengaConfigurator()
        logger.info(" Configuration loaded")

        # Initialize state manager
        state_manager = StateManager()
        logger.info(" State Manager initialized")

        # Initialize integrations
        integration_config = {
            "google_places_api_key": config.business.get("google_places_api_key", ""),
            "oms_base_url": config.business.get("oms_base_url", ""),
            "oms_api_key": config.business.get("oms_api_key", ""),
            "driver_app_base_url": config.business.get("driver_app_base_url", ""),
            "driver_app_api_key": config.business.get("driver_app_api_key", ""),
        }
        integration_manager = IntegrationManager(integration_config)
        logger.info(" Integration Manager initialized")

        # Initialize decision engine (with enhanced version)
        decision_engine = DecisionEngine()
        logger.info(" Decision Engine initialized")

        # Initialize multi-scale coordinator
        multi_scale_coordinator = MultiScaleCoordinator()
        logger.info(" Multi-Scale Coordinator initialized")

        logger.info("=" * 60)
        logger.info("Senga SDE started successfully!")
        logger.info("API available at http://localhost:8000")
        logger.info("Docs available at http://localhost:8000/docs")
        # logger.info("Streamlit UI: streamlit run streamlit_app_complete.py")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to start Senga SDE: {e}", exc_info=True)
        raise


# ============= Order Management Endpoints =============


@app.post("/orders", response_model=OrderResponse)
async def create_order(order_request: OrderCreateRequest):
    """
    Create a new order and persist to database

    This endpoint:
    1. Receives order in API format
    2. Converts to Shipment dataclass using adapter
    3. Persists to SQLite database via StateManager
    4. Returns order in API format for UI display
    """
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        # Convert API format to dict for adapter
        order_dict = order_request.dict()

        # Use adapter to convert to Shipment dataclass
        shipment = OrderAdapter.from_api_to_shipment(order_dict)

        # Persist to database
        success = state_manager.add_shipment(shipment)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to save order to database"
            )

        logger.info(f" Order {shipment.id} created and persisted to database")

        # Convert back to API format for response
        response_dict = OrderAdapter.from_shipment_to_api(shipment)

        return OrderResponse(**response_dict)

    except Exception as e:
        logger.error(f"Failed to create order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Order creation failed: {str(e)}")


@app.get("/orders/pending", response_model=List[OrderResponse])
async def get_pending_orders():
    """
    Get all pending orders from database

    Returns orders in API format ready for UI display
    """
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        # Get pending shipments from database
        pending_shipments = state_manager.get_pending_shipments()

        # Convert to API format using adapter
        orders_api_format = OrderAdapter.batch_from_shipments_to_api(pending_shipments)

        logger.info(f" Retrieved {len(orders_api_format)} pending orders from database")

        return [OrderResponse(**order) for order in orders_api_format]

    except Exception as e:
        logger.error(f"Failed to retrieve pending orders: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve orders: {str(e)}"
        )


@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get specific order by ID"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        shipment = state_manager.get_shipment(order_id)

        if not shipment:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        response_dict = OrderAdapter.from_shipment_to_api(shipment)
        return OrderResponse(**response_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve order {order_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve order: {str(e)}"
        )


@app.get("/routes/completed")
async def get_completed_routes(limit: int = 20):
    """
    Get completed routes with outcomes

    Args:
        limit: Maximum number of routes to return (default: 20)

    Returns:
        List of completed routes with performance data
    """
    try:
        if not state_manager:
            raise HTTPException(status_code=503, detail="State manager not initialized")

        # Get completed routes from state manager
        completed_routes = state_manager.get_completed_routes(limit=limit)

        # Format response
        routes_data = []
        for route, outcome in completed_routes:
            route_dict = route.to_dict()

            # Enrich with outcome data
            route_dict["outcome"] = {
                "estimated_cost": outcome["estimated_cost"],
                "actual_cost": outcome["actual_cost"],
                "cost_variance": outcome["actual_cost"] - outcome["estimated_cost"],
                "estimated_duration_hours": outcome["estimated_duration"],
                "actual_duration_hours": outcome["actual_duration"],
                "duration_variance": outcome["actual_duration"]
                - outcome["estimated_duration"],
                "utilization": outcome["utilization"],
                "on_time_deliveries": outcome["on_time_deliveries"],
                "total_deliveries": outcome["total_deliveries"],
                "on_time_rate": (
                    outcome["on_time_deliveries"] / outcome["total_deliveries"]
                    if outcome["total_deliveries"] > 0
                    else 0
                ),
                "delay_penalties": outcome["delay_penalties"],
            }

            routes_data.append(route_dict)

        logger.info(f"Retrieved {len(routes_data)} completed routes")
        return routes_data

    except Exception as e:
        logger.error(f"Failed to get completed routes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ALTERNATIVE: If routes table doesn't have status='completed', use this version

# @app.get("/routes/completed")
# async def get_completed_routes_alt(limit: int = 20):
#     """
#     Get completed routes - Alternative implementation if status tracking not implemented

#     Args:
#         limit: Maximum number of routes to return (default: 20)

#     Returns:
#         List of completed routes
#     """
#     try:
#         if not state_manager:
#             raise HTTPException(status_code=503, detail="State manager not initialized")

#         # Query route_outcomes table directly
#         cursor = state_manager.conn.execute("""
#             SELECT
#                 ro.route_id,
#                 ro.estimated_cost,
#                 ro.actual_cost,
#                 ro.estimated_duration,
#                 ro.actual_duration,
#                 ro.utilization,
#                 ro.on_time_deliveries,
#                 ro.total_deliveries,
#                 ro.delay_penalties,
#                 ro.created_at,
#                 r.data as route_data
#             FROM route_outcomes ro
#             LEFT JOIN routes r ON ro.route_id = r.id
#             ORDER BY ro.created_at DESC
#             LIMIT ?
#         """, (limit,))

#         routes_data = []
#         for row in cursor.fetchall():
#             import json

#             route_dict = {}
#             if row[10]:  # If route data exists
#                 try:
#                     route_dict = json.loads(row[10])
#                 except:
#                     pass

#             routes_data.append({
#                 'route_id': row[0],
#                 'completed_at': row[9],
#                 'outcome': {
#                     'estimated_cost': row[1],
#                     'actual_cost': row[2],
#                     'cost_variance': row[2] - row[1],
#                     'estimated_duration_hours': row[3],
#                     'actual_duration_hours': row[4],
#                     'duration_variance': row[4] - row[3],
#                     'utilization': row[5],
#                     'on_time_deliveries': row[6],
#                     'total_deliveries': row[7],
#                     'on_time_rate': row[6] / row[7] if row[7] > 0 else 0,
#                     'delay_penalties': row[8]
#                 },
#                 **route_dict
#             })

#         logger.info(f"Retrieved {len(routes_data)} completed routes")
#         return routes_data

#     except Exception as e:
#         logger.error(f"Failed to get completed routes: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
# ============= Decision Engine Endpoints =============
from src.api.route_completion_api import router as routes_router

app.include_router(routes_router)


@app.post("/decisions/consolidation-cycle", response_model=ConsolidationCycleResponse)
async def trigger_consolidation_cycle(request: ConsolidationCycleRequest):
    """
    Trigger a consolidation cycle

    This is the main endpoint for batch decision-making:
    1. Gets all pending orders from database
    2. Runs decision engine cycle (meta-controller → PFA/CFA/VFA/DLA)
    3. Executes decision (dispatch batches or wait)
    4. Returns which orders were dispatched and which are waiting
    """
    if not decision_engine or not state_manager:
        raise HTTPException(status_code=503, detail="System not fully initialized")

    try:
        # Get current system state (all pending orders + vehicles)
        current_state = state_manager.get_current_state(force_refresh=True)

        logger.info(
            f"Running consolidation cycle with {len(current_state.pending_shipments)} pending orders"
        )

        # Run decision cycle using DecisionEngine.run_cycle()
        cycle_result = decision_engine.run_cycle()

        # Get updated state after execution
        updated_state = state_manager.get_current_state(force_refresh=True)

        # Separate dispatched vs waiting orders
        dispatched_orders = []
        waiting_orders = []

        for shipment in current_state.pending_shipments:
            # Check if shipment status changed to EN_ROUTE or BATCHED
            updated_shipment = state_manager.get_shipment(shipment.id)
            if updated_shipment and updated_shipment.status in [
                ShipmentStatus.EN_ROUTE
            ]:
                dispatched_orders.append(
                    OrderAdapter.from_shipment_to_api(updated_shipment)
                )
            else:
                waiting_orders.append(OrderAdapter.from_shipment_to_api(shipment))

        # Build batch information from decision
        dispatched_batches = []
        if cycle_result.decision.action_type == "DISPATCH":
            batches = cycle_result.decision.action_details.get("batches", [])
            for batch in batches:
                batch_info = {
                    "vehicle_id": batch.get("vehicle"),
                    "shipments": batch.get("shipments", []),
                    "route": batch.get("route", []),
                    "estimated_distance_km": batch.get("distance", 0),
                    "estimated_duration_hours": batch.get("duration", 0),
                }
                dispatched_batches.append(batch_info)

        logger.info(
            f" Consolidation cycle complete: {len(dispatched_orders)} dispatched, {len(waiting_orders)} waiting"
        )

        return ConsolidationCycleResponse(
            timestamp=datetime.now().isoformat(),
            total_pending_orders=len(current_state.pending_shipments),
            orders_dispatched=len(dispatched_orders),
            orders_waiting=len(waiting_orders),
            batches_created=len(dispatched_batches),
            function_class_used=cycle_result.decision.function_class.value,
            reasoning=cycle_result.decision.reasoning,
            dispatched_batches=dispatched_batches,
            waiting_orders=waiting_orders,
        )

    except Exception as e:
        logger.error(f"Consolidation cycle failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Consolidation cycle failed: {str(e)}"
        )


# ============= Demo and Testing Endpoints =============


@app.post("/demo/initialize")
async def initialize_demo_data():
    """Initialize demo data for testing"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    try:
        # This endpoint can load pre-configured demo orders
        # For now, return success
        return {"status": "success", "message": "Demo initialization complete"}

    except Exception as e:
        logger.error(f"Demo initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============= Run Server =============

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
