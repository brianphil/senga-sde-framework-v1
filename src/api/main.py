# main.py

"""
Senga Sequential Decision Engine - Main Application
Entry point for the consolidation optimization system
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Core components
from src.core.decision_engine import DecisionEngine, EngineStatus
from src.core.multi_scale_coordinator import MultiScaleCoordinator
from src.core.state_manager import StateManager, Shipment, ShipmentStatus
from src.integrations.external_systems import IntegrationManager
from src.config.senga_config import SengaConfigurator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('senga_sde.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Senga Sequential Decision Engine",
    description="AI-powered freight consolidation and route optimization for African logistics",
    version="1.0.0"
)

# Global instances (initialized on startup)
decision_engine: Optional[DecisionEngine] = None
multi_scale_coordinator: Optional[MultiScaleCoordinator] = None
integration_manager: Optional[IntegrationManager] = None
state_manager: Optional[StateManager] = None
config: Optional[SengaConfigurator] = None

# Request/Response Models
class OrderIngest(BaseModel):
    """Request model for ingesting new orders"""
    order_id: str
    customer_id: str
    pickup_location: str
    delivery_location: str
    weight_kg: float
    volume_m3: float
    declared_value: float
    delivery_deadline: str  # ISO format
    priority: str = "standard"

class TriggerCycleResponse(BaseModel):
    """Response model for cycle trigger"""
    cycle_number: int
    timestamp: str
    decision_type: str
    function_class: str
    shipments_dispatched: int
    vehicles_utilized: int
    execution_time_ms: float

class SystemStatus(BaseModel):
    """System status response"""
    engine_status: str
    integrations_online: bool
    pending_shipments: int
    available_vehicles: int
    current_cycle: int
    last_decision_time: Optional[str]

# Startup and Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize all system components"""
    global decision_engine, multi_scale_coordinator, integration_manager, state_manager, config
    
    logger.info("=" * 60)
    logger.info("Starting Senga Sequential Decision Engine")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = SengaConfigurator()
        logger.info("✓ Configuration loaded")
        
        # Initialize state manager
        state_manager = StateManager()
        logger.info("✓ State Manager initialized")
        
        # Initialize integrations
        integration_config = {
            'google_places_api_key': config.business.get('google_places_api_key', ''),
            'oms_base_url': config.business.get('oms_base_url', ''),
            'oms_api_key': config.business.get('oms_api_key', ''),
            'driver_app_base_url': config.business.get('driver_app_base_url', ''),
            'driver_app_api_key': config.business.get('driver_app_api_key', '')
        }
        integration_manager = IntegrationManager(integration_config)
        logger.info("✓ Integration Manager initialized")
        
        # Initialize decision engine
        decision_engine = DecisionEngine()
        logger.info("✓ Decision Engine initialized")
        
        # Initialize multi-scale coordinator
        multi_scale_coordinator = MultiScaleCoordinator()
        logger.info("✓ Multi-Scale Coordinator initialized")
        
        logger.info("=" * 60)
        logger.info("Senga SDE started successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to start Senga SDE: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Senga Sequential Decision Engine")
    
    # Save any pending state
    if state_manager:
        state_manager.close()
    
    logger.info("Shutdown complete")

# Health and Status Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    current_state = state_manager.get_current_state()
    metrics = decision_engine.get_performance_metrics()
    
    last_cycle = decision_engine.cycle_history[-1] if decision_engine.cycle_history else None
    
    return SystemStatus(
        engine_status=decision_engine.status.value,
        integrations_online=integration_manager.is_online(),
        pending_shipments=len(current_state.pending_shipments),
        available_vehicles=len([v for v in current_state.vehicles if v.status.value == 'available']),
        current_cycle=decision_engine.current_cycle,
        last_decision_time=last_cycle.timestamp.isoformat() if last_cycle else None
    )

@app.get("/integrations/status")
async def get_integration_status():
    """Get status of all external integrations"""
    if not integration_manager:
        raise HTTPException(status_code=503, detail="Integration manager not initialized")
    
    return integration_manager.get_health_status()

# Order Management Endpoints
@app.post("/orders/ingest")
async def ingest_order(order: OrderIngest):
    """
    Ingest a new order into the system
    
    This endpoint receives orders from the OMS and adds them to the pending queue
    """
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        # Geocode addresses using Google Places
        pickup_geocode = integration_manager.google_places.geocode_address(
            order.pickup_location,
            bias_location=(-1.286389, 36.817223)  # Nairobi bias
        )
        
        delivery_geocode = integration_manager.google_places.geocode_address(
            order.delivery_location,
            bias_location=(-1.286389, 36.817223)
        )
        
        # Create shipment
        shipment = Shipment(
            shipment_id=order.order_id,
            origin_lat=pickup_geocode.latitude,
            origin_lon=pickup_geocode.longitude,
            origin_address=pickup_geocode.formatted_address,
            dest_lat=delivery_geocode.latitude,
            dest_lon=delivery_geocode.longitude,
            dest_address=delivery_geocode.formatted_address,
            weight_kg=order.weight_kg,
            volume_m3=order.volume_m3,
            declared_value=order.declared_value,
            customer_id=order.customer_id,
            created_at=datetime.now(),
            delivery_deadline=datetime.fromisoformat(order.delivery_deadline),
            priority=order.priority,
            status=ShipmentStatus.PENDING,
            geocoding_confidence=min(pickup_geocode.confidence, delivery_geocode.confidence)
        )
        
        # Add to state
        state_manager.add_shipment(shipment)
        
        logger.info(f"Order {order.order_id} ingested successfully")
        
        return {
            "status": "success",
            "order_id": order.order_id,
            "geocoding_confidence": shipment.geocoding_confidence,
            "message": "Order added to pending queue"
        }
        
    except Exception as e:
        logger.error(f"Failed to ingest order {order.order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/pending")
async def get_pending_orders():
    """Get all pending orders"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    current_state = state_manager.get_current_state()
    
    return {
        "count": len(current_state.pending_shipments),
        "orders": [
            {
                "order_id": s.shipment_id,
                "origin": s.origin_address,
                "destination": s.dest_address,
                "weight_kg": s.weight_kg,
                "age_hours": (datetime.now() - s.created_at).total_seconds() / 3600,
                "deadline": s.delivery_deadline.isoformat(),
                "priority": s.priority
            }
            for s in current_state.pending_shipments
        ]
    }

# Decision Engine Endpoints
@app.post("/engine/cycle", response_model=TriggerCycleResponse)
async def trigger_decision_cycle():
    """
    Manually trigger a decision cycle
    
    Useful for testing or emergency situations
    """
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = decision_engine.run_cycle()
        
        return TriggerCycleResponse(
            cycle_number=result.timestamp.timestamp(),
            timestamp=result.timestamp.isoformat(),
            decision_type=result.decision.action_type,
            function_class=result.decision.function_class.value,
            shipments_dispatched=result.shipments_dispatched,
            vehicles_utilized=result.vehicles_utilized,
            execution_time_ms=result.execution_time_ms
        )
        
    except Exception as e:
        logger.error(f"Cycle execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/engine/start")
async def start_continuous_engine(background_tasks: BackgroundTasks):
    """
    Start the engine in continuous mode (background task)
    
    Runs decision cycles at configured intervals
    """
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if decision_engine.status == EngineStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Engine already running")
    
    # Run in background
    background_tasks.add_task(decision_engine.run_continuous)
    
    return {
        "status": "started",
        "message": "Decision engine started in continuous mode"
    }

@app.post("/engine/stop")
async def stop_continuous_engine():
    """Stop the continuous engine"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Set status to paused (the continuous loop will check this)
    decision_engine.status = EngineStatus.PAUSED
    
    return {
        "status": "stopped",
        "message": "Decision engine stopped"
    }

@app.get("/engine/metrics")
async def get_engine_metrics():
    """Get performance metrics"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    metrics = decision_engine.get_performance_metrics()
    
    return {
        "total_cycles": metrics.total_cycles,
        "total_shipments_processed": metrics.total_shipments_processed,
        "total_dispatches": metrics.total_dispatches,
        "avg_utilization": f"{metrics.avg_utilization:.2%}",
        "avg_sla_compliance": f"{metrics.avg_sla_compliance:.2%}",
        "avg_cycle_time_ms": f"{metrics.avg_cycle_time_ms:.1f}",
        "function_class_usage": metrics.function_class_usage
    }

@app.get("/engine/history")
async def get_decision_history(limit: int = 20):
    """Get recent decision history"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    history = decision_engine.cycle_history[-limit:]
    
    return {
        "count": len(history),
        "decisions": [
            {
                "cycle": i,
                "timestamp": result.timestamp.isoformat(),
                "function_class": result.decision.function_class.value,
                "action": result.decision.action_type,
                "shipments_dispatched": result.shipments_dispatched,
                "vehicles_used": result.vehicles_utilized,
                "reasoning": result.decision.reasoning,
                "confidence": result.decision.confidence,
                "execution_time_ms": result.execution_time_ms
            }
            for i, result in enumerate(history)
        ]
    }

# Learning and Analytics Endpoints
@app.post("/learning/daily-update")
async def trigger_daily_update():
    """
    Trigger daily strategic learning update
    
    Aggregates previous day's performance and updates models
    """
    if not multi_scale_coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        analytics = multi_scale_coordinator.run_daily_strategic_update()
        
        return {
            "status": "success",
            "date": analytics.date.isoformat(),
            "total_routes": analytics.total_routes,
            "total_shipments": analytics.total_shipments,
            "avg_utilization": f"{analytics.avg_utilization:.2%}",
            "sla_compliance_rate": f"{analytics.sla_compliance_rate:.2%}",
            "cost_prediction_error": f"{analytics.cost_prediction_error:.2%}",
            "duration_prediction_error": f"{analytics.duration_prediction_error:.2%}",
            "top_delay_causes": analytics.top_delay_causes
        }
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/weekly-analysis")
async def trigger_weekly_analysis():
    """
    Trigger weekly strategic analysis
    
    Deep analysis of patterns, network topology, and optimization opportunities
    """
    if not multi_scale_coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        insights = multi_scale_coordinator.run_weekly_strategic_analysis()
        
        return {
            "status": "success",
            "week_start": insights.week_start.isoformat(),
            "route_patterns": len(insights.route_patterns_discovered),
            "optimal_consolidation_windows": insights.optimal_consolidation_windows,
            "utilization_by_day": insights.fleet_utilization_by_day,
            "fleet_recommendations": insights.recommended_fleet_adjustments,
            "network_topology": insights.network_topology_updates,
            "customer_insights": insights.customer_pattern_insights
        }
        
    except Exception as e:
        logger.error(f"Weekly analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/utilization")
async def get_utilization_analytics():
    """Get utilization analytics"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    analytics = state_manager.get_analytics()
    
    return {
        "current_utilization": analytics.get('current_utilization', 0.0),
        "avg_utilization_7d": analytics.get('avg_utilization_7d', 0.0),
        "avg_utilization_30d": analytics.get('avg_utilization_30d', 0.0),
        "target_utilization": config.business.get('min_utilization_threshold', 0.75)
    }

@app.get("/analytics/sla")
async def get_sla_analytics():
    """Get SLA compliance analytics"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    analytics = state_manager.get_analytics()
    
    return {
        "compliance_rate_7d": analytics.get('sla_compliance_7d', 0.0),
        "compliance_rate_30d": analytics.get('sla_compliance_30d', 0.0),
        "at_risk_shipments": analytics.get('at_risk_shipments', 0),
        "average_delay_hours": analytics.get('avg_delay_hours', 0.0)
    }

# Fleet Management Endpoints
@app.get("/fleet/vehicles")
async def get_fleet_status():
    """Get current fleet status"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    current_state = state_manager.get_current_state()
    
    vehicles_by_status = {}
    for vehicle in current_state.vehicles:
        status = vehicle.status.value
        vehicles_by_status[status] = vehicles_by_status.get(status, 0) + 1
    
    return {
        "total_vehicles": len(current_state.vehicles),
        "by_status": vehicles_by_status,
        "vehicles": [
            {
                "vehicle_id": v.vehicle_id,
                "status": v.status.value,
                "capacity_kg": v.capacity_kg,
                "capacity_m3": v.capacity_m3,
                "current_location": v.current_location,
                "assigned_shipments": v.assigned_shipments
            }
            for v in current_state.vehicles
        ]
    }

@app.get("/fleet/utilization")
async def get_fleet_utilization():
    """Get detailed fleet utilization metrics"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    analytics = state_manager.get_analytics()
    
    return {
        "overall_utilization": analytics.get('avg_utilization', 0.0),
        "vehicles_in_use": analytics.get('vehicles_in_use', 0),
        "vehicles_available": analytics.get('vehicles_available', 0),
        "avg_shipments_per_route": analytics.get('avg_shipments_per_route', 0),
        "avg_route_distance_km": analytics.get('avg_route_distance', 0)
    }

# Configuration Management (delegates to existing config API)
@app.get("/config/business")
async def get_business_config():
    """Get current business configuration"""
    if not config:
        raise HTTPException(status_code=503, detail="Config not initialized")
    
    return config.business._cache

@app.get("/config/model")
async def get_model_config():
    """Get current model configuration"""
    if not config:
        raise HTTPException(status_code=503, detail="Config not initialized")
    
    return config.model_config

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Senga Sequential Decision Engine")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Senga SDE API server on {args.host}:{args.port}")

    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload, workers=args.workers)
