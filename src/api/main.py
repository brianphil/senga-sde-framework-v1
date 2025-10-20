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
from datetime import timedelta
# Core components
from src.core.decision_engine import DecisionEngine, EngineStatus
from src.core.multi_scale_coordinator import MultiScaleCoordinator
from src.core.state_manager import StateManager, Shipment, ShipmentStatus, Location
from src.integrations.external_systems import IntegrationManager
from src.config.senga_config import SengaConfigurator
from src.api.demo_routes import router as demo_router
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
app.include_router(demo_router)
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

@app.get("/state/current")
async def get_current_system_state():
    """Get current system state for dashboard"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        current_state = state_manager.get_current_state()
        
        # Count shipments by status
        pending_count = len(current_state.pending_shipments)
        active_routes_count = len(current_state.active_routes)
        
        # Count vehicles by status
        available_drivers = len([v for v in current_state.fleet_state 
                                if v.status == VehicleStatus.IDLE])
        busy_drivers = len([v for v in current_state.fleet_state 
                           if v.status in [VehicleStatus.EN_ROUTE, VehicleStatus.LOADING]])
        
        # Get completed today (this would need to query from decision log)
        # For now, return 0 as placeholder
        completed_today = 0
        
        # Get learning iterations if available
        learning_iterations = 0
        if decision_engine and hasattr(decision_engine, 'vfa'):
            learning_iterations = decision_engine.vfa.update_count if hasattr(decision_engine.vfa, 'update_count') else 0
        
        return {
            "timestamp": current_state.timestamp.isoformat(),
            "active_orders": pending_count,
            "available_drivers": available_drivers,
            "busy_drivers": busy_drivers,
            "active_routes": active_routes_count,
            "completed_today": completed_today,
            "learning_iterations": learning_iterations,
            "pending_shipments": [
                {
                    "id": s.id,
                    "customer_id": s.customer_id,
                    "origin": s.origin.formatted_address,
                    "destinations": [d.formatted_address for d in s.destinations],
                    "weight": s.weight,
                    "volume": s.volume,
                    "priority": s.priority,
                    "status": s.status.value,
                    "time_to_deadline_hours": s.time_to_deadline(current_state.timestamp).total_seconds() / 3600
                }
                for s in current_state.pending_shipments[:20]  # Limit to 20 for performance
            ],
            "fleet": [
                {
                    "id": v.id,
                    "type": v.vehicle_type,
                    "status": v.status.value,
                    "capacity_volume": v.capacity.volume,
                    "capacity_weight": v.capacity.weight,
                    "location": v.current_location.formatted_address,
                    "current_route": v.current_route_id
                }
                for v in current_state.fleet_state
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get current state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/learning")
async def get_learning_metrics():
    """Get VFA learning metrics for dashboard"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        if not hasattr(decision_engine, 'vfa'):
            return {
                "available": False,
                "message": "VFA not initialized yet"
            }
        
        vfa = decision_engine.vfa
        
        # Get basic metrics
        metrics = {
            "available": True,
            "update_count": getattr(vfa, 'update_count', 0),
            "learning_rate": getattr(vfa, 'learning_rate', 0.0),
            "epsilon": getattr(vfa, 'epsilon', 0.0),
        }
        
        # Get TD error history if available
        if hasattr(vfa, 'td_error_history'):
            metrics["td_error_history"] = vfa.td_error_history[-100:]  # Last 100
            metrics["avg_td_error"] = sum(vfa.td_error_history[-20:]) / min(20, len(vfa.td_error_history)) if vfa.td_error_history else 0.0
        
        # Get feature weights if available
        if hasattr(vfa, 'weights') and vfa.weights is not None:
            # Convert numpy array to dict if needed
            if hasattr(vfa.weights, 'shape'):
                metrics["num_features"] = vfa.weights.shape[0]
                metrics["weight_norm"] = float(sum(w**2 for w in vfa.weights)**0.5)
            else:
                metrics["feature_weights"] = dict(vfa.weights) if isinstance(vfa.weights, dict) else {}
        
        # Get convergence metrics
        if hasattr(vfa, 'convergence_score'):
            metrics["convergence_score"] = vfa.convergence_score
        
        # Performance history
        if hasattr(vfa, 'performance_history'):
            metrics["performance_history"] = vfa.performance_history[-50:]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}", exc_info=True)
        return {
            "available": False,
            "error": str(e)
        }


@app.get("/orders")
async def get_orders(status: Optional[str] = None, limit: int = 50):
    """Get orders with optional status filter"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        current_state = state_manager.get_current_state()
        
        shipments = current_state.pending_shipments
        
        # Filter by status if provided
        if status:
            try:
                status_enum = ShipmentStatus(status.lower())
                shipments = [s for s in shipments if s.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Limit results
        shipments = shipments[:limit]
        
        return {
            "count": len(shipments),
            "orders": [
                {
                    "order_id": s.id,
                    "customer_id": s.customer_id,
                    "origin": s.origin.formatted_address,
                    "destinations": [d.formatted_address for d in s.destinations],
                    "weight": s.weight,
                    "volume": s.volume,
                    "priority": s.priority,
                    "status": s.status.value,
                    "created_at": s.creation_time.isoformat(),
                    "deadline": s.deadline.isoformat(),
                    "time_to_deadline_hours": s.time_to_deadline(current_state.timestamp).total_seconds() / 3600
                }
                for s in shipments
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orders")
async def create_order(request: dict):
    """
    Create a new order - flexible endpoint that accepts both formats
    Accepts either OrderIngest format or Streamlit demo format
    """
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        from uuid import uuid4
        
        # Generate order ID if not provided
        order_id = request.get("order_id", f"ORD{uuid4().hex[:8].upper()}")
        
        # Handle customer info
        customer_id = request.get("customer_id", f"CUST{uuid4().hex[:8].upper()}")
        customer_name = request.get("customer_name", "Unknown Customer")
        
        # Handle locations - could be string or dict
        pickup_location_data = request.get("pickup_location")
        delivery_location_data = request.get("delivery_location")
        
        # Convert to Location objects
        if isinstance(pickup_location_data, dict):
            pickup_location = Location(
                place_id=f"demo_{pickup_location_data.get('address', 'unknown').replace(' ', '_').lower()}",
                lat=pickup_location_data.get("latitude", 0.0),
                lng=pickup_location_data.get("longitude", 0.0),
                formatted_address=pickup_location_data.get("address", "Unknown"),
                zone_id="general"
            )
        else:
            # String address - create a simple location
            pickup_location = Location(
                place_id=f"demo_{pickup_location_data.replace(' ', '_').lower() if pickup_location_data else 'unknown'}",
                lat=0.0,
                lng=0.0,
                formatted_address=pickup_location_data or "Unknown",
                zone_id="general"
            )
        
        if isinstance(delivery_location_data, dict):
            delivery_location = Location(
                place_id=f"demo_{delivery_location_data.get('address', 'unknown').replace(' ', '_').lower()}",
                lat=delivery_location_data.get("latitude", 0.0),
                lng=delivery_location_data.get("longitude", 0.0),
                formatted_address=delivery_location_data.get("address", "Unknown"),
                zone_id="general"
            )
        else:
            delivery_location = Location(
                place_id=f"demo_{delivery_location_data.replace(' ', '_').lower() if delivery_location_data else 'unknown'}",
                lat=0.0,
                lng=0.0,
                formatted_address=delivery_location_data or "Unknown",
                zone_id="general"
            )
        
        # Handle weight and volume
        weight = request.get("package_weight") or request.get("weight_kg") or 10.0
        volume = request.get("volume_m3", weight * 0.01)  # Estimate if not provided
        
        # Handle priority
        priority_str = request.get("priority", "standard")
        priority = 2 if priority_str == "standard" else (1 if priority_str == "urgent" else 3)
        
        # Handle deadline
        deadline_str = request.get("delivery_deadline") or request.get("created_at")
        if deadline_str:
            creation_time = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
        else:
            creation_time = datetime.now()
        
        deadline = creation_time + timedelta(days=2)  # Default 2-day deadline
        
        # Create shipment
        shipment = Shipment(
            id=order_id,
            customer_id=customer_id,
            origin=pickup_location,
            destinations=[delivery_location],
            weight=weight,
            volume=volume,
            creation_time=creation_time,
            deadline=deadline,
            priority=priority,
            status=ShipmentStatus.PENDING
        )
        
        # Add to state manager
        state_manager.add_shipment(shipment)
        
        return {
            "success": True,
            "order_id": order_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "pickup_location": pickup_location_data,
            "delivery_location": delivery_location_data,
            "package_weight": weight,
            "priority": priority_str,
            "status": "pending",
            "created_at": creation_time.isoformat(),
            "deadline": deadline.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# REPLACE your /decisions/route endpoint with this EXACT code:

@app.post("/decisions/route")
async def get_routing_decision(request: dict):
    """Get routing decision for a specific order"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        order_id = request.get("order_id")
        context = request.get("context", {})
        
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id is required")
        
        # Trigger decision cycle
        result = decision_engine.run_cycle()
        
        # Extract function class safely
        func_class = "UNKNOWN"  # Default
        try:
            if hasattr(result, 'decision') and hasattr(result.decision, 'function_class'):
                # It's a FunctionClass enum, get the value
                func_class = result.decision.function_class.value.upper()
        except Exception as e:
            logger.error(f"Error extracting function_class: {e}")
            func_class = "UNKNOWN"
        
        # Extract action type
        action = "WAIT"
        try:
            if hasattr(result, 'decision') and hasattr(result.decision, 'action_type'):
                action = result.decision.action_type
        except:
            pass
        
        # Extract reasoning
        reasoning = "Waiting for more shipments to consolidate"
        try:
            if hasattr(result, 'decision') and hasattr(result.decision, 'reasoning'):
                reasoning = result.decision.reasoning
        except:
            pass
        
        # Extract confidence
        confidence = 0.5
        try:
            if hasattr(result, 'decision') and hasattr(result.decision, 'confidence'):
                confidence = result.decision.confidence
        except:
            pass
        
        # Return response in format Streamlit expects
        return {
            "success": True,
            "order_id": order_id,
            "decision": action,
            "function_used": func_class,
            "selection_reason": reasoning,
            "reasoning": reasoning,
            "confidence": confidence,
            "driver_id": "N/A",  # Will be N/A for WAIT decisions
            "estimated_time_minutes": 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get routing decision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/current")
async def get_current_system_state():
    """Get current system state for dashboard"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        current_state = state_manager.get_current_state()
        
        # Count shipments by status
        pending_count = len(current_state.pending_shipments)
        active_routes_count = len(current_state.active_routes)
        
        # Count vehicles by status
        available_drivers = len([v for v in current_state.fleet_state 
                                if v.status == VehicleStatus.IDLE])
        busy_drivers = len([v for v in current_state.fleet_state 
                           if v.status in [VehicleStatus.EN_ROUTE, VehicleStatus.LOADING]])
        
        # Get completed today (this would need to query from decision log)
        # For now, return 0 as placeholder
        completed_today = 0
        
        # Get learning iterations if available
        learning_iterations = 0
        if decision_engine and hasattr(decision_engine, 'vfa'):
            learning_iterations = decision_engine.vfa.update_count if hasattr(decision_engine.vfa, 'update_count') else 0
        
        return {
            "timestamp": current_state.timestamp.isoformat(),
            "active_orders": pending_count,
            "available_drivers": available_drivers,
            "busy_drivers": busy_drivers,
            "active_routes": active_routes_count,
            "completed_today": completed_today,
            "learning_iterations": learning_iterations,
            "pending_shipments": [
                {
                    "id": s.id,
                    "customer_id": s.customer_id,
                    "origin": s.origin.formatted_address,
                    "destinations": [d.formatted_address for d in s.destinations],
                    "weight": s.weight,
                    "volume": s.volume,
                    "priority": s.priority,
                    "status": s.status.value,
                    "time_to_deadline_hours": s.time_to_deadline(current_state.timestamp).total_seconds() / 3600
                }
                for s in current_state.pending_shipments[:20]  # Limit to 20 for performance
            ],
            "fleet": [
                {
                    "id": v.id,
                    "type": v.vehicle_type,
                    "status": v.status.value,
                    "capacity_volume": v.capacity.volume,
                    "capacity_weight": v.capacity.weight,
                    "location": v.current_location.formatted_address,
                    "current_route": v.current_route_id
                }
                for v in current_state.fleet_state
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get current state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/learning")
async def get_learning_metrics():
    """Get VFA learning metrics for dashboard"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        if not hasattr(decision_engine, 'vfa'):
            return {
                "available": False,
                "message": "VFA not initialized yet"
            }
        
        vfa = decision_engine.vfa
        
        # Get basic metrics
        metrics = {
            "available": True,
            "update_count": getattr(vfa, 'update_count', 0),
            "learning_rate": getattr(vfa, 'learning_rate', 0.0),
            "epsilon": getattr(vfa, 'epsilon', 0.0),
        }
        
        # Get TD error history if available
        if hasattr(vfa, 'td_error_history'):
            metrics["td_error_history"] = vfa.td_error_history[-100:]  # Last 100
            metrics["avg_td_error"] = sum(vfa.td_error_history[-20:]) / min(20, len(vfa.td_error_history)) if vfa.td_error_history else 0.0
        
        # Get feature weights if available
        if hasattr(vfa, 'weights') and vfa.weights is not None:
            # Convert numpy array to dict if needed
            if hasattr(vfa.weights, 'shape'):
                metrics["num_features"] = vfa.weights.shape[0]
                metrics["weight_norm"] = float(sum(w**2 for w in vfa.weights)**0.5)
            else:
                metrics["feature_weights"] = dict(vfa.weights) if isinstance(vfa.weights, dict) else {}
        
        # Get convergence metrics
        if hasattr(vfa, 'convergence_score'):
            metrics["convergence_score"] = vfa.convergence_score
        
        # Performance history
        if hasattr(vfa, 'performance_history'):
            metrics["performance_history"] = vfa.performance_history[-50:]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}", exc_info=True)
        return {
            "available": False,
            "error": str(e)
        }


@app.get("/orders")
async def get_orders(status: Optional[str] = None, limit: int = 50):
    """Get orders with optional status filter"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        current_state = state_manager.get_current_state()
        
        shipments = current_state.pending_shipments
        
        # Filter by status if provided
        if status:
            try:
                status_enum = ShipmentStatus(status.lower())
                shipments = [s for s in shipments if s.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Limit results
        shipments = shipments[:limit]
        
        return {
            "count": len(shipments),
            "orders": [
                {
                    "order_id": s.id,
                    "customer_id": s.customer_id,
                    "origin": s.origin.formatted_address,
                    "destinations": [d.formatted_address for d in s.destinations],
                    "weight": s.weight,
                    "volume": s.volume,
                    "priority": s.priority,
                    "status": s.status.value,
                    "created_at": s.creation_time.isoformat(),
                    "deadline": s.deadline.isoformat(),
                    "time_to_deadline_hours": s.time_to_deadline(current_state.timestamp).total_seconds() / 3600
                }
                for s in shipments
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orders")
async def create_order(request: dict):
    """
    Create a new order - flexible endpoint that accepts both formats
    Accepts either OrderIngest format or Streamlit demo format
    """
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        from uuid import uuid4
        
        # Generate order ID if not provided
        order_id = request.get("order_id", f"ORD{uuid4().hex[:8].upper()}")
        
        # Handle customer info
        customer_id = request.get("customer_id", f"CUST{uuid4().hex[:8].upper()}")
        customer_name = request.get("customer_name", "Unknown Customer")
        
        # Handle locations - could be string or dict
        pickup_location_data = request.get("pickup_location")
        delivery_location_data = request.get("delivery_location")
        
        # Convert to Location objects
        if isinstance(pickup_location_data, dict):
            pickup_location = Location(
                place_id=f"demo_{pickup_location_data.get('address', 'unknown').replace(' ', '_').lower()}",
                lat=pickup_location_data.get("latitude", 0.0),
                lng=pickup_location_data.get("longitude", 0.0),
                formatted_address=pickup_location_data.get("address", "Unknown"),
                zone_id="general"
            )
        else:
            # String address - create a simple location
            pickup_location = Location(
                place_id=f"demo_{pickup_location_data.replace(' ', '_').lower() if pickup_location_data else 'unknown'}",
                lat=0.0,
                lng=0.0,
                formatted_address=pickup_location_data or "Unknown",
                zone_id="general"
            )
        
        if isinstance(delivery_location_data, dict):
            delivery_location = Location(
                place_id=f"demo_{delivery_location_data.get('address', 'unknown').replace(' ', '_').lower()}",
                lat=delivery_location_data.get("latitude", 0.0),
                lng=delivery_location_data.get("longitude", 0.0),
                formatted_address=delivery_location_data.get("address", "Unknown"),
                zone_id="general"
            )
        else:
            delivery_location = Location(
                place_id=f"demo_{delivery_location_data.replace(' ', '_').lower() if delivery_location_data else 'unknown'}",
                lat=0.0,
                lng=0.0,
                formatted_address=delivery_location_data or "Unknown",
                zone_id="general"
            )
        
        # Handle weight and volume
        weight = request.get("package_weight") or request.get("weight_kg") or 10.0
        volume = request.get("volume_m3", weight * 0.01)  # Estimate if not provided
        
        # Handle priority
        priority_str = request.get("priority", "standard")
        priority = 2 if priority_str == "standard" else (1 if priority_str == "urgent" else 3)
        
        # Handle deadline
        deadline_str = request.get("delivery_deadline") or request.get("created_at")
        if deadline_str:
            creation_time = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
        else:
            creation_time = datetime.now()
        
        deadline = creation_time + timedelta(days=2)  # Default 2-day deadline
        
        # Create shipment
        shipment = Shipment(
            id=order_id,
            customer_id=customer_id,
            origin=pickup_location,
            destinations=[delivery_location],
            weight=weight,
            volume=volume,
            creation_time=creation_time,
            deadline=deadline,
            priority=priority,
            status=ShipmentStatus.PENDING
        )
        
        # Add to state manager
        state_manager.add_shipment(shipment)
        
        return {
            "success": True,
            "order_id": order_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "pickup_location": pickup_location_data,
            "delivery_location": delivery_location_data,
            "package_weight": weight,
            "priority": priority_str,
            "status": "pending",
            "created_at": creation_time.isoformat(),
            "deadline": deadline.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decisions/route")
async def get_routing_decision(request: dict):
    """
    Get routing decision for a specific order
    This triggers a decision cycle focused on the given order
    """
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        order_id = request.get("order_id")
        context = request.get("context", {})
        
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id is required")
        
        # Get current state
        current_state = state_manager.get_current_state()
        
        # Find the shipment
        shipment = None
        for s in current_state.pending_shipments:
            if s.id == order_id:
                shipment = s
                break
        
        if not shipment:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found in pending shipments")
        
        logger.info(f"Processing routing decision for order {order_id}")
        logger.info(f"Current state: {len(current_state.pending_shipments)} pending, {len(current_state.fleet_state)} vehicles")
        
        # Trigger a decision cycle (this will use the meta-controller)
        result = decision_engine.run_cycle()
        
        logger.info(f"Decision result: {result.decision.action_type} via {result.decision.function_class.value}")
        logger.info(f"Decision details: {result.decision.action_details}")
        
        # Extract vehicle and time info from decision
        vehicle_id = None
        estimated_time = 0
        
        if result.decision.action_details:
            # Try to get vehicle from batches
            batches = result.decision.action_details.get('batches', [])
            if batches and len(batches) > 0:
                vehicle_id = batches[0].get('vehicle_id') or batches[0].get('vehicle')
                # Estimate time (rough calculation)
                estimated_time = batches[0].get('estimated_duration', 0)
        
        # Return in the format Streamlit expects
        return {
            "success": True,
            "order_id": order_id,
            "decision": result.decision.action_type,
            "function_used": result.decision.function_class.value.upper(),  # PFA, CFA, DLA
            "function_class": result.decision.function_class.value,
            "selection_reason": result.decision.reasoning,
            "reasoning": result.decision.reasoning,
            "confidence": result.decision.confidence,
            "driver_id": vehicle_id or "N/A",
            "vehicle_assigned": vehicle_id,
            "estimated_time_minutes": estimated_time / 60 if estimated_time else 0,
            "estimated_dispatch_time": result.decision.action_details.get("dispatch_time") if result.decision.action_details else None,
            "context_used": context,
            "shipments_dispatched": result.shipments_dispatched,
            "vehicles_utilized": result.vehicles_utilized,
            # Debug info
            "debug": {
                "pending_shipments": len(current_state.pending_shipments),
                "available_vehicles": len([v for v in current_state.fleet_state if v.status == VehicleStatus.IDLE]),
                "action_details": result.decision.action_details
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get routing decision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
from src.core.state_manager import VehicleStatus

@app.get("/state/current")
async def get_current_system_state():
    """Get current system state for dashboard"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        current_state = state_manager.get_current_state()
        
        # Count shipments by status
        pending_count = len(current_state.pending_shipments)
        active_routes_count = len(current_state.active_routes)
        
        # Count vehicles by status
        available_drivers = len([v for v in current_state.fleet_state 
                                if v.status == VehicleStatus.IDLE])
        busy_drivers = len([v for v in current_state.fleet_state 
                           if v.status in [VehicleStatus.EN_ROUTE, VehicleStatus.LOADING]])
        
        # Get completed today (this would need to query from decision log)
        # For now, return 0 as placeholder
        completed_today = 0
        
        # Get learning iterations if available
        learning_iterations = 0
        if decision_engine and hasattr(decision_engine, 'vfa'):
            learning_iterations = decision_engine.vfa.update_count if hasattr(decision_engine.vfa, 'update_count') else 0
        
        return {
            "timestamp": current_state.timestamp.isoformat(),
            "active_orders": pending_count,
            "available_drivers": available_drivers,
            "busy_drivers": busy_drivers,
            "active_routes": active_routes_count,
            "completed_today": completed_today,
            "learning_iterations": learning_iterations,
            "pending_shipments": [
                {
                    "id": s.id,
                    "customer_id": s.customer_id,
                    "origin": s.origin.formatted_address,
                    "destinations": [d.formatted_address for d in s.destinations],
                    "weight": s.weight,
                    "volume": s.volume,
                    "priority": s.priority,
                    "status": s.status.value,
                    "time_to_deadline_hours": s.time_to_deadline(current_state.timestamp).total_seconds() / 3600
                }
                for s in current_state.pending_shipments[:20]  # Limit to 20 for performance
            ],
            "fleet": [
                {
                    "id": v.id,
                    "type": v.vehicle_type,
                    "status": v.status.value,
                    "capacity_volume": v.capacity.volume,
                    "capacity_weight": v.capacity.weight,
                    "location": v.current_location.formatted_address,
                    "current_route": v.current_route_id
                }
                for v in current_state.fleet_state
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get current state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/learning")
async def get_learning_metrics():
    """Get VFA learning metrics for dashboard"""
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        if not hasattr(decision_engine, 'vfa'):
            return {
                "available": False,
                "message": "VFA not initialized yet"
            }
        
        vfa = decision_engine.vfa
        
        # Get basic metrics
        metrics = {
            "available": True,
            "update_count": getattr(vfa, 'update_count', 0),
            "learning_rate": getattr(vfa, 'learning_rate', 0.0),
            "epsilon": getattr(vfa, 'epsilon', 0.0),
        }
        
        # Get TD error history if available
        if hasattr(vfa, 'td_error_history'):
            metrics["td_error_history"] = vfa.td_error_history[-100:]  # Last 100
            metrics["avg_td_error"] = sum(vfa.td_error_history[-20:]) / min(20, len(vfa.td_error_history)) if vfa.td_error_history else 0.0
        
        # Get feature weights if available
        if hasattr(vfa, 'weights') and vfa.weights is not None:
            # Convert numpy array to dict if needed
            if hasattr(vfa.weights, 'shape'):
                metrics["num_features"] = vfa.weights.shape[0]
                metrics["weight_norm"] = float(sum(w**2 for w in vfa.weights)**0.5)
            else:
                metrics["feature_weights"] = dict(vfa.weights) if isinstance(vfa.weights, dict) else {}
        
        # Get convergence metrics
        if hasattr(vfa, 'convergence_score'):
            metrics["convergence_score"] = vfa.convergence_score
        
        # Performance history
        if hasattr(vfa, 'performance_history'):
            metrics["performance_history"] = vfa.performance_history[-50:]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}", exc_info=True)
        return {
            "available": False,
            "error": str(e)
        }


@app.get("/orders")
async def get_orders(status: Optional[str] = None, limit: int = 50):
    """Get orders with optional status filter"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        current_state = state_manager.get_current_state()
        
        shipments = current_state.pending_shipments
        
        # Filter by status if provided
        if status:
            try:
                status_enum = ShipmentStatus(status.lower())
                shipments = [s for s in shipments if s.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Limit results
        shipments = shipments[:limit]
        
        return {
            "count": len(shipments),
            "orders": [
                {
                    "order_id": s.id,
                    "customer_id": s.customer_id,
                    "origin": s.origin.formatted_address,
                    "destinations": [d.formatted_address for d in s.destinations],
                    "weight": s.weight,
                    "volume": s.volume,
                    "priority": s.priority,
                    "status": s.status.value,
                    "created_at": s.creation_time.isoformat(),
                    "deadline": s.deadline.isoformat(),
                    "time_to_deadline_hours": s.time_to_deadline(current_state.timestamp).total_seconds() / 3600
                }
                for s in shipments
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orders")
async def create_order(order: OrderIngest):
    """Create a new order (alias for /orders/ingest for compatibility)"""
    return await ingest_order(order)


@app.post("/decisions/route")
async def get_routing_decision(request: dict):
    """
    Get routing decision for a specific order
    This triggers a decision cycle focused on the given order
    """
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        order_id = request.get("order_id")
        context = request.get("context", {})
        
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id is required")
        
        # Get current state
        current_state = state_manager.get_current_state()
        
        # Find the shipment
        shipment = None
        for s in current_state.pending_shipments:
            if s.id == order_id:
                shipment = s
                break
        
        if not shipment:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        
        # Trigger a decision cycle (this will use the meta-controller)
        result = decision_engine.run_cycle()
        
        return {
            "order_id": order_id,
            "decision": result.decision.action_type,
            "function_class": result.decision.function_class.value,
            "reasoning": result.decision.reasoning,
            "confidence": result.decision.confidence,
            "vehicle_assigned": result.decision.action_details.get("vehicle_id") if result.decision.action_details else None,
            "estimated_dispatch_time": result.decision.action_details.get("dispatch_time") if result.decision.action_details else None,
            "context_used": context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get routing decision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
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
# ADD THIS ENDPOINT TO main.py FOR DEBUGGING

@app.get("/debug/cfa")
async def debug_cfa_optimization():
    """
    Debug endpoint to see what CFA is actually doing
    """
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Get current state
        current_state = state_manager.get_current_state()
        
        # Get CFA and run optimization directly
        from src.core.cfa import CostFunctionApproximator
        cfa = CostFunctionApproximator()
        
        # Run optimization
        solution = cfa.optimize(current_state, value_function=None)
        
        # Return detailed debug info
        return {
            "status": solution.status,
            "reasoning": solution.reasoning,
            "num_batches": len(solution.batches),
            "total_cost": solution.total_cost,
            "avg_utilization": solution.avg_utilization,
            "solver_time": solution.solver_time,
            "state_info": {
                "pending_shipments": len(current_state.pending_shipments),
                "available_vehicles": len(current_state.get_available_vehicles()),
                "shipment_details": [
                    {
                        "id": s.id,
                        "weight": s.weight,
                        "volume": s.volume,
                        "origin": s.origin.formatted_address,
                        "destinations": [d.formatted_address for d in s.destinations]
                    }
                    for s in current_state.pending_shipments[:5]  # First 5 for brevity
                ],
                "vehicle_details": [
                    {
                        "id": v.id,
                        "type": v.vehicle_type,
                        "capacity_volume": v.capacity.volume,
                        "capacity_weight": v.capacity.weight,
                        "status": v.status.value
                    }
                    for v in current_state.fleet_state
                ]
            },
            "batches": [
                {
                    "id": b.id,
                    "num_shipments": len(b.shipments),
                    "vehicle_id": b.vehicle.id,
                    "utilization": b.utilization,
                    "total_cost": b.total_cost,
                    "shipment_ids": [s.id for s in b.shipments]
                }
                for b in solution.batches
            ] if solution.batches else []
        }
        
    except Exception as e:
        logger.error(f"CFA debug failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
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
