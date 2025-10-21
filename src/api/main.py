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
from ..core.decision_engine import DecisionEngine
from ..core.state_manager import StateManager, ShipmentStatus
from ..config.senga_config import SengaConfigurator

# Import the new adapter
from .adapters import OrderAdapter, VehicleAdapter

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'senga_sde.log'),
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
    pickup_location: Dict[str, Any] = Field(..., description="Pickup location with address, latitude, longitude")
    delivery_location: Dict[str, Any] = Field(..., description="Delivery location with address, latitude, longitude")
    package_weight: float = Field(..., gt=0, description="Package weight in kg")
    volume_m3: Optional[float] = Field(None, description="Package volume in cubic meters")
    priority: str = Field("standard", description="Priority: standard, urgent, or emergency")
    created_at: Optional[str] = Field(None, description="Creation timestamp (ISO format)")
    customer_id: Optional[str] = Field(None, description="Existing customer ID")
    order_id: Optional[str] = Field(None, description="Specific order ID (auto-generated if not provided)")

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
    force_dispatch: bool = Field(False, description="Force dispatch even if utilization is low")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (traffic, weather, etc.)")

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
        logger.info("✓ Configuration loaded")
        
        # Initialize state manager (with database persistence)
        state_manager = StateManager()
        logger.info("✓ State Manager initialized with persistent storage")
        
        # Initialize decision engine
        decision_engine = DecisionEngine()
        logger.info("✓ Decision Engine initialized")
        
        # Load existing pending orders count
        current_state = state_manager.get_current_state()
        logger.info(f"✓ Loaded {len(current_state.pending_shipments)} pending orders from database")
        
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
            "config": config is not None
        }
    }

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status"""
    if not state_manager or not decision_engine:
        raise HTTPException(status_code=503, detail="System not fully initialized")
    
    current_state = state_manager.get_current_state()
    
    # Count available vehicles
    available_vehicles = sum(1 for v in current_state.fleet_state if v.is_available(datetime.now()))
    
    return SystemStatusResponse(
        status="operational",
        pending_orders=len(current_state.pending_shipments),
        available_vehicles=available_vehicles,
        active_routes=len(current_state.active_routes),
        timestamp=datetime.now().isoformat()
    )

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
            raise HTTPException(status_code=500, detail="Failed to save order to database")
        
        logger.info(f"✓ Order {shipment.id} created and persisted to database")
        
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
        
        logger.info(f"✓ Retrieved {len(orders_api_format)} pending orders from database")
        
        return [OrderResponse(**order) for order in orders_api_format]
        
    except Exception as e:
        logger.error(f"Failed to retrieve pending orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve orders: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve order: {str(e)}")

# ============= Decision Engine Endpoints =============

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
        
        logger.info(f"Running consolidation cycle with {len(current_state.pending_shipments)} pending orders")
        
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
            if updated_shipment and updated_shipment.status in [ShipmentStatus.EN_ROUTE, ShipmentStatus.BATCHED]:
                dispatched_orders.append(OrderAdapter.from_shipment_to_api(updated_shipment))
            else:
                waiting_orders.append(OrderAdapter.from_shipment_to_api(shipment))
        
        # Build batch information from decision
        dispatched_batches = []
        if cycle_result.decision.action_type == 'DISPATCH':
            batches = cycle_result.decision.action_details.get('batches', [])
            for batch in batches:
                batch_info = {
                    'vehicle_id': batch.get('vehicle'),
                    'shipments': batch.get('shipments', []),
                    'route': batch.get('route', []),
                    'estimated_distance_km': batch.get('distance', 0),
                    'estimated_duration_hours': batch.get('duration', 0)
                }
                dispatched_batches.append(batch_info)
        
        logger.info(f"✓ Consolidation cycle complete: {len(dispatched_orders)} dispatched, {len(waiting_orders)} waiting")
        
        return ConsolidationCycleResponse(
            timestamp=datetime.now().isoformat(),
            total_pending_orders=len(current_state.pending_shipments),
            orders_dispatched=len(dispatched_orders),
            orders_waiting=len(waiting_orders),
            batches_created=len(dispatched_batches),
            function_class_used=cycle_result.decision.function_class.value,
            reasoning=cycle_result.decision.reasoning,
            dispatched_batches=dispatched_batches,
            waiting_orders=waiting_orders
        )
        
    except Exception as e:
        logger.error(f"Consolidation cycle failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Consolidation cycle failed: {str(e)}")

# ============= Demo and Testing Endpoints =============

@app.post("/demo/initialize")
async def initialize_demo_data():
    """Initialize demo data for testing"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="State manager not initialized")
    
    try:
        # This endpoint can load pre-configured demo orders
        # For now, return success
        return {
            "status": "success",
            "message": "Demo initialization complete"
        }
        
    except Exception as e:
        logger.error(f"Demo initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============= Run Server =============

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )