# src/api/config_api.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..config.senga_config import SengaConfigurator

app = FastAPI(title="Senga Configuration API")

# Get singleton instance
def get_config():
    return SengaConfigurator()

# ============= Request/Response Models =============

class UpdateBusinessParamRequest(BaseModel):
    key: str
    value: Any
    updated_by: str = "api_user"
    
    @validator('value')
    def validate_value(cls, v, values):
        # Additional validation can be added here
        return v

class BusinessParamResponse(BaseModel):
    key: str
    value: Any
    value_type: str
    description: str
    min_value: Optional[float]
    max_value: Optional[float]
    updated_at: Optional[datetime]
    updated_by: Optional[str]

class VehicleRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str
    capacity_volume_m3: float
    capacity_weight_kg: float
    cost_per_km: float
    fixed_cost_per_trip: float
    home_location_place_id: Optional[str] = None
    
    @validator('capacity_volume_m3', 'capacity_weight_kg')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Capacity must be positive')
        return v

class VehicleUpdateRequest(BaseModel):
    vehicle_type: Optional[str] = None
    capacity_volume_m3: Optional[float] = None
    capacity_weight_kg: Optional[float] = None
    cost_per_km: Optional[float] = None
    fixed_cost_per_trip: Optional[float] = None
    is_active: Optional[bool] = None
    home_location_place_id: Optional[str] = None

class CustomerPriorityRequest(BaseModel):
    customer_id: str
    priority_tier: int
    priority_multiplier: Optional[float] = None
    
    @validator('priority_tier')
    def validate_tier(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('Priority tier must be 1, 2, or 3')
        return v

class ZoneConfigRequest(BaseModel):
    zone_id: str
    zone_name: str
    delivery_difficulty_factor: float = 1.0
    traffic_multiplier_peak: float = 1.5
    traffic_multiplier_offpeak: float = 1.0
    peak_hours_start: str = "07:00"
    peak_hours_end: str = "10:00"
    notes: Optional[str] = None

# ============= Business Configuration Endpoints =============

@app.get("/config/business/all", response_model=Dict[str, BusinessParamResponse])
async def get_all_business_params(config: SengaConfigurator = Depends(get_config)):
    """Get all business configuration parameters"""
    params = config.business.get_all_parameters()
    
    return {
        key: BusinessParamResponse(
            key=param.key,
            value=param.value,
            value_type=param.value_type.value,
            description=param.description,
            min_value=param.min_value,
            max_value=param.max_value,
            updated_at=param.updated_at,
            updated_by=param.updated_by
        )
        for key, param in params.items()
    }

@app.get("/config/business/{key}", response_model=BusinessParamResponse)
async def get_business_param(key: str, config: SengaConfigurator = Depends(get_config)):
    """Get specific business parameter"""
    param = config.business.get_parameter(key)
    if not param:
        raise HTTPException(status_code=404, detail=f"Parameter {key} not found")
    
    return BusinessParamResponse(
        key=param.key,
        value=param.value,
        value_type=param.value_type.value,
        description=param.description,
        min_value=param.min_value,
        max_value=param.max_value,
        updated_at=param.updated_at,
        updated_by=param.updated_by
    )

@app.put("/config/business/{key}")
async def update_business_param(
    key: str,
    request: UpdateBusinessParamRequest,
    config: SengaConfigurator = Depends(get_config)
):
    """Update business parameter"""
    try:
        success = config.set_business_param(key, request.value, request.updated_by)
        if success:
            return {"status": "success", "key": key, "new_value": request.value}
        else:
            raise HTTPException(status_code=400, detail="Update failed")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/business/reload")
async def reload_business_config(config: SengaConfigurator = Depends(get_config)):
    """Hot reload business configuration from database"""
    config.business.reload()
    return {"status": "success", "message": "Business configuration reloaded"}

# ============= Fleet Configuration Endpoints =============

@app.get("/config/fleet")
async def get_fleet(config: SengaConfigurator = Depends(get_config)):
    """Get all active vehicles"""
    return {"fleet": config.fleet}

@app.post("/config/fleet/vehicle")
async def add_vehicle(
    request: VehicleRequest,
    config: SengaConfigurator = Depends(get_config)
):
    """Add new vehicle to fleet"""
    try:
        success = config.business.add_vehicle(
            vehicle_id=request.vehicle_id,
            vehicle_type=request.vehicle_type,
            capacity_volume=request.capacity_volume_m3,
            capacity_weight=request.capacity_weight_kg,
            cost_per_km=request.cost_per_km,
            fixed_cost=request.fixed_cost_per_trip,
            home_location=request.home_location_place_id
        )
        if success:
            return {"status": "success", "vehicle_id": request.vehicle_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/config/fleet/vehicle/{vehicle_id}")
async def update_vehicle(
    vehicle_id: str,
    request: VehicleUpdateRequest,
    config: SengaConfigurator = Depends(get_config)
):
    """Update vehicle configuration"""
    try:
        # Filter out None values
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        success = config.business.update_vehicle(vehicle_id, **updates)
        if success:
            return {"status": "success", "vehicle_id": vehicle_id, "updates": updates}
        else:
            raise HTTPException(status_code=404, detail="Vehicle not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/config/fleet/vehicle/{vehicle_id}")
async def deactivate_vehicle(
    vehicle_id: str,
    config: SengaConfigurator = Depends(get_config)
):
    """Deactivate vehicle (soft delete)"""
    try:
        success = config.business.update_vehicle(vehicle_id, is_active=False)
        if success:
            return {"status": "success", "vehicle_id": vehicle_id, "action": "deactivated"}
        else:
            raise HTTPException(status_code=404, detail="Vehicle not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============= Customer Configuration Endpoints =============

@app.get("/config/customer/{customer_id}")
async def get_customer_config(
    customer_id: str,
    config: SengaConfigurator = Depends(get_config)
):
    """Get customer-specific configuration"""
    customer_config = config.business.get_customer_config(customer_id)
    if not customer_config:
        return {
            "customer_id": customer_id,
            "priority_tier": 1,
            "priority_multiplier": 1.0,
            "custom_sla_hours": None,
            "message": "Using default configuration"
        }
    return customer_config

@app.put("/config/customer/priority")
async def set_customer_priority(
    request: CustomerPriorityRequest,
    config: SengaConfigurator = Depends(get_config)
):
    """Set customer priority tier"""
    try:
        success = config.business.set_customer_priority(
            customer_id=request.customer_id,
            tier=request.priority_tier,
            multiplier=request.priority_multiplier
        )
        if success:
            return {
                "status": "success",
                "customer_id": request.customer_id,
                "priority_tier": request.priority_tier
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============= Zone Configuration Endpoints =============

@app.get("/config/zone/{zone_id}")
async def get_zone_config(
    zone_id: str,
    config: SengaConfigurator = Depends(get_config)
):
    """Get zone-specific configuration"""
    zone_config = config.business.get_zone_config(zone_id)
    if not zone_config:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    return zone_config

# ============= Model Configuration Endpoints (Read-Only) =============

@app.get("/config/model/vfa")
async def get_vfa_config(config: SengaConfigurator = Depends(get_config)):
    """Get VFA model parameters (read-only)"""
    return config.model.get_section('vfa')

@app.get("/config/model/cfa")
async def get_cfa_config(config: SengaConfigurator = Depends(get_config)):
    """Get CFA model parameters (read-only)"""
    return config.model.get_section('cfa')

@app.get("/config/model/dla")
async def get_dla_config(config: SengaConfigurator = Depends(get_config)):
    """Get DLA model parameters (read-only)"""
    return config.model.get_section('dla')

@app.get("/config/model/all")
async def get_all_model_config(config: SengaConfigurator = Depends(get_config)):
    """Get all model parameters (read-only)"""
    return {
        'vfa': config.model.config.vfa,
        'cfa': config.model.config.cfa,
        'dla': config.model.config.dla,
        'pfa': config.model.config.pfa,
        'learning': config.model.config.learning,
        'coordination': config.model.config.coordination,
        'system': config.model.config.system,
        'integration': config.model.config.integration
    }

@app.get("/config/model/export")
async def export_model_config(config: SengaConfigurator = Depends(get_config)):
    """Export model configuration as JSON"""
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        config.model.export_json(f.name)
        with open(f.name, 'r') as exported:
            return json.load(exported)

# ============= Unified Configuration Endpoints =============

@app.get("/config/summary")
async def get_config_summary(config: SengaConfigurator = Depends(get_config)):
    """Get summary of key configuration parameters"""
    return {
        "business": {
            "min_utilization": config.min_utilization,
            "sla_hours": config.sla_hours,
            "cost_per_km": config.cost_per_km,
            "emergency_threshold_hours": config.emergency_threshold_hours,
            "active_vehicles": len(config.fleet)
        },
        "model": {
            "vfa_learning_rate": config.vfa_learning_rate,
            "cfa_solver_time_limit": config.cfa_solver_time_limit,
            "dla_lookahead_horizon": config.dla_lookahead_horizon
        }
    }

@app.post("/config/reload/all")
async def reload_all_config(config: SengaConfigurator = Depends(get_config)):
    """Reload both business and model configurations"""
    config.reload_all()
    return {
        "status": "success",
        "message": "All configurations reloaded",
        "timestamp": datetime.now()
    }

# ============= Health Check =============

@app.get("/health")
async def health_check():
    """API health check"""
    return {"status": "healthy", "service": "senga-config-api"}