# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Senga SDE Framework** is an AI-powered Sequential Decision Engine for African freight logistics implementing Warren Powell's framework for sequential decision-making under uncertainty. The system provides real-time freight consolidation, route optimization, and adaptive learning for logistics operations.

## Core Architecture

### Powell's Four Function Approximators

The system implements Powell's unified framework with four complementary function classes that work together:

1. **PFA (Policy Function Approximator)** - `src/core/pfa.py`
   - Fast, rule-based decisions for emergencies and simple cases
   - Uses policy gradient learning with feature-based state-action representation
   - Triggers on: emergencies (deadline < 2hrs), simple states, high confidence scenarios

2. **CFA (Cost Function Approximator)** - `src/core/cfa.py`
   - MIP-based batch formation and optimization using OR-Tools
   - Balances utilization, timeliness, and cost constraints
   - Primary workhorse for medium complexity decisions

3. **VFA (Value Function Approximator)** - `src/core/vfa_neural.py`
   - Neural network-based value estimation (not decision maker)
   - Provides guidance to CFA and baseline for PFA
   - Learns via TD(λ) from actual route outcomes

4. **DLA (Direct Lookahead Approximator)** - `src/core/dla.py`
   - Multi-step lookahead for high-stakes/complex decisions
   - Evaluates candidate actions by simulating future outcomes
   - Uses VFA for downstream state valuation

### Meta-Controller Coordination

**`src/core/meta_controller.py`** orchestrates function class selection:

Decision flow:
```
1. Emergency? → PFA
2. Simple state? → PFA
3. High stakes/complexity? → DLA
4. Default → CFA with VFA guidance
```

### State Management

**`src/core/state_manager.py`** provides centralized state tracking with SQLite persistence:

- **SystemState**: Complete state space S_t (shipments, vehicles, routes)
- **Shipment**: Individual orders with multi-destination support
- **VehicleState**: Fleet status, capacity, location, availability
- **Route**: Active route execution tracking
- **DecisionEvent**: Decision history for learning

Database schema supports:
- Offline-first operation with sync capability
- State transitions for TD learning
- Route outcomes for learning feedback
- Learning update tracking

### Decision Engine

**`src/core/decision_engine.py`** implements the closed learning loop:

Cycle execution:
```
1. Observe state S_t
2. Estimate V(S_t) using VFA
3. Make decision a_t via meta-controller
4. Execute action, observe S_{t+1}
5. Calculate reward r_t
6. Update VFA: θ ← θ + α * δ_t
   where δ_t = r_t + γ*V(S_{t+1}) - V(S_t)
```

Critical methods:
- `run_cycle()`: Single decision cycle with complete learning
- `_execute_decision()`: Batch dispatch with state persistence
- `_trigger_learning_update()`: TD learning update
- `_dispatch_batch()`: Route creation and vehicle assignment

### Multi-Scale Learning

**`src/core/multi_scale_coordinator.py`** coordinates tactical and strategic learning:

- **Tactical**: Batch-level VFA updates from route outcomes (Week 5)
- **Strategic**: Weekly pattern mining and policy refinement
- Integrates completed route feedback for continuous improvement

## API Architecture

**`src/api/main.py`** provides FastAPI REST interface:

Key endpoints:
- `POST /orders`: Create new order (persisted to database)
- `GET /orders/pending`: Retrieve pending orders
- `POST /decisions/consolidation-cycle`: Trigger decision cycle
- `POST /route/complete`: Report route outcome (triggers learning)
- `GET /metrics/performance`: System performance metrics
- `GET /learning/vfa-metrics`: VFA learning statistics

**`src/api/adapters.py`** handles format conversion:
- `OrderAdapter`: API format ↔ Shipment dataclass
- `VehicleAdapter`: API format ↔ VehicleState dataclass

## Configuration

**Dual Configuration System:**

1. **Business Config** (`src/config/business_config.py`):
   - SQLite database for runtime-editable parameters
   - Updated via UI/API (min utilization, SLA hours, costs)
   - Changes persist and take effect immediately

2. **Model Config** (`src/config/model_config.py`):
   - YAML-based for model hyperparameters
   - Learning rates, solver time limits, exploration params
   - Environment-specific (dev/staging/production)

**`src/config/senga_config.py`** provides unified interface:
```python
config = SengaConfigurator()
config.get('min_utilization_threshold')  # → business config
config.get('vfa.learning.learning_rate')  # → model config
```

## Development Workflow

### Running the System

**Start Backend API:**
```bash
# Activate virtual environment (if using venv)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix

# Start FastAPI server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# API documentation available at http://localhost:8000/docs
```

**Initialize Demo Fleet:**
```bash
python init_demo.py
```

**Run Tests:**
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_integration.py

# Run with verbose output
pytest -v tests/

# Run single test
pytest tests/test_meta_controller.py::test_meta_controller_decides
```

### Data Persistence

All state persisted in SQLite:
- **Database**: `data/senga_state.db`
- **Logs**: `logs/senga_sde.log`

Database tables:
- `shipments`: Order tracking with status
- `routes`: Active and completed routes
- `vehicle_states`: Fleet status
- `decision_log`: Decision history
- `state_transitions`: TD learning data
- `route_outcomes`: Completed route feedback
- `learning_updates`: VFA update tracking

### Testing Strategy

**Test Files:**
- `tests/test_integration.py`: End-to-end integration tests
- `tests/test_meta_controller.py`: Meta-controller decision routing
- `tests/test_week1_foundation.py`: Core state management tests
- `tests/test_week2_routing.py`: Routing and optimization tests

**Common Test Patterns:**
```python
# Create test state
state = create_test_state(num_shipments=5, num_vehicles=2)

# Run decision cycle
result = decision_engine.run_cycle()

# Verify state transitions
assert state_manager.get_shipment(shipment_id).status == ShipmentStatus.EN_ROUTE
```

## Important Implementation Details

### Batch Format Conversion

PFA returns simple format; CFA/DLA return batch format. Meta-controller normalizes:
```python
# PFA format (old)
{
    'shipments': ['S1', 'S2'],
    'vehicle': 'VEH001'
}

# Standardized batch format
{
    'type': 'DISPATCH',
    'batches': [{
        'id': 'batch_xyz',
        'shipments': ['S1', 'S2'],
        'vehicle': 'VEH001',
        'estimated_cost': 5000.0,
        'utilization': 0.8
    }]
}
```

### State Update Flow

When dispatching batches in `_dispatch_batch()`:
1. Update shipment statuses → EN_ROUTE
2. Update vehicle status → EN_ROUTE with route assignment
3. Create Route record with sequence
4. All updates persisted to SQLite immediately

### Learning Feedback Loop

Route completion triggers tactical learning:
1. Driver app reports completion via `POST /route/complete`
2. StateManager saves RouteOutcome
3. MultiScaleCoordinator processes outcome
4. VFA updated with actual vs predicted values
5. TD error logged for monitoring

### Feature Engineering

VFA expects state features:
```python
{
    'pending_count': int,
    'fleet_available': int,
    'avg_urgency': float,
    'urgent_count': int,
    'total_volume': float,
    'total_weight': float,
    'fleet_utilization': float
}
```

## Common Pitfalls

1. **Missing batch 'id' key**: Always include `id` in batch dictionaries
2. **Cache invalidation**: Call `state_manager._invalidate_cache()` after state updates
3. **Action type mismatches**: Use 'DISPATCH', 'WAIT', 'DISPATCH_IMMEDIATE' (not 'EMERGENCY_DISPATCH')
4. **Database locks**: StateManager uses `check_same_thread=False` but be cautious with concurrent writes
5. **VFA signature**: `vfa.update(state, actual_outcome)` not `update(state, action, reward, next_state)`

## Dependencies

Core requirements (Python 3.10-3.13):
- **FastAPI 0.109.0**: REST API framework
- **OR-Tools 9.7.2996**: MIP and VRP optimization
- **SQLAlchemy 2.0.25**: Database ORM
- **NumPy 1.26.3**: Numerical computation
- **Pandas 2.1.4**: Data manipulation
- **Streamlit 1.29.0**: UI framework (optional)

See `requirements.txt` for complete list.

## Logging

Structured logging throughout:
```python
import logging
logger = logging.getLogger(__name__)

logger.info("High-level flow information")
logger.debug("Detailed diagnostic information")
logger.warning("Recoverable issues")
logger.error("Serious problems", exc_info=True)
```

Logs written to:
- Console (stdout)
- File: `logs/senga_sde.log`

## Production Considerations

- Database path configurable via `StateManager(db_path="...")`
- CORS configured for all origins (restrict in production)
- No authentication implemented (add for production)
- OR-Tools protobuf pinned to 4.25.3 (avoid conflicts with Streamlit)
- Connection pooling not configured (add for high concurrency)

## Key Mathematical Concepts

**TD(λ) Learning:**
```
δ_t = r_t + γ*V(s_{t+1}) - V(s_t)  # TD error
θ ← θ + α * δ_t * ∇V(s_t)          # Weight update
```

**Policy Gradient (PFA):**
```
∇J(θ) = E[∇log π(a|s;θ) * (R - b)]  # b = VFA baseline
```

**Reward Structure:**
```
R = utilization_bonus + on_time_bonus
    - operational_cost - delay_penalties
```

## Extension Points

To add new functionality:

1. **New Function Class**: Implement in `src/core/`, integrate in MetaController
2. **New Decision Feature**: Add to `_extract_state_features()` in decision_engine.py
3. **New Reward Component**: Extend `RewardCalculator` in `src/core/reward_calculator.py`
4. **New API Endpoint**: Add to `src/api/main.py`, create adapter if needed
5. **New Configuration**: Add to appropriate config class with database migration

## File Structure Reference

```
src/
├── core/              # Decision-making components
│   ├── decision_engine.py      # Main orchestration loop
│   ├── meta_controller.py      # Function class coordinator
│   ├── state_manager.py        # State tracking & persistence
│   ├── pfa.py                  # Policy function approximator
│   ├── cfa.py                  # Cost function approximator
│   ├── vfa_neural.py           # Value function approximator
│   ├── dla.py                  # Direct lookahead approximator
│   ├── reward_calculator.py    # Reward computation
│   └── multi_scale_coordinator.py  # Learning coordinator
├── api/               # REST API layer
│   ├── main.py                 # FastAPI application
│   └── adapters.py             # Format conversion
├── config/            # Configuration management
│   ├── senga_config.py         # Unified config interface
│   ├── business_config.py      # Runtime parameters
│   └── model_config.py         # Model hyperparameters
├── algorithms/        # Optimization algorithms
│   └── route_optimizer.py      # VRP solving
├── utils/             # Utility functions
│   └── distance_calculator.py  # Geospatial utilities
└── integrations/      # External system integrations
    └── external_systems.py     # API clients

tests/                 # Test suite
data/                  # SQLite database storage
logs/                  # Application logs
config/                # YAML configuration files
```
