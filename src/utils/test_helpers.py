"""
Backend helper for testing - generates realistic route outcomes
Place in: src/utils/test_helpers.py
"""

import random
from typing import Dict, List
from datetime import datetime, timedelta

# Traffic delay probabilities by route
ROUTE_DELAYS = {
    "Nairobi-Thika": {"probability": 0.4, "avg_delay_minutes": 25},
    "Nairobi-Nakuru": {"probability": 0.3, "avg_delay_minutes": 45},
    "Nairobi-Mombasa": {"probability": 0.5, "avg_delay_minutes": 120},
    "Nairobi-Eldoret": {"probability": 0.35, "avg_delay_minutes": 60},
    "Nairobi-Kisumu": {"probability": 0.4, "avg_delay_minutes": 75},
}

# Cost factors based on distance
BASE_COST_PER_KM = 45  # KES

def generate_realistic_outcome(
    route_id: str,
    pickup: str,
    delivery: str,
    estimated_distance_km: float,
    estimated_duration_hours: float,
    num_shipments: int,
    time_of_day: int = None
) -> Dict:
    """
    Generate realistic route outcome for testing learning
    
    Returns outcome data suitable for POST /route/complete endpoint
    """
    
    # Determine if delay occurs
    route_key = f"{pickup.split()[0]}-{delivery.split()[0]}"
    delay_config = ROUTE_DELAYS.get(route_key, {"probability": 0.2, "avg_delay_minutes": 30})
    
    has_delay = random.random() < delay_config["probability"]
    delay_minutes = 0
    delays = []
    issues = []
    
    if has_delay:
        # Generate realistic delay
        delay_minutes = int(random.gauss(delay_config["avg_delay_minutes"], 15))
        delay_minutes = max(10, delay_minutes)  # Min 10 minutes
        
        delay_reasons = [
            "traffic_jam",
            "road_construction",
            "vehicle_breakdown",
            "customer_unavailable",
            "weather_conditions"
        ]
        reason = random.choice(delay_reasons)
        
        delays.append({
            "reason": reason,
            "duration_minutes": delay_minutes,
            "location": delivery
        })
        issues.append(f"{reason.replace('_', ' ').title()}: {delay_minutes}min delay")
    
    # Calculate actual vs predicted
    actual_duration = estimated_duration_hours + (delay_minutes / 60)
    
    # Traffic factor based on time of day
    if time_of_day is None:
        time_of_day = datetime.now().hour
    
    traffic_multiplier = 1.0
    if 7 <= time_of_day <= 9:  # Morning rush
        traffic_multiplier = 1.8
    elif 16 <= time_of_day <= 19:  # Evening rush
        traffic_multiplier = 2.0
    elif 20 <= time_of_day <= 6:  # Night
        traffic_multiplier = 0.7
    
    actual_distance = estimated_distance_km * (0.95 + random.random() * 0.1)  # ±5% variance
    
    # Cost calculation
    base_cost = actual_distance * BASE_COST_PER_KM
    traffic_cost = base_cost * (traffic_multiplier - 1.0) * 0.5  # 50% of excess time as cost
    delay_penalty = delay_minutes * 10 if has_delay else 0  # KES 10/minute penalty
    
    actual_cost = base_cost + traffic_cost + delay_penalty
    
    # SLA compliance (assume 4-hour window for most routes)
    sla_threshold_hours = 4.0
    sla_compliant = actual_duration <= sla_threshold_hours
    
    # Delivery success rate
    delivered = num_shipments
    if has_delay and random.random() < 0.1:  # 10% chance of failed delivery if delayed
        delivered -= 1
        issues.append("One delivery rescheduled - customer unavailable")
    
    return {
        "route_id": route_id,
        "actual_cost": round(actual_cost, 2),
        "actual_duration_hours": round(actual_duration, 2),
        "shipments_delivered": delivered,
        "total_shipments": num_shipments,
        "sla_compliant": sla_compliant,
        "delays": delays,
        "issues": issues,
        "metadata": {
            "traffic_multiplier": traffic_multiplier,
            "time_of_day": time_of_day,
            "route_type": route_key
        }
    }


def generate_batch_outcomes(batch_info: Dict) -> List[Dict]:
    """
    Generate outcomes for all routes in a dispatched batch
    """
    outcomes = []
    
    for route in batch_info.get("dispatched_batches", []):
        outcome = generate_realistic_outcome(
            route_id=route.get("route_id", f"ROUTE_{random.randint(1000, 9999)}"),
            pickup=route.get("pickup_location", "Nairobi"),
            delivery=route.get("delivery_location", "Unknown"),
            estimated_distance_km=route.get("estimated_distance_km", 100),
            estimated_duration_hours=route.get("estimated_duration_hours", 2.5),
            num_shipments=len(route.get("shipments", []))
        )
        outcomes.append(outcome)
    
    return outcomes


# Example usage for testing
if __name__ == "__main__":
    import json
    
    # Test single outcome
    outcome = generate_realistic_outcome(
        route_id="ROUTE_TEST123",
        pickup="Industrial Area",
        delivery="Nakuru",
        estimated_distance_km=160,
        estimated_duration_hours=2.5,
        num_shipments=3,
        time_of_day=17  # Evening rush
    )
    
    print("Sample Route Outcome:")
    print(json.dumps(outcome, indent=2))