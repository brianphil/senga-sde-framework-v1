"""
Demo Data Loader for Senga SDE
Pre-configured realistic Kenyan logistics scenarios
"""

from datetime import datetime, timedelta
import random
from typing import List, Dict
import uuid

# Realistic Kenyan locations with actual coordinates
KENYAN_LOCATIONS = {
    # Nairobi Area
    "Nairobi CBD - Kenyatta Avenue": {
        "lat": -1.2864, "lng": 36.8172,
        "type": "commercial", "traffic_pattern": "high"
    },
    "Westlands - Waiyaki Way": {
        "lat": -1.2676, "lng": 36.8070,
        "type": "commercial", "traffic_pattern": "high"
    },
    "Industrial Area - Mombasa Road": {
        "lat": -1.3167, "lng": 36.8500,
        "type": "industrial", "traffic_pattern": "medium"
    },
    "Kilimani - Argwings Kodhek": {
        "lat": -1.2884, "lng": 36.7826,
        "type": "residential", "traffic_pattern": "medium"
    },
    "Karen - Karen Road": {
        "lat": -1.3194, "lng": 36.7000,
        "type": "residential", "traffic_pattern": "low"
    },
    
    # Nakuru
    "Nakuru Town - Kenyatta Avenue": {
        "lat": -0.3031, "lng": 36.0800,
        "type": "commercial", "traffic_pattern": "medium"
    },
    "Nakuru Industrial Area": {
        "lat": -0.2827, "lng": 36.0664,
        "type": "industrial", "traffic_pattern": "low"
    },
    
    # Eldoret
    "Eldoret Town - Uganda Road": {
        "lat": 0.5143, "lng": 35.2698,
        "type": "commercial", "traffic_pattern": "medium"
    },
    "Eldoret West": {
        "lat": 0.5287, "lng": 35.2415,
        "type": "residential", "traffic_pattern": "low"
    },
    
    # Kitale
    "Kitale Town Centre": {
        "lat": 1.0157, "lng": 35.0062,
        "type": "commercial", "traffic_pattern": "low"
    },
    
    # Thika
    "Thika Town - Kenyatta Highway": {
        "lat": -1.0332, "lng": 37.0690,
        "type": "commercial", "traffic_pattern": "high"
    },
    
    # Kiambu
    "Kiambu Town": {
        "lat": -1.1714, "lng": 36.8356,
        "type": "residential", "traffic_pattern": "medium"
    },
    
    # Mombasa Road Corridor
    "Syokimau - Mombasa Road": {
        "lat": -1.3633, "lng": 36.9500,
        "type": "residential", "traffic_pattern": "high"
    },
    "Mlolongo - Mombasa Road": {
        "lat": -1.3833, "lng": 36.9667,
        "type": "industrial", "traffic_pattern": "high"
    }
}

# Realistic customer names (Kenyan)
CUSTOMER_NAMES = [
    "John Mwangi", "Mary Wanjiru", "Peter Ochieng", "Grace Akinyi",
    "David Kamau", "Elizabeth Njeri", "James Kipchoge", "Sarah Wambui",
    "Daniel Omondi", "Lucy Nafula", "Michael Karanja", "Jane Chebet",
    "Joseph Mutua", "Rebecca Awino", "Samuel Kiprotich", "Faith Nyambura"
]

# Driver profiles with Kenyan details
DEMO_DRIVERS = [
    {
        "driver_id": "DRV001",
        "name": "Patrick Kimani",
        "phone": "+254722123456",
        "vehicle_type": "pickup_truck",
        "capacity_kg": 1000,
        "current_location": "Nairobi CBD - Kenyatta Avenue",
        "status": "available",
        "experience_level": "expert",
        "rating": 4.8
    },
    {
        "driver_id": "DRV002",
        "name": "Moses Otieno",
        "phone": "+254733234567",
        "vehicle_type": "van",
        "capacity_kg": 500,
        "current_location": "Westlands - Waiyaki Way",
        "status": "available",
        "experience_level": "intermediate",
        "rating": 4.5
    },
    {
        "driver_id": "DRV003",
        "name": "Jane Wanjiku",
        "phone": "+254711345678",
        "vehicle_type": "truck",
        "capacity_kg": 2000,
        "current_location": "Industrial Area - Mombasa Road",
        "status": "available",
        "experience_level": "expert",
        "rating": 4.9
    },
    {
        "driver_id": "DRV004",
        "name": "Benjamin Kiplagat",
        "phone": "+254700456789",
        "vehicle_type": "pickup_truck",
        "capacity_kg": 800,
        "current_location": "Nakuru Town - Kenyatta Avenue",
        "status": "available",
        "experience_level": "intermediate",
        "rating": 4.6
    },
    {
        "driver_id": "DRV005",
        "name": "Alice Nyokabi",
        "phone": "+254788567890",
        "vehicle_type": "van",
        "capacity_kg": 600,
        "current_location": "Thika Town - Kenyatta Highway",
        "status": "available",
        "experience_level": "beginner",
        "rating": 4.3
    }
]

# Realistic traffic patterns for Nairobi
TRAFFIC_PATTERNS = {
    "morning_rush": {
        "hours": [7, 8, 9, 10],
        "multiplier": 1.8,
        "description": "Heavy morning rush hour traffic"
    },
    "midday": {
        "hours": [11, 12, 13, 14, 15],
        "multiplier": 1.0,
        "description": "Normal traffic flow"
    },
    "evening_rush": {
        "hours": [16, 17, 18, 19],
        "multiplier": 2.0,
        "description": "Peak evening traffic"
    },
    "night": {
        "hours": [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6],
        "multiplier": 0.7,
        "description": "Light overnight traffic"
    }
}

def generate_demo_orders(num_orders: int = 10) -> List[Dict]:
    """Generate realistic demo orders for Kenyan routes"""
    orders = []
    locations = list(KENYAN_LOCATIONS.keys())
    
    for i in range(num_orders):
        # Select random pickup and delivery (ensure different)
        pickup_loc = random.choice(locations)
        delivery_loc = random.choice([loc for loc in locations if loc != pickup_loc])
        
        # Generate order
        order = {
            "order_id": f"ORD{str(uuid.uuid4())[:8].upper()}",
            "customer_name": random.choice(CUSTOMER_NAMES),
            "customer_phone": f"+2547{random.randint(10000000, 99999999)}",
            "pickup_location": {
                "address": pickup_loc,
                "latitude": KENYAN_LOCATIONS[pickup_loc]["lat"],
                "longitude": KENYAN_LOCATIONS[pickup_loc]["lng"]
            },
            "delivery_location": {
                "address": delivery_loc,
                "latitude": KENYAN_LOCATIONS[delivery_loc]["lat"],
                "longitude": KENYAN_LOCATIONS[delivery_loc]["lng"]
            },
            "package_weight": round(random.uniform(5.0, 500.0), 1),
            "priority": random.choice(["standard", "standard", "standard", "urgent", "emergency"]),
            "created_at": (datetime.now() - timedelta(hours=random.randint(0, 48))).isoformat(),
            "status": "pending",
            "special_instructions": generate_special_instructions(pickup_loc, delivery_loc)
        }
        
        orders.append(order)
    
    return orders

def generate_special_instructions(pickup: str, delivery: str) -> str:
    """Generate realistic African logistics special instructions"""
    instructions = [
        "Call customer 30 minutes before arrival",
        "Delivery at back entrance near the matatu stage",
        "Contact security guard first - ask for John",
        "Building has no elevator - use stairs",
        "Payment on delivery - M-PESA or cash",
        "Fragile items - handle with care",
        "Customer available between 2 PM - 6 PM only",
        "Office closes at 5 PM sharp - arrive early",
        "Near the blue water tank landmark",
        "Ask for directions at the petrol station",
        "Compound entrance is narrow - small vehicle needed",
        "Weekend delivery preferred - customer works weekdays"
    ]
    
    return random.choice(instructions)

def generate_demo_scenarios() -> List[Dict]:
    """Generate realistic operational scenarios"""
    scenarios = [
        {
            "name": "Nairobi to Nakuru - Morning Rush",
            "description": "Multiple deliveries along Nairobi-Nakuru highway during morning traffic",
            "orders": [
                {
                    "pickup": "Nairobi CBD - Kenyatta Avenue",
                    "delivery": "Nakuru Town - Kenyatta Avenue",
                    "weight": 150.0,
                    "priority": "urgent"
                },
                {
                    "pickup": "Westlands - Waiyaki Way",
                    "delivery": "Nakuru Industrial Area",
                    "weight": 200.0,
                    "priority": "standard"
                }
            ],
            "time_of_day": "morning_rush",
            "weather": "clear",
            "traffic_multiplier": 1.8
        },
        {
            "name": "Multi-City Long Haul",
            "description": "Nairobi → Nakuru → Eldoret → Kitale sequential delivery",
            "orders": [
                {
                    "pickup": "Industrial Area - Mombasa Road",
                    "delivery": "Nakuru Town - Kenyatta Avenue",
                    "weight": 300.0,
                    "priority": "standard"
                },
                {
                    "pickup": "Nakuru Town - Kenyatta Avenue",
                    "delivery": "Eldoret Town - Uganda Road",
                    "weight": 250.0,
                    "priority": "standard"
                },
                {
                    "pickup": "Eldoret Town - Uganda Road",
                    "delivery": "Kitale Town Centre",
                    "weight": 150.0,
                    "priority": "urgent"
                }
            ],
            "time_of_day": "midday",
            "weather": "clear",
            "traffic_multiplier": 1.0
        },
        {
            "name": "Emergency Delivery - Heavy Rain",
            "description": "Urgent medical supplies during Nairobi rainy season",
            "orders": [
                {
                    "pickup": "Industrial Area - Mombasa Road",
                    "delivery": "Karen - Karen Road",
                    "weight": 50.0,
                    "priority": "emergency"
                }
            ],
            "time_of_day": "evening_rush",
            "weather": "heavy_rain",
            "traffic_multiplier": 2.5
        },
        {
            "name": "Evening Rush Hour Chaos",
            "description": "Multiple deliveries during peak Nairobi traffic",
            "orders": [
                {
                    "pickup": "Nairobi CBD - Kenyatta Avenue",
                    "delivery": "Kilimani - Argwings Kodhek",
                    "weight": 75.0,
                    "priority": "urgent"
                },
                {
                    "pickup": "Westlands - Waiyaki Way",
                    "delivery": "Thika Town - Kenyatta Highway",
                    "weight": 100.0,
                    "priority": "standard"
                },
                {
                    "pickup": "Industrial Area - Mombasa Road",
                    "delivery": "Syokimau - Mombasa Road",
                    "weight": 120.0,
                    "priority": "standard"
                }
            ],
            "time_of_day": "evening_rush",
            "weather": "clear",
            "traffic_multiplier": 2.0
        },
        {
            "name": "Weekend Deliveries",
            "description": "Light traffic weekend operations",
            "orders": [
                {
                    "pickup": "Kiambu Town",
                    "delivery": "Karen - Karen Road",
                    "weight": 80.0,
                    "priority": "standard"
                },
                {
                    "pickup": "Nairobi CBD - Kenyatta Avenue",
                    "delivery": "Westlands - Waiyaki Way",
                    "weight": 60.0,
                    "priority": "standard"
                }
            ],
            "time_of_day": "midday",
            "weather": "clear",
            "traffic_multiplier": 0.6
        }
    ]
    
    return scenarios

def get_current_traffic_pattern() -> str:
    """Determine current traffic pattern based on time"""
    current_hour = datetime.now().hour
    
    for pattern_name, pattern_data in TRAFFIC_PATTERNS.items():
        if current_hour in pattern_data["hours"]:
            return pattern_name
    
    return "midday"

def generate_contextual_state() -> Dict:
    """Generate current contextual state for decision making"""
    current_pattern = get_current_traffic_pattern()
    
    return {
        "time_of_day": current_pattern,
        "traffic_level": TRAFFIC_PATTERNS[current_pattern]["multiplier"] / 2.0,
        "weather": random.choice(["clear", "clear", "clear", "rain", "heavy_rain"]),
        "available_drivers": len([d for d in DEMO_DRIVERS if d["status"] == "available"]),
        "pending_orders": random.randint(5, 20),
        "avg_delivery_time_today": random.randint(45, 120),
        "on_time_percentage": random.uniform(0.75, 0.95)
    }

# African logistics challenges for demo scenarios
AFRICAN_CHALLENGES = {
    "address_resolution": [
        "Near the big blue water tank",
        "Opposite the matatu stage",
        "Next to Mama Njeri's shop",
        "Behind the church with red roof",
        "Third building after the roundabout"
    ],
    "infrastructure": [
        "Road under construction - use alternative route",
        "Potholes on main road - expect delays",
        "Bridge closed for repairs",
        "Flooding on low-lying sections",
        "Power outage affecting traffic lights"
    ],
    "cultural_patterns": [
        "Customer prefers afternoon delivery (2-5 PM)",
        "Business closes for lunch hour (1-2 PM)",
        "Weekend delivery only - shop closed weekdays",
        "Early morning preferred (before 9 AM)",
        "Avoid delivery during prayer times"
    ],
    "payment_challenges": [
        "Cash on delivery required",
        "M-PESA payment pending confirmation",
        "Customer requests to split payment",
        "Payment via mobile money only",
        "Invoice delivery - payment next week"
    ],
    "vehicle_issues": [
        "Vehicle breakdown on highway - need replacement",
        "Fuel shortage - need refueling stop",
        "Tire puncture - repair needed",
        "Engine overheating - cooling break required",
        "Battery issue - jumpstart needed"
    ]
}

def get_random_challenge() -> Dict:
    """Get a random African logistics challenge"""
    challenge_type = random.choice(list(AFRICAN_CHALLENGES.keys()))
    challenge_description = random.choice(AFRICAN_CHALLENGES[challenge_type])
    
    return {
        "type": challenge_type,
        "description": challenge_description,
        "impact": random.choice(["low", "medium", "high"]),
        "resolution_time": random.randint(5, 60)
    }

def create_demo_initialization_data() -> Dict:
    """Create complete demo initialization package"""
    return {
        "drivers": DEMO_DRIVERS,
        "orders": generate_demo_orders(15),
        "scenarios": generate_demo_scenarios(),
        "locations": KENYAN_LOCATIONS,
        "current_context": generate_contextual_state(),
        "active_challenges": [get_random_challenge() for _ in range(3)],
        "traffic_patterns": TRAFFIC_PATTERNS
    }

if __name__ == "__main__":
    # Test data generation
    demo_data = create_demo_initialization_data()
    print(f"Generated {len(demo_data['orders'])} orders")
    print(f"Configured {len(demo_data['drivers'])} drivers")
    print(f"Created {len(demo_data['scenarios'])} scenarios")
    print("\nSample Order:")
    print(demo_data['orders'][0])