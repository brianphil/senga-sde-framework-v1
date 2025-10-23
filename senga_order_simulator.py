# senga_order_simulator.py

"""
Senga Order Simulator
Simulates orders arriving: Nairobi pickups → Upcountry deliveries
"""

import sys
sys.path.append('src')

import time
import random
from datetime import datetime, timedelta
from src.core.state_manager import StateManager, Shipment, Location, ShipmentStatus

class SengaOrderSimulator:
    def __init__(self):
        self.state_manager = StateManager()
        self.order_counter = 1
        
        # Nairobi pickups
        self.nairobi_pickups = [
            {"name": "CBD", "lat": -1.2864, "lng": 36.8172},
            {"name": "Westlands", "lat": -1.2636, "lng": 36.8053},
            {"name": "Industrial Area", "lat": -1.3229, "lng": 36.8467},
            {"name": "Eastleigh", "lat": -1.2816, "lng": 36.8469},
        ]
        
        # Destinations by lane
        self.lanes = {
            "Coastal": [
                {"name": "Mombasa", "lat": -4.0435, "lng": 39.6682, "km": 480},
                {"name": "Voi", "lat": -3.3967, "lng": 38.5564, "km": 340},
                {"name": "Malindi", "lat": -3.2175, "lng": 40.1169, "km": 550},
            ],
            "Western": [
                {"name": "Nakuru", "lat": -0.3031, "lng": 36.0800, "km": 160},
                {"name": "Eldoret", "lat": 0.5143, "lng": 35.2698, "km": 310},
                {"name": "Kisumu", "lat": -0.0917, "lng": 34.7680, "km": 340},
                {"name": "Kisii", "lat": -0.6817, "lng": 34.7673, "km": 300},
            ],
            "Central": [
                {"name": "Nyeri", "lat": -0.4197, "lng": 36.9470, "km": 150},
                {"name": "Meru", "lat": 0.0469, "lng": 37.6556, "km": 270},
                {"name": "Nanyuki", "lat": -0.0167, "lng": 37.0667, "km": 200},
            ]
        }
        
        # Lane weights (Coastal and Western more common)
        self.lane_weights = {
            "Coastal": 0.4,
            "Western": 0.45,
            "Central": 0.15
        }
    
    def create_order(self):
        """Create single order"""
        # Select lane based on weights
        lane = random.choices(
            list(self.lane_weights.keys()),
            weights=list(self.lane_weights.values())
        )[0]
        
        # Nairobi pickup
        pickup = random.choice(self.nairobi_pickups)
        
        # Upcountry destination
        dest = random.choice(self.lanes[lane])
        
        # Package properties
        weight = random.uniform(30, 600)
        volume = random.uniform(0.5, 6.0)
        is_urgent = random.random() < 0.12
        
        # Deadline based on distance
        if dest['km'] < 150:
            hours = 8 if is_urgent else 12
        elif dest['km'] < 300:
            hours = 18 if is_urgent else 24
        else:
            hours = 24 if is_urgent else 48
        
        shipment = Shipment(
            id=f"SNG{self.order_counter:06d}",
            origin=Location(
                place_id=f"pickup_{self.order_counter}",
                lat=pickup["lat"],
                lng=pickup["lng"],
                formatted_address=f"{pickup['name']}, Nairobi",
                zone_id="Nairobi"
            ),
            destinations=[Location(
                place_id=f"dest_{self.order_counter}",
                lat=dest["lat"],
                lng=dest["lng"],
                formatted_address=f"{dest['name']} ({lane} Lane)",
                zone_id=lane
            )],
            weight=weight,
            volume=volume,
            declared_value=random.uniform(15000, 80000),
            customer_id=f"CUST{random.randint(1, 200):04d}",
            created_at=datetime.now(),
            delivery_deadline=datetime.now() + timedelta(hours=hours),
            priority="high" if is_urgent else "standard",
            status=ShipmentStatus.PENDING
        )
        
        self.state_manager.add_shipment(shipment)
        self.order_counter += 1
        
        return shipment, lane, dest
    
    def run(self, min_interval=45, max_interval=300):
        """
        Run continuous simulation
        
        Args:
            min_interval: Min seconds between orders (default: 45s = ~1/min peak)
            max_interval: Max seconds (default: 300s = 5min during slow periods)
        """
        print("🚀 Senga Order Simulator")
        print("   Nairobi Pickups → Upcountry Deliveries")
        print(f"   Orders every {min_interval}-{max_interval} seconds")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                order, lane, dest = self.create_order()
                
                urgent_flag = "🚨" if order.priority == "high" else "  "
                lane_emoji = "🌊" if lane == "Coastal" else "⛰️" if lane == "Western" else "🏔️"
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {urgent_flag} {lane_emoji} {order.id}")
                print(f"   {order.origin.formatted_address} → {dest['name']} ({dest['km']}km)")
                print(f"   {order.weight:.0f}kg | {order.volume:.1f}m³ | "
                      f"Deadline: {order.delivery_deadline.strftime('%H:%M')}")
                
                # Current state
                state = self.state_manager.get_current_state()
                coastal = sum(1 for s in state.pending_shipments if s.destinations[0].zone_id == 'Coastal')
                western = sum(1 for s in state.pending_shipments if s.destinations[0].zone_id == 'Western')
                central = sum(1 for s in state.pending_shipments if s.destinations[0].zone_id == 'Central')
                
                print(f"   📊 Pending: 🌊{coastal} ⛰️{western} 🏔️{central}\n")
                
                # Wait
                wait = random.randint(min_interval, max_interval)
                time.sleep(wait)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Simulator stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-interval', type=int, default=45)
    parser.add_argument('--max-interval', type=int, default=300)
    
    args = parser.parse_args()
    
    sim = SengaOrderSimulator()
    sim.run(args.min_interval, args.max_interval)