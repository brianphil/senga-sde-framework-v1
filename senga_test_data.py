# senga_test_data.py

"""
Senga-Specific Test Data Generator
Reflects actual business model: Pickups in Nairobi → Long-distance deliveries
"""

import sys
sys.path.append('src')

import random
from datetime import datetime, timedelta
from src.core.state_manager import StateManager, Shipment, VehicleState, Location, Capacity, ShipmentStatus

class SengaDataGenerator:
    """Generate realistic test data matching Senga's operations"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.order_counter = 1
        
        # NAIROBI PICKUP LOCATIONS (where customers drop off packages)
        self.nairobi_pickups = [
            {"name": "CBD - Kenyatta Avenue", "lat": -1.2864, "lng": 36.8172},
            {"name": "Westlands - Sarit Centre", "lat": -1.2636, "lng": 36.8053},
            {"name": "Kilimani - Yaya Centre", "lat": -1.2896, "lng": 36.7823},
            {"name": "Industrial Area - Enterprise Rd", "lat": -1.3229, "lng": 36.8467},
            {"name": "Upperhill - Britam Tower", "lat": -1.2921, "lng": 36.8219},
            {"name": "Eastleigh - 1st Avenue", "lat": -1.2816, "lng": 36.8469},
            {"name": "Parklands - Limuru Rd", "lat": -1.2624, "lng": 36.8253},
        ]
        
        # COASTAL LANE (Mombasa Road corridor)
        self.coastal_destinations = [
            {"name": "Mombasa - Nyali", "lat": -4.0435, "lng": 39.7292, "distance_km": 480},
            {"name": "Mombasa - CBD", "lat": -4.0435, "lng": 39.6682, "distance_km": 480},
            {"name": "Mombasa - Likoni", "lat": -4.0832, "lng": 39.6668, "distance_km": 490},
            {"name": "Voi", "lat": -3.3967, "lng": 38.5564, "distance_km": 340},
            {"name": "Mariakani", "lat": -3.8572, "lng": 39.4748, "distance_km": 400},
            {"name": "Malindi", "lat": -3.2175, "lng": 40.1169, "distance_km": 550},
        ]
        
        # WESTERN LANE (Nakuru-Eldoret-Kisumu corridor)
        self.western_destinations = [
            {"name": "Nakuru - Town", "lat": -0.3031, "lng": 36.0800, "distance_km": 160},
            {"name": "Mai Mahiu", "lat": -0.9167, "lng": 36.4500, "distance_km": 75},
            {"name": "Naivasha", "lat": -0.7167, "lng": 36.4333, "distance_km": 90},
            {"name": "Eldoret - Town", "lat": 0.5143, "lng": 35.2698, "distance_km": 310},
            {"name": "Kitale", "lat": 1.0167, "lng": 34.9667, "distance_km": 380},
            {"name": "Kisumu - CBD", "lat": -0.0917, "lng": 34.7680, "distance_km": 340},
            {"name": "Kisii - Town", "lat": -0.6817, "lng": 34.7673, "distance_km": 300},
            {"name": "Kericho", "lat": -0.3677, "lng": 35.2839, "distance_km": 260},
        ]
        
        # CENTRAL LANE (Mount Kenya region)
        self.central_destinations = [
            {"name": "Nyeri - Town", "lat": -0.4197, "lng": 36.9470, "distance_km": 150},
            {"name": "Nanyuki", "lat": -0.0167, "lng": 37.0667, "distance_km": 200},
            {"name": "Meru - Town", "lat": 0.0469, "lng": 37.6556, "distance_km": 270},
            {"name": "Embu", "lat": -0.5312, "lng": 37.4570, "distance_km": 130},
            {"name": "Thika", "lat": -1.0332, "lng": 37.0693, "distance_km": 45},
            {"name": "Karatina", "lat": -0.4833, "lng": 37.1333, "distance_km": 120},
        ]
        
        # Senga depot in Nairobi
        self.depot = Location(
            place_id="senga_depot",
            lat=-1.2921,
            lng=36.8219,
            formatted_address="Senga Depot - Upperhill, Nairobi",
            zone_id="Nairobi-HQ"
        )
        
        # Lane definitions
        self.lanes = {
            "Coastal": self.coastal_destinations,
            "Western": self.western_destinations,
            "Central": self.central_destinations
        }
    
    def create_vehicles(self, coastal=2, western=2, central=1):
        """
        Create vehicles assigned to lanes
        
        Args:
            coastal: Number of vehicles for coastal lane
            western: Number of vehicles for western lane
            central: Number of vehicles for central lane
        """
        vehicles = []
        
        # Vehicle types for long-distance
        truck_types = [
            {"name": "10-ton truck", "capacity": Capacity(10000, 35), "cost_per_km": 65},
            {"name": "7-ton truck", "capacity": Capacity(7000, 25), "cost_per_km": 55},
            {"name": "5-ton truck", "capacity": Capacity(5000, 20), "cost_per_km": 50},
        ]
        
        vehicle_id = 1
        
        # Coastal lane vehicles
        for i in range(coastal):
            vtype = random.choice(truck_types)
            vehicle = VehicleState(
                id=f"COASTAL-{vehicle_id:02d}",
                type=f"{vtype['name']} (Coastal)",
                capacity=vtype['capacity'],
                current_location=self.depot,
                status="available",
                driver_id=f"DRV-C{vehicle_id:02d}",
                fixed_cost_per_trip=5000.0,  # Higher for long distance
                cost_per_km=vtype['cost_per_km']
            )
            self.state_manager.add_vehicle(vehicle)
            vehicles.append(vehicle)
            vehicle_id += 1
            print(f"✓ Created {vehicle.id} - {vtype['name']}")
        
        # Western lane vehicles
        for i in range(western):
            vtype = random.choice(truck_types)
            vehicle = VehicleState(
                id=f"WESTERN-{vehicle_id:02d}",
                type=f"{vtype['name']} (Western)",
                capacity=vtype['capacity'],
                current_location=self.depot,
                status="available",
                driver_id=f"DRV-W{vehicle_id:02d}",
                fixed_cost_per_trip=5000.0,
                cost_per_km=vtype['cost_per_km']
            )
            self.state_manager.add_vehicle(vehicle)
            vehicles.append(vehicle)
            vehicle_id += 1
            print(f"✓ Created {vehicle.id} - {vtype['name']}")
        
        # Central lane vehicles
        for i in range(central):
            vtype = random.choice(truck_types)
            vehicle = VehicleState(
                id=f"CENTRAL-{vehicle_id:02d}",
                type=f"{vtype['name']} (Central)",
                capacity=vtype['capacity'],
                current_location=self.depot,
                status="available",
                driver_id=f"DRV-M{vehicle_id:02d}",
                fixed_cost_per_trip=4000.0,
                cost_per_km=vtype['cost_per_km']
            )
            self.state_manager.add_vehicle(vehicle)
            vehicles.append(vehicle)
            vehicle_id += 1
            print(f"✓ Created {vehicle.id} - {vtype['name']}")
        
        return vehicles
    
    def create_order(self, lane=None):
        """
        Create single order matching Senga's model
        Pickup in Nairobi → Delivery upcountry
        
        Args:
            lane: 'Coastal', 'Western', 'Central', or None for random
        """
        # Random Nairobi pickup
        pickup = random.choice(self.nairobi_pickups)
        
        # Select lane
        if lane is None:
            lane = random.choice(list(self.lanes.keys()))
        
        # Random destination in that lane
        destinations_pool = self.lanes[lane]
        num_destinations = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        selected_dests = random.sample(destinations_pool, min(num_destinations, len(destinations_pool)))
        
        # Package properties (typical for Senga)
        weight = random.uniform(20, 800)  # 20kg to 800kg
        volume = random.uniform(0.3, 8.0)  # Small to medium packages
        is_urgent = random.random() < 0.1  # 10% urgent
        
        # Calculate realistic deadline based on distance
        avg_distance = selected_dests[0]['distance_km']
        if avg_distance < 100:
            base_hours = 6
        elif avg_distance < 200:
            base_hours = 12
        elif avg_distance < 350:
            base_hours = 24
        else:
            base_hours = 36
        
        deadline_hours = base_hours if is_urgent else base_hours * 1.5
        
        # Create shipment
        shipment = Shipment(
            id=f"SNG{self.order_counter:06d}",
            origin=Location(
                place_id=f"pickup_{self.order_counter}",
                lat=pickup["lat"],
                lng=pickup["lng"],
                formatted_address=f"{pickup['name']}, Nairobi",
                zone_id="Nairobi"
            ),
            destinations=[
                Location(
                    place_id=f"dest_{self.order_counter}_{i}",
                    lat=dest["lat"],
                    lng=dest["lng"],
                    formatted_address=f"{dest['name']} ({lane} Lane)",
                    zone_id=lane
                )
                for i, dest in enumerate(selected_dests)
            ],
            weight=weight,
            volume=volume,
            declared_value=random.uniform(10000, 100000),
            customer_id=f"CUST{random.randint(1, 200):04d}",
            created_at=datetime.now(),
            delivery_deadline=datetime.now() + timedelta(hours=deadline_hours),
            priority="high" if is_urgent else "standard",
            status=ShipmentStatus.PENDING,
            special_instructions=None
        )
        
        self.state_manager.add_shipment(shipment)
        self.order_counter += 1
        
        return shipment, lane
    
    def create_orders_batch(self, coastal=5, western=5, central=3):
        """
        Create batch of orders across lanes
        
        Args:
            coastal: Number of coastal lane orders
            western: Number of western lane orders
            central: Number of central lane orders
        """
        print("\n📦 Creating orders batch...")
        
        orders = []
        
        # Coastal orders
        print(f"\n🌊 Coastal Lane ({coastal} orders):")
        for _ in range(coastal):
            order, _ = self.create_order(lane="Coastal")
            orders.append(order)
            print(f"   {order.id}: {order.origin.formatted_address} → "
                  f"{order.destinations[0].formatted_address}")
        
        # Western orders
        print(f"\n⛰️  Western Lane ({western} orders):")
        for _ in range(western):
            order, _ = self.create_order(lane="Western")
            orders.append(order)
            print(f"   {order.id}: {order.origin.formatted_address} → "
                  f"{order.destinations[0].formatted_address}")
        
        # Central orders
        print(f"\n🏔️  Central Lane ({central} orders):")
        for _ in range(central):
            order, _ = self.create_order(lane="Central")
            orders.append(order)
            print(f"   {order.id}: {order.origin.formatted_address} → "
                  f"{order.destinations[0].formatted_address}")
        
        return orders
    
    def print_summary(self):
        """Print summary of current state"""
        state = self.state_manager.get_current_state()
        
        print("\n" + "="*70)
        print("📊 SENGA OPERATIONS SUMMARY")
        print("="*70)
        
        # Vehicles by lane
        print(f"\n🚚 Vehicles: {len(state.fleet_state)}")
        coastal_vehicles = [v for v in state.fleet_state if 'COASTAL' in v.id]
        western_vehicles = [v for v in state.fleet_state if 'WESTERN' in v.id]
        central_vehicles = [v for v in state.fleet_state if 'CENTRAL' in v.id]
        
        print(f"   🌊 Coastal Lane: {len(coastal_vehicles)} trucks")
        print(f"   ⛰️  Western Lane: {len(western_vehicles)} trucks")
        print(f"   🏔️  Central Lane: {len(central_vehicles)} trucks")
        
        # Orders by lane
        print(f"\n📦 Pending Orders: {len(state.pending_shipments)}")
        coastal_orders = [s for s in state.pending_shipments if s.destinations[0].zone_id == 'Coastal']
        western_orders = [s for s in state.pending_shipments if s.destinations[0].zone_id == 'Western']
        central_orders = [s for s in state.pending_shipments if s.destinations[0].zone_id == 'Central']
        
        print(f"   🌊 Coastal Lane: {len(coastal_orders)} orders")
        print(f"   ⛰️  Western Lane: {len(western_orders)} orders")
        print(f"   🏔️  Central Lane: {len(central_orders)} orders")
        
        if state.pending_shipments:
            total_weight = sum(s.weight for s in state.pending_shipments)
            total_volume = sum(s.volume for s in state.pending_shipments)
            urgent_orders = [s for s in state.pending_shipments if s.priority == 'high']
            
            print(f"\n   Total Weight: {total_weight:.0f} kg")
            print(f"   Total Volume: {total_volume:.1f} m³")
            print(f"   Urgent Orders: {len(urgent_orders)}")
        
        print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Senga-specific test data')
    parser.add_argument('--scenario', type=str, default='standard',
                       choices=['standard', 'coastal_heavy', 'western_heavy', 'mixed'],
                       help='Predefined scenario')
    parser.add_argument('--coastal-vehicles', type=int, default=None)
    parser.add_argument('--western-vehicles', type=int, default=None)
    parser.add_argument('--central-vehicles', type=int, default=None)
    parser.add_argument('--coastal-orders', type=int, default=None)
    parser.add_argument('--western-orders', type=int, default=None)
    parser.add_argument('--central-orders', type=int, default=None)
    
    args = parser.parse_args()
    
    generator = SengaDataGenerator()
    
    if args.scenario == 'standard':
        print("\n🚚 Standard Operations Scenario")
        generator.create_vehicles(coastal=2, western=2, central=1)
        generator.create_orders_batch(coastal=5, western=5, central=3)
    
    elif args.scenario == 'coastal_heavy':
        print("\n🌊 Coastal-Heavy Scenario")
        generator.create_vehicles(coastal=3, western=1, central=1)
        generator.create_orders_batch(coastal=10, western=3, central=2)
    
    elif args.scenario == 'western_heavy':
        print("\n⛰️  Western-Heavy Scenario")
        generator.create_vehicles(coastal=1, western=3, central=1)
        generator.create_orders_batch(coastal=3, western=10, central=2)
    
    elif args.scenario == 'mixed':
        print("\n📦 Mixed Operations Scenario")
        generator.create_vehicles(coastal=2, western=3, central=2)
        generator.create_orders_batch(coastal=7, western=8, central=5)
    
    else:
        # Custom configuration
        if args.coastal_vehicles or args.western_vehicles or args.central_vehicles:
            generator.create_vehicles(
                coastal=args.coastal_vehicles or 2,
                western=args.western_vehicles or 2,
                central=args.central_vehicles or 1
            )
        
        if args.coastal_orders or args.western_orders or args.central_orders:
            generator.create_orders_batch(
                coastal=args.coastal_orders or 5,
                western=args.western_orders or 5,
                central=args.central_orders or 3
            )
    
    generator.print_summary()
    
    print("\n✅ Test data ready!")
    print("\nNext steps:")
    print("  1. python src/api/main.py")
    print("  2. curl -X POST http://localhost:8000/autonomous/start?cycle_interval_minutes=2")
    print("  3. streamlit run operational_dashboard.py")


if __name__ == "__main__":
    main()