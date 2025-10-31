import unittest
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import uuid

# ====================================================================
# MOCK CLASSES (Minimal definitions based on state_manager.py)
# ====================================================================

@dataclass
class Location:
    place_id: str
    lat: float
    lng: float
    formatted_address: str
    zone_id: Optional[str] = None

@dataclass
class Shipment:
    # Shipments need an ID and a route_plan containing at least the pickup location
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    route_plan: List[Location] = field(default_factory=list)
    # Add other attributes as needed, but not necessary for this test

# ====================================================================
# MetaController Minimal Implementation (Contains the method under test)
# ====================================================================

class MetaControllerTestHelper:
    """
    A minimal class containing ONLY the _cluster_shipments method for unit testing.
    The DBSCAN parameters (eps) are critical and must be tuned based on your region.
    eps = 0.05 degrees is approximately 5.5 km.
    """
    def _cluster_shipments(self, shipments: List[Shipment]) -> List[List[Shipment]]:
        """Uses DBSCAN to cluster shipments based on pick-up location (lat, lng)."""
        if not shipments:
            return []

        coordinates = []
        shipment_list = []
        for s in shipments:
            if s.route_plan and s.route_plan[0].location:
                loc = s.route_plan[0].location
                coordinates.append((loc.lat, loc.lng))
                shipment_list.append(s)

        if not coordinates:
            return []

        X = np.array(coordinates)

        # DBSCAN parameters:
        # eps=0.05 degrees is a good starting point for a metropolitan area clustering (~5.5km)
        # min_samples=2 ensures a cluster must have at least two points.
        clustering = DBSCAN(eps=0.05, min_samples=2, algorithm='ball_tree').fit(X)
        labels = clustering.labels_

        # 3. Consolidate results
        clusters = {}
        noise_points = []
        
        for i, label in enumerate(labels):
            shipment = shipment_list[i]
            if label != -1: # Standard cluster
                clusters.setdefault(label, []).append(shipment)
            else:           # Noise point (far from others), treat as its own cluster of size 1
                noise_points.append(shipment)

        # Final list of clusters: standard clusters + individual noise points
        final_clusters = list(clusters.values()) + [[s] for s in noise_points]
        
        return final_clusters

# ====================================================================
# UNIT TEST CLASS
# ====================================================================

class TestGeographicalClustering(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up the coordinates for the test cases."""
        # Approximate coordinates for key cities in Kenya
        cls.NAIROBI_CBD = (-1.2863, 36.8172)
        cls.NAIROBI_SUBURB = (-1.301, 36.832)  # ~2.5 km from CBD
        cls.MOMBASA_PORT = (-4.05, 39.66)      # Far from Nairobi
        cls.ELDORET_TOWN = (0.52, 35.26)       # Far from all
        
        cls.controller = MetaControllerTestHelper()

    def _create_shipment(self, lat: float, lng: float, name: str = "TestShipment") -> Shipment:
        """Helper to create a Shipment object for testing."""
        loc = Location(
            place_id=f"PID_{name}",
            lat=lat,
            lng=lng,
            formatted_address=f"{name} address"
        )
        return Shipment(
            id=f"S_{name}_{uuid.uuid4()}",
            route_plan=[loc]
        )

    # --- Test Case 1: Ensures close points are clustered (Nairobi) ---
    def test_tight_clustering(self):
        """Two shipments close together should form a single cluster."""
        shipment_a = self._create_shipment(*self.NAIROBI_CBD, name="NBO_A")
        shipment_b = self._create_shipment(*self.NAIROBI_SUBURB, name="NBO_B")
        
        clusters = self.controller._cluster_shipments([shipment_a, shipment_b])
        
        self.assertEqual(len(clusters), 1, "Should result in exactly one cluster.")
        self.assertEqual(len(clusters[0]), 2, "The single cluster should contain both shipments.")

    # --- Test Case 2: Ensures distant points are separated (Nairobi vs Mombasa) ---
    def test_geographical_separation(self):
        """Shipments far apart should be split into distinct clusters."""
        shipment_nbo = self._create_shipment(*self.NAIROBI_CBD, name="NBO")
        shipment_mba = self._create_shipment(*self.MOMBASA_PORT, name="MBA")
        
        clusters = self.controller._cluster_shipments([shipment_nbo, shipment_mba])
        
        # DBSCAN treats separated points with min_samples=2 as noise (label -1), 
        # meaning they become two separate clusters of size 1.
        self.assertEqual(len(clusters), 2, "Should result in two separate clusters (both noise).")
        self.assertTrue(all(len(c) == 1 for c in clusters), "Each cluster should be a single noise point.")

    # --- Test Case 3: Mixed Scenario (3 Clusters: NBO Group, MBA Group, Eldoret Noise) ---
    def test_mixed_clustering_scenario(self):
        """Test with a complex mix of two tight groups and one distant outlier."""
        
        # Cluster 1: Nairobi (Tight group)
        s1 = self._create_shipment(*self.NAIROBI_CBD, name="S1")
        s2 = self._create_shipment(*self.NAIROBI_SUBURB, name="S2")

        # Cluster 2: Mombasa (Tight group)
        s3 = self._create_shipment(self.MOMBASA_PORT[0] + 0.001, self.MOMBASA_PORT[1], name="S3")
        s4 = self._create_shipment(self.MOMBASA_PORT[0] - 0.001, self.MOMBASA_PORT[1], name="S4")

        # Cluster 3: Eldoret (Outlier/Noise)
        s5 = self._create_shipment(*self.ELDORET_TOWN, name="S5")
        
        shipments = [s1, s2, s3, s4, s5]
        clusters = self.controller._cluster_shipments(shipments)
        
        self.assertEqual(len(clusters), 3, "Expected 3 resulting clusters: NBO group, MBA group, and Eldoret noise.")
        
        # Check cluster sizes: Two clusters of size 2, one of size 1 (the noise point)
        sizes = sorted([len(c) for c in clusters])
        self.assertEqual(sizes, [1, 2, 2], "Cluster sizes should be [1 (Eldoret), 2 (NBO), 2 (MBA)].")
        
    # --- Test Case 4: Edge Cases ---
    def test_edge_cases(self):
        """Test with empty list, single shipment, and shipments without location data."""
        
        # 1. Empty Input
        clusters_empty = self.controller._cluster_shipments([])
        self.assertEqual(len(clusters_empty), 0, "Empty list should return an empty list of clusters.")
        
        # 2. Single Shipment
        s_single = self._create_shipment(*self.NAIROBI_CBD, name="Single")
        clusters_single = self.controller._cluster_shipments([s_single])
        self.assertEqual(len(clusters_single), 1, "Single shipment should return a cluster of size 1 (as noise).")
        self.assertEqual(len(clusters_single[0]), 1, "Single cluster should contain the shipment.")

if __name__ == '__main__':
    unittest.main()