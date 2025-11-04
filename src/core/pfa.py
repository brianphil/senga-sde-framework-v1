# src/core/pfa.py
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PFAAction:
    action_type: str
    shipment_ids: List[str]
    reasoning: str
    confidence: float


class PolicyFunctionApproximator:
    def __init__(self):
        self.queue_threshold = 3.0
        self.urgency_threshold = 0.7
        self.time_window_threshold = 2.0
        self.consolidation_threshold = 0.6
        logger.info("PFA initialized")

    def select_action(self, state) -> PFAAction:
        # CORRECT: state.pending_shipments is a List
        pending = state.pending_shipments

        if not pending:
            return PFAAction("wait", [], "No pending shipments", 1.0)

        # Rule 1: Critical urgency
        critical = []
        for s in pending:
            hours_to_deadline = (s.deadline - state.timestamp).total_seconds() / 3600
            if hours_to_deadline < self.time_window_threshold:
                critical.append(s)

        if critical:
            return PFAAction(
                "dispatch_batch" if len(critical) > 1 else "dispatch_single",
                [s.id for s in critical],
                f"Emergency: {len(critical)} critical shipments",
                0.95,
            )

        # Rule 2: Queue size trigger
        if len(pending) >= self.queue_threshold:
            batch = self._find_batch(pending)
            if batch:
                return PFAAction(
                    "dispatch_batch",
                    [s.id for s in batch],
                    f"Queue size {len(pending)} >= {self.queue_threshold}",
                    0.8,
                )

        # Rule 3: Consolidation opportunity
        batch = self._find_batch(pending)
        if batch and len(batch) >= 2:
            avg_urgency = np.mean([self._urgency(s, state.timestamp) for s in batch])
            if avg_urgency < self.urgency_threshold:
                return PFAAction(
                    "dispatch_batch",
                    [s.id for s in batch],
                    f"Good consolidation: {len(batch)} shipments",
                    0.75,
                )

        return PFAAction("wait", [], "Waiting for consolidation", 0.6)

    def _find_batch(self, shipments: List) -> Optional[List]:
        if len(shipments) < 2:
            return None

        best_batch, best_score = None, 0
        for i, s1 in enumerate(shipments):
            for s2 in shipments[i + 1 :]:
                sim = self._similarity(s1, s2)
                if sim > self.consolidation_threshold and sim > best_score:
                    best_batch, best_score = [s1, s2], sim

        return best_batch

    import numpy as np

    def _similarity(self, s1, s2, origin_lat=-1.286389, origin_lng=36.817223) -> float:
        """
        Compute geographic similarity between s1 and s2 based on:
        1. Haversine distance between their first destinations (accuracy over Earth's curvature)
        2. Direction similarity from a common origin (e.g., Nairobi)

        Returns a score between 0.0 and 1.0
        """

        # --- safety check ---
        if not s1.destinations or not s2.destinations:
            return 0.0

        d1, d2 = s1.destinations[0], s2.destinations[0]
        lat1, lng1 = np.radians(d1.lat), np.radians(d1.lng)
        lat2, lng2 = np.radians(d2.lat), np.radians(d2.lng)
        origin_lat, origin_lng = np.radians(origin_lat), np.radians(origin_lng)

        # --- helper functions ---
        def haversine(lat1, lng1, lat2, lng2):
            R = 6371.0  # Earth radius (km)
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
            )
            return R * 2 * np.arcsin(np.sqrt(a))

        def bearing(lat1, lng1, lat2, lng2):
            dlon = lng2 - lng1
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            return np.arctan2(y, x)

        # --- compute accurate distance ---
        dist = haversine(lat1, lng1, lat2, lng2)

        # --- compute bearings from origin to each destination ---
        b1 = bearing(origin_lat, origin_lng, lat1, lng1)
        b2 = bearing(origin_lat, origin_lng, lat2, lng2)

        # --- angular difference between directions ---
        angle_diff = np.abs(np.arctan2(np.sin(b1 - b2), np.cos(b1 - b2)))

        # --- similarity components ---
        distance_sim = np.exp(-dist / 100)  # decay by 100 km
        direction_sim = np.exp(-angle_diff / 0.5)  # decay by ~30° difference

        # --- combined similarity ---
        return float(distance_sim * direction_sim)

    def _urgency(self, shipment, current_time) -> float:
        hours = (shipment.deadline - current_time).total_seconds() / 3600
        return max(0, min(1, 1.0 - hours / 24))

    def update_from_outcome(self, action: PFAAction, reward: float):
        pass  # Simple version - no learning yet
