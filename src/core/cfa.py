# src/core/cfa_neural.py
"""
Neural Cost Function Approximator - Learnable Consolidation
Learns batch formation from completed route outcomes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging
from datetime import datetime

from .state_manager import SystemState, Shipment, VehicleState
from .standard_types import StandardBatch

logger = logging.getLogger(__name__)


@dataclass
class CFASolution:
    """Matches existing CFA interface"""

    batches: List[StandardBatch]
    unassigned_shipments: List[Shipment]
    total_cost: float
    avg_utilization: float
    status: str
    solver_time_seconds: float
    reasoning: str


class ShipmentEncoder(nn.Module):
    """
    Maps shipment features → embedding

    Input features:
    - Volume, weight, urgency
    - Pickup/delivery location embeddings
    - Time windows
    """

    def __init__(self, input_dim=16, embed_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: [batch_size, input_dim]"""
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return self.layer_norm(h)


class BatchValueEstimator(nn.Module):
    """
    Predicts batch quality from shipment embeddings

    Outputs:
    - Expected utilization
    - Expected cost
    - Consolidation compatibility score
    """

    def __init__(self, embed_dim=32):
        super().__init__()
        # Aggregate embeddings
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        # Predict outcomes
        self.util_head = nn.Linear(32, 1)
        self.cost_head = nn.Linear(32, 1)
        self.compat_head = nn.Linear(32, 1)

    def forward(self, embeddings):
        """
        embeddings: [batch_size, num_shipments, embed_dim]
        Returns: dict with utilization, cost, compatibility predictions
        """
        # Mean pooling across shipments
        batch_repr = embeddings.mean(dim=1)  # [batch_size, embed_dim]
        h = self.aggregator(batch_repr)

        return {
            "utilization": torch.sigmoid(self.util_head(h)),  # [0, 1]
            "cost": F.softplus(self.cost_head(h)),  # >= 0
            "compatibility": torch.sigmoid(self.compat_head(h)),  # [0, 1]
        }


class NeuralCFA:
    """
    Neural Cost Function Approximator

    Learning:
    - Train from completed route outcomes
    - Supervised learning: predict actual util, cost
    - RL signal: total route reward

    Usage:
    1. optimize() - called by decision_engine (inference)
    2. learn_from_outcome() - called by multi_scale_coordinator (training)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")  # No GPU required

        # Neural components
        self.shipment_encoder = ShipmentEncoder().to(self.device)
        self.batch_estimator = BatchValueEstimator().to(self.device)

        # Optimizer
        params = list(self.shipment_encoder.parameters()) + list(
            self.batch_estimator.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=0.001)

        # Training state
        self.num_updates = 0
        self.recent_losses = []

        # Fallback to MIP if neural isn't confident yet
        self.use_neural = False
        self.min_training_samples = 50

        logger.info("Neural CFA initialized - will use MIP until trained")

    def optimize(self, state: SystemState, value_function=None) -> CFASolution:
        """
        Main decision interface - called by decision_engine

        Flow:
        1. If not enough training -> fallback to simple heuristic
        2. Encode all shipments
        3. Generate candidate batches
        4. Score with neural network
        5. Select best batches
        """
        import time

        start = time.time()

        shipments = state.pending_shipments
        vehicles = state.get_available_vehicles()

        if not shipments or not vehicles:
            return CFASolution([], shipments, 0, 0, "INFEASIBLE", 0, "No resources")

        # Use heuristic if not trained enough
        if self.num_updates < self.min_training_samples:
            return self._heuristic_dispatch(shipments, vehicles, state)

        # Neural batch formation
        try:
            batches = self._neural_batch_formation(shipments, vehicles, state)

            if not batches:
                return CFASolution(
                    [],
                    shipments,
                    0,
                    0,
                    "NO_BATCHES",
                    time.time() - start,
                    "Neural found no good batches",
                )

            # Convert to StandardBatch format
            standard_batches = []
            assigned_ids = set()

            for batch_info in batches:
                batch_ships = batch_info["shipments"]
                vehicle = batch_info["vehicle"]

                # Simple route sequence (TSP would go here)
                route_seq = self._simple_route_sequence(batch_ships, vehicle)
                dist_km, dur_hrs = self._estimate_route_metrics(route_seq)

                total_vol = sum(s.volume for s in batch_ships)
                utilization = (
                    total_vol / vehicle.capacity.volume
                    if vehicle.capacity.volume > 0
                    else 0
                )

                standard_batches.append(
                    StandardBatch(
                        id=f"batch_{len(standard_batches)}",
                        shipments=batch_ships,
                        vehicle=vehicle,
                        route_sequence=route_seq,
                        total_distance_km=dist_km,
                        total_duration_hours=dur_hrs,
                        utilization=utilization,
                        total_cost=dist_km * self.config.get("cost_per_km", 50),
                        estimated_completion=state.timestamp,
                    )
                )

                assigned_ids.update(s.id for s in batch_ships)

            unassigned = [s for s in shipments if s.id not in assigned_ids]
            total_cost = sum(b.total_cost for b in standard_batches)
            avg_util = np.mean([b.utilization for b in standard_batches])

            return CFASolution(
                batches=standard_batches,
                unassigned_shipments=unassigned,
                total_cost=total_cost,
                avg_utilization=avg_util,
                status="OPTIMAL",
                solver_time_seconds=time.time() - start,
                reasoning=f"Neural: {len(standard_batches)} batches, {avg_util:.1%} util",
            )

        except Exception as e:
            logger.error(f"Neural CFA failed: {e}", exc_info=True)
            return self._heuristic_dispatch(shipments, vehicles, state)

    def _neural_batch_formation(self, shipments, vehicles, state):
        """Use neural network to form batches"""
        # Encode all shipments
        ship_features = torch.stack(
            [self._extract_shipment_features(s, state) for s in shipments]
        ).to(self.device)

        self.shipment_encoder.eval()
        with torch.no_grad():
            embeddings = self.shipment_encoder(ship_features)

        # Generate and score candidate batches
        candidates = []
        for vehicle in vehicles:
            # Try different batch sizes
            for batch_size in range(1, min(5, len(shipments) + 1)):
                # Greedy: take most urgent shipments that fit
                batch_ships = self._select_urgent_shipments(
                    shipments, vehicle, batch_size, state
                )

                if not batch_ships:
                    continue

                # Get embeddings for this batch
                batch_indices = [shipments.index(s) for s in batch_ships]
                batch_embeds = embeddings[batch_indices].unsqueeze(0)

                # Score batch
                self.batch_estimator.eval()
                with torch.no_grad():
                    scores = self.batch_estimator(batch_embeds)

                compatibility = scores["compatibility"].item()
                pred_util = scores["utilization"].item()
                pred_cost = scores["cost"].item()

                # Combined score
                score = compatibility * pred_util - 0.01 * pred_cost

                candidates.append(
                    {
                        "shipments": batch_ships,
                        "vehicle": vehicle,
                        "score": score,
                        "pred_util": pred_util,
                        "pred_cost": pred_cost,
                    }
                )

        # Select non-overlapping batches greedily
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        assigned = set()

        for cand in candidates:
            ship_ids = {s.id for s in cand["shipments"]}
            if ship_ids & assigned:
                continue

            selected.append(cand)
            assigned.update(ship_ids)

            if len(assigned) >= len(shipments):
                break

        return selected

    def learn_from_outcome(self, batch_formation, actual_outcome):
        """
        Update neural networks from completed route

        Args:
            batch_formation: dict with shipments, vehicle, predicted metrics
            actual_outcome: RouteOutcome from database
        """
        self.shipment_encoder.train()
        self.batch_estimator.train()

        # Extract features
        shipments = batch_formation["shipments"]
        ship_features = torch.stack(
            [
                self._extract_shipment_features(s, None)  # No state needed for training
                for s in shipments
            ]
        ).to(self.device)

        # Forward pass
        embeddings = self.shipment_encoder(ship_features)
        predictions = self.batch_estimator(embeddings.unsqueeze(0))

        # Loss: supervised learning on actual outcomes
        actual_util = actual_outcome.actual_utilization
        actual_cost = actual_outcome.actual_cost

        loss_util = F.mse_loss(
            predictions["utilization"],
            torch.tensor([[actual_util]], device=self.device),
        )
        loss_cost = F.mse_loss(
            predictions["cost"], torch.tensor([[actual_cost]], device=self.device)
        )

        # Reward signal (optional RL component)
        success_reward = 1.0 if actual_outcome.on_time_delivery_rate > 0.9 else 0.0
        loss_compat = -success_reward * torch.log(predictions["compatibility"] + 1e-8)

        total_loss = loss_util + 0.01 * loss_cost + 0.1 * loss_compat.mean()

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.shipment_encoder.parameters())
            + list(self.batch_estimator.parameters()),
            1.0,
        )
        self.optimizer.step()

        # Track progress
        self.num_updates += 1
        self.recent_losses.append(total_loss.item())
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        if self.num_updates % 10 == 0:
            avg_loss = np.mean(self.recent_losses)
            logger.info(
                f"CFA learning: updates={self.num_updates}, "
                f"loss={avg_loss:.4f}, util_err={loss_util.item():.4f}"
            )

    def _extract_shipment_features(self, shipment, state=None):
        """Convert shipment to feature vector"""
        # Basic features
        features = [
            shipment.volume / 100.0,  # Normalize
            shipment.weight / 1000.0,
            shipment.urgency_score if hasattr(shipment, "urgency_score") else 0.5,
        ]

        # Location embeddings (simple: use first 4 digits of lat/long)
        if shipment.pickup_location:
            features.extend(
                [
                    shipment.pickup_location.latitude / 90.0,
                    shipment.pickup_location.longitude / 180.0,
                ]
            )
        else:
            features.extend([0.0, 0.0])

        if len(shipment.delivery_locations) > 0:
            features.extend(
                [
                    shipment.delivery_locations[0].latitude / 90.0,
                    shipment.delivery_locations[0].longitude / 180.0,
                ]
            )
        else:
            features.extend([0.0, 0.0])

        # Time pressure
        if state:
            time_pressure = shipment.time_pressure(state.timestamp)
            features.append(min(time_pressure, 10.0) / 10.0)
        else:
            features.append(0.5)

        # Pad to fixed size
        while len(features) < 16:
            features.append(0.0)

        return torch.tensor(features[:16], dtype=torch.float32)

    def _heuristic_dispatch(self, shipments, vehicles, state):
        """Simple fallback when neural not ready"""
        # Sort by urgency
        sorted_ships = sorted(
            shipments, key=lambda s: s.time_pressure(state.timestamp), reverse=True
        )

        batches = []
        assigned = set()

        for vehicle in vehicles:
            batch_ships = []
            vol = 0

            for ship in sorted_ships:
                if ship.id in assigned:
                    continue

                if vol + ship.volume <= vehicle.capacity.volume * 0.9:
                    batch_ships.append(ship)
                    vol += ship.volume
                    assigned.add(ship.id)

                if len(batch_ships) >= 4:
                    break

            if batch_ships:
                route_seq = self._simple_route_sequence(batch_ships, vehicle)
                dist_km, dur_hrs = self._estimate_route_metrics(route_seq)
                util = vol / vehicle.capacity.volume

                batches.append(
                    StandardBatch(
                        id=f"batch_{len(batches)}",
                        shipments=batch_ships,
                        vehicle=vehicle,
                        route_sequence=route_seq,
                        total_distance_km=dist_km,
                        total_duration_hours=dur_hrs,
                        utilization=util,
                        total_cost=dist_km * self.config.get("cost_per_km", 50),
                        estimated_completion=state.timestamp,
                    )
                )

        unassigned = [s for s in shipments if s.id not in assigned]
        total_cost = sum(b.total_cost for b in batches)
        avg_util = np.mean([b.utilization for b in batches]) if batches else 0

        return CFASolution(
            batches,
            unassigned,
            total_cost,
            avg_util,
            "HEURISTIC",
            0,
            f"Fallback: {len(batches)} batches",
        )

    def _select_urgent_shipments(self, shipments, vehicle, batch_size, state):
        """Select most urgent shipments that fit vehicle"""
        candidates = [s for s in shipments if s.volume <= vehicle.capacity.volume * 0.3]

        candidates.sort(key=lambda s: s.time_pressure(state.timestamp), reverse=True)

        selected = []
        vol = 0

        for ship in candidates[:batch_size]:
            if vol + ship.volume <= vehicle.capacity.volume * 0.9:
                selected.append(ship)
                vol += ship.volume

        return selected

    def _simple_route_sequence(self, shipments, vehicle):
        """Placeholder for TSP solver"""
        # Nearest neighbor heuristic
        return [s.id for s in shipments]

    def _estimate_route_metrics(self, route_seq):
        """Estimate distance and duration"""
        # Placeholder - should use actual routing
        num_stops = len(route_seq)
        dist_km = num_stops * 15.0  # Avg 15km per stop
        dur_hrs = num_stops * 0.5  # Avg 30min per stop
        return dist_km, dur_hrs

    def save_weights(self, path):
        """Save neural network weights"""
        torch.save(
            {
                "shipment_encoder": self.shipment_encoder.state_dict(),
                "batch_estimator": self.batch_estimator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_updates": self.num_updates,
            },
            path,
        )
        logger.info(f"CFA weights saved: {self.num_updates} updates")

    def load_weights(self, path):
        """Load neural network weights"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.shipment_encoder.load_state_dict(checkpoint["shipment_encoder"])
            self.batch_estimator.load_state_dict(checkpoint["batch_estimator"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.num_updates = checkpoint["num_updates"]
            logger.info(f"CFA weights loaded: {self.num_updates} updates")
        except FileNotFoundError:
            logger.info("No CFA weights found - starting fresh")
