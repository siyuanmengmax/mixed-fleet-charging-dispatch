"""
compliance.py — Logit-based behavioral compliance model.

Models the probability that a human taxi driver follows a dispatch recommendation
vs. deviating to their self-preferred action.

Reference: McFadden (1974), Train (2009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from chicago_loop.env.fleet import VehicleState, VehicleType


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepositionAction:
    """Dispatch to a demand zone for passenger pickup."""
    zone_id: str
    action_type: str = "reposition"


@dataclass(frozen=True)
class ChargeAction:
    """Send vehicle to a charging station."""
    station_id: str
    action_type: str = "charge"


Action = RepositionAction | ChargeAction


# ---------------------------------------------------------------------------
# Compliance model
# ---------------------------------------------------------------------------

@dataclass
class ComplianceParams:
    """
    Behavioral weight parameters for the driver utility model (Eq. 3.3.1).

    Calibrated from Urata et al. (2021) and Lei et al. (2022):
      mu1=1.0  — earnings weight (normalized)
      mu2=0.6  — travel cost weight (moderate distance sensitivity)
      mu3=0.8  — SOC urgency weight (strong aversion to low battery)
      theta    — compliance sensitivity (∈ [0.5, 2.5] in sensitivity analysis)
    """
    mu1: float = 1.0    # earnings weight
    mu2: float = 0.6    # travel cost (distance) weight
    mu3: float = 0.8    # SOC urgency weight
    theta: float = 1.5  # compliance sensitivity parameter


class LogitComplianceModel:
    """
    Binary Logit compliance model for human taxi drivers.

    For each human driver i assigned action a*, the compliance probability is:

        π_i(a*) = 1 / (1 + exp(−θ · ΔU_i(a*)))

    where ΔU_i(a*) = U_i(a*) − U_i(a_i^self) is the utility advantage of the
    recommended action over the driver's self-preferred alternative.

    AV compliance is always 1.0 by definition.
    """

    def __init__(self, params: Optional[ComplianceParams] = None):
        self.params = params or ComplianceParams()

    # ------------------------------------------------------------------
    # Utility function  (Eq. 3.3.1 in paper)
    # ------------------------------------------------------------------

    def driver_utility(
        self,
        vehicle: VehicleState,
        action: Action,
        zone_demand_density: dict[str, float],   # zone_id -> expected trips/interval
        travel_time_min: float,
        avg_trip_time_min: float = 8.0,
    ) -> float:
        """
        U_i(a) = μ1·R_i(a) − μ2·(τ_{i,a}/τ̄) − μ3·𝟏[a∈reposition]·ψ_i

        Parameters
        ----------
        vehicle : VehicleState
        action : RepositionAction or ChargeAction
        zone_demand_density : dict mapping zone_id to expected demand this step
        travel_time_min : travel time from vehicle's current location to action destination
        avg_trip_time_min : τ̄ — normalisation constant
        """
        p = self.params

        # Expected earnings R_i(a)
        if isinstance(action, RepositionAction):
            # Proportional to zone demand density (normalised to [0,1])
            total_demand = sum(zone_demand_density.values()) or 1.0
            earnings = zone_demand_density.get(action.zone_id, 0.0) / total_demand
        else:
            # Charging generates no direct revenue
            earnings = 0.0

        # Normalised travel cost
        travel_cost = travel_time_min / max(avg_trip_time_min, 1.0)

        # SOC urgency penalty: penalises reposition if vehicle is close to e_min
        from chicago_loop.env.fleet import SOC_THRESHOLD
        if isinstance(action, RepositionAction):
            soc_urgency = max(0.0, SOC_THRESHOLD - vehicle.soc) / SOC_THRESHOLD
        else:
            soc_urgency = 0.0  # No urgency penalty for charging action

        utility = (p.mu1 * earnings
                   - p.mu2 * travel_cost
                   - p.mu3 * soc_urgency)
        return utility

    # ------------------------------------------------------------------
    # Self-preferred action
    # ------------------------------------------------------------------

    def self_preferred_action(
        self,
        vehicle: VehicleState,
        zone_ids: list[str],
        station_ids: list[str],
        zone_demand_density: dict[str, float],
        travel_times_to_zones: dict[str, float],   # zone_id -> minutes
        travel_times_to_stations: dict[str, float], # station_id -> minutes
        avg_trip_time_min: float = 8.0,
    ) -> Action:
        """
        a_i^self — the action the driver would take without any recommendation.

        Heuristic: if SOC < e_min, go to nearest station; otherwise go to
        nearest high-demand zone (greedy local-information behaviour).
        """
        from chicago_loop.env.fleet import SOC_THRESHOLD

        if vehicle.soc < SOC_THRESHOLD and station_ids:
            # Low battery: nearest station
            nearest_station = min(
                station_ids,
                key=lambda s: travel_times_to_stations.get(s, float("inf"))
            )
            return ChargeAction(station_id=nearest_station)

        if zone_ids:
            # Nearest zone with highest demand/distance ratio (myopic)
            def zone_score(z: str) -> float:
                t = travel_times_to_zones.get(z, float("inf"))
                d = zone_demand_density.get(z, 0.0)
                return d / max(t, 0.1)

            best_zone = max(zone_ids, key=zone_score)
            return RepositionAction(zone_id=best_zone)

        # Fallback: idle
        return RepositionAction(zone_id=zone_ids[0]) if zone_ids else ChargeAction(station_id=station_ids[0])

    # ------------------------------------------------------------------
    # Compliance probability  (Eq. 3.3.2 in paper)
    # ------------------------------------------------------------------

    def compliance_probability(
        self,
        vehicle: VehicleState,
        recommended_action: Action,
        zone_demand_density: dict[str, float],
        travel_times_to_zones: dict[str, float],
        travel_times_to_stations: dict[str, float],
        zone_ids: list[str],
        station_ids: list[str],
        avg_trip_time_min: float = 8.0,
    ) -> float:
        """
        Returns π_i(a*) ∈ [0, 1].

        For AV vehicles always returns 1.0.
        For human vehicles applies the Logit formula.
        """
        if vehicle.vehicle_type == VehicleType.AV:
            return 1.0

        # Travel time to recommended action destination
        if isinstance(recommended_action, RepositionAction):
            t_rec = travel_times_to_zones.get(recommended_action.zone_id, 0.0)
        else:
            t_rec = travel_times_to_stations.get(recommended_action.station_id, 0.0)

        u_rec = self.driver_utility(
            vehicle, recommended_action, zone_demand_density,
            t_rec, avg_trip_time_min
        )

        # Self-preferred action
        self_action = self.self_preferred_action(
            vehicle, zone_ids, station_ids,
            zone_demand_density,
            travel_times_to_zones, travel_times_to_stations,
            avg_trip_time_min
        )

        if isinstance(self_action, RepositionAction):
            t_self = travel_times_to_zones.get(self_action.zone_id, 0.0)
        else:
            t_self = travel_times_to_stations.get(self_action.station_id, 0.0)

        u_self = self.driver_utility(
            vehicle, self_action, zone_demand_density,
            t_self, avg_trip_time_min
        )

        delta_u = u_rec - u_self
        pi = 1.0 / (1.0 + math.exp(-self.params.theta * delta_u))
        return pi

    # ------------------------------------------------------------------
    # Batch computation: effective supply coefficients
    # ------------------------------------------------------------------

    def effective_supply_coefficients(
        self,
        vehicle: VehicleState,
        zone_ids: list[str],
        station_ids: list[str],
        zone_demand_density: dict[str, float],
        travel_times_to_zones: dict[str, float],
        travel_times_to_stations: dict[str, float],
        avg_trip_time_min: float = 8.0,
    ) -> dict:
        """
        Pre-compute all compliance-related coefficients for one vehicle.

        Returns a dict with:
          'pi_zone'    : {zone_id: π_i(r_j)}   compliance prob for each zone assignment
          'pi_station' : {station_id: π_i(c_s)} compliance prob for each station assignment
          'self_action': the self-preferred Action object
          'pi_self'    : compliance prob of self action (used to compute baseline contribution)
        """
        if vehicle.vehicle_type == VehicleType.AV:
            return {
                'pi_zone': {z: 1.0 for z in zone_ids},
                'pi_station': {s: 1.0 for s in station_ids},
                'self_action': None,
                'pi_self': 1.0,
            }

        self_action = self.self_preferred_action(
            vehicle, zone_ids, station_ids,
            zone_demand_density,
            travel_times_to_zones, travel_times_to_stations,
            avg_trip_time_min
        )

        pi_zone = {}
        for z in zone_ids:
            pi_zone[z] = self.compliance_probability(
                vehicle, RepositionAction(z),
                zone_demand_density,
                travel_times_to_zones, travel_times_to_stations,
                zone_ids, station_ids, avg_trip_time_min
            )

        pi_station = {}
        for s in station_ids:
            pi_station[s] = self.compliance_probability(
                vehicle, ChargeAction(s),
                zone_demand_density,
                travel_times_to_zones, travel_times_to_stations,
                zone_ids, station_ids, avg_trip_time_min
            )

        # Compliance prob of the self-preferred action (always 1 by construction, but kept for clarity)
        pi_self = 1.0

        return {
            'pi_zone': pi_zone,
            'pi_station': pi_station,
            'self_action': self_action,
            'pi_self': pi_self,
        }
