"""
fleet.py — Vehicle state model for mixed AV/human EV taxi fleet.

Defines VehicleType, OperationalStatus, VehicleState, and StationState.
All downstream modules import from here to ensure a single source of truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VehicleType(enum.Enum):
    AV = "AV"          # Autonomous vehicle: always complies with dispatch
    HUMAN = "HUMAN"    # Human-driven taxi: complies probabilistically


class OperationalStatus(enum.Enum):
    VACANT = "VACANT"            # Available for dispatch
    OCCUPIED = "OCCUPIED"        # Carrying a passenger
    EN_ROUTE_CHARGE = "EN_ROUTE_CHARGE"   # Heading to charging station
    EN_ROUTE_ZONE = "EN_ROUTE_ZONE"       # Heading to demand zone (reposition)
    CHARGING = "CHARGING"        # Currently charging at a station
    LOW_BATTERY = "LOW_BATTERY"  # SOC < e_min, needs charging before anything else


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Complete state of one EV taxi at a given control step."""

    # Identity
    vehicle_id: str
    vehicle_type: VehicleType

    # Location & motion
    node_id: str               # Current network node (map-matched)
    edge_id: str = ""          # Current edge in SUMO network
    x: float = 0.0            # Geographic x-coordinate (m)
    y: float = 0.0            # Geographic y-coordinate (m)

    # Energy
    soc: float = 1.0           # State of charge ∈ [0, 1]
    battery_capacity_kwh: float = 60.0   # B_i
    consumption_rate: float = 0.20       # η_i  kWh/km
    max_charge_power_kw: float = 50.0    # P_i^max

    # Operational
    status: OperationalStatus = OperationalStatus.VACANT
    current_assignment: Optional[str] = None  # zone_id or station_id
    remaining_trip_time: float = 0.0          # minutes until current trip ends

    # Shift / preference (for compliance model)
    shift_remaining_min: float = 240.0   # minutes left in current shift
    preferred_zone: Optional[str] = None  # driver's habitual high-demand zone

    # Metrics accumulation
    total_km_empty: float = 0.0    # empty km (repositioning + en-route charge)
    total_km_occupied: float = 0.0

    @property
    def is_dispatchable(self) -> bool:
        """True if the vehicle can receive a new assignment this step."""
        return self.status == OperationalStatus.VACANT

    @property
    def needs_charging(self) -> bool:
        """True if SOC is below the recharging threshold."""
        return self.soc < SOC_THRESHOLD

    @property
    def range_km(self) -> float:
        """Estimated remaining range in km."""
        return self.soc * self.battery_capacity_kwh / self.consumption_rate

    def energy_needed_to_80pct(self) -> float:
        """kWh needed to reach 80% SOC from current SOC."""
        return max(0.0, (0.80 - self.soc) * self.battery_capacity_kwh)

    def nominal_charge_duration_min(self) -> float:
        """Estimated minutes to charge from current SOC to 80% at max power."""
        return self.energy_needed_to_80pct() / self.max_charge_power_kw * 60.0


# ---------------------------------------------------------------------------
# Charging station state
# ---------------------------------------------------------------------------

@dataclass
class StationState:
    """State of one charging station at a given control step."""

    station_id: str
    node_id: str               # Network node
    x: float = 0.0
    y: float = 0.0
    num_piles: int = 4         # N_s
    power_kw: float = 50.0    # P_s per pile

    # Dynamic occupancy
    occupied_piles: int = 0
    remaining_charge_times: list = field(default_factory=list)  # minutes per occupied pile
    queue_length: int = 0
    # Backlog: remaining charging minutes of the vehicles in service, and (arrival in min,
    # charging minutes) of the vehicles queued (arrival 0) or driving to the station.
    in_service: list = field(default_factory=list)
    committed: list = field(default_factory=list)

    def predicted_wait(self, arrival_min: float) -> float:
        """Wait of a vehicle arriving after arrival_min: replay the backlog FCFS over the piles."""
        import heapq
        free = sorted(self.in_service)[: self.num_piles]
        free += [0.0] * (self.num_piles - len(free))
        heapq.heapify(free)
        for eta, dur in sorted(self.committed):
            if eta > arrival_min:
                break
            start = max(eta, heapq.heappop(free))
            heapq.heappush(free, start + dur)
        return max(0.0, min(free) - arrival_min)

    @property
    def idle_piles(self) -> int:
        return max(0, self.num_piles - self.occupied_piles)

    def available_capacity_min(self, delta_t: float) -> float:
        """
        Temporal capacity C_s(k) for control interval ΔT (minutes).

        C_s(k) = Σ_{m=1}^{N_s} max(0, ΔT − τ_{s,m}^rem)
        """
        cap = self.idle_piles * delta_t
        for t_rem in self.remaining_charge_times:
            cap += max(0.0, delta_t - t_rem)
        return cap


# ---------------------------------------------------------------------------
# Fleet snapshot (passed to optimizer each step)
# ---------------------------------------------------------------------------

@dataclass
class FleetSnapshot:
    """All vehicle and station states at one control step."""
    step: int
    sim_time: float                          # seconds
    vehicles: dict[str, VehicleState] = field(default_factory=dict)
    stations: dict[str, StationState] = field(default_factory=dict)

    @property
    def dispatchable(self) -> list[VehicleState]:
        return [v for v in self.vehicles.values() if v.is_dispatchable]

    @property
    def av_dispatchable(self) -> list[VehicleState]:
        return [v for v in self.dispatchable if v.vehicle_type == VehicleType.AV]

    @property
    def human_dispatchable(self) -> list[VehicleState]:
        return [v for v in self.dispatchable if v.vehicle_type == VehicleType.HUMAN]


# ---------------------------------------------------------------------------
# Global constants (shared across modules)
# ---------------------------------------------------------------------------

SOC_THRESHOLD = 0.20          # e^min: triggers charging eligibility
DELTA_T_MIN = 5.0             # Control interval ΔT in minutes
BATTERY_CAPACITY_KWH = 60.0   # B_i (BYD e6)
CONSUMPTION_RATE = 0.20       # η_i kWh/km
MAX_CHARGE_POWER_KW = 50.0    # P_i^max
NOMINAL_CHARGE_DURATION_MIN = 18.0   # q̄_i (to 80% SOC)
CHARGE_DURATION_UNCERTAINTY = 3.0    # q̂_i
