"""
optimizer.py — rolling-horizon dispatch MILP for a mixed fleet of AVs and human-driven EVs.

At each control step:

  min   α·Σ_j u_j                              uncovered demand
      + β·Σ_{i,j} E[τ]_{ij}·x_{ij}             repositioning travel time
      − δ·Σ_{i,j} π_ij·d̂_j(k+1)·x_{ij}·1[τ_{ij}≤ΔT]   look-ahead reward
      + Σ_{i,s} (β·E[τ]_{is} + γ_w·E[w]_{is})·y_{is}  station travel time and predicted wait
      + M·Σ_s v_s                               (static capacity variant only) overflow

  s.t.  Σ_i x̃_{ij} + u_j ≥ d̂_j + m·ξ_j (+ Γ_π protection)   demand coverage, all j
        at most one order per vehicle; vehicles with SOC < e_min must charge

x̃_{ij} is the supply of vehicle i to zone j: x_{ij} for a compliance-blind dispatcher, or the
expected supply under the compliance coefficients of LogitComplianceModel (followed order,
fallback position after an ignored order, and the vehicle left without an order). The
station choice follows one of STATION_RULES: with charge_wait_weight > 0 the station wait w_{is} is
predicted by StationState.predicted_wait (or a heuristic rule) and the one-interval capacity
constraint (Γ_c-robust, duals z_s, p_{is}) is dropped; otherwise a capacity constraint with each
vehicle's own charging time q_i limits the stations.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import os

import pulp

from chicago_loop.env.fleet import SOC_THRESHOLD as SOC_THRESHOLD_

from chicago_loop.env.fleet import StationState, VehicleState, VehicleType
from chicago_loop.env.fleet import BATTERY_CAPACITY_KWH, MAX_CHARGE_POWER_KW

# Full session from the SOC threshold to the target SOC (0.2 -> 0.8 at 50 kW: 43.2 min)
SESSION_MIN = (0.80 - SOC_THRESHOLD_) * BATTERY_CAPACITY_KWH / MAX_CHARGE_POWER_KW * 60.0
UNAVAILABLE_WAIT_MIN = 100.0    # "available" rule: stations without an idle charger rank after all others

logger = logging.getLogger(__name__)

# Station-choice rules for the vehicles that must charge.
#   static     capacity free in the next interval, Σ_i q_i·ỹ_is ≤ C_s + v_s (a vehicle may get no station)
#   static_mc  the same constraint, but every vehicle that must charge receives a station
#   horizon    capacity over one full session H: Σ_n max(0, H − r_sn) minus the charging minutes of
#              the vehicles queued or driving to the station
#   wait       predicted wait from replaying the backlog (StationState.predicted_wait)
#   wait_seq   the same, with the vehicles that must charge in this step assigned one at a time
#              (nearest first), each added to the backlog of its station before the next is assigned
#   wait_fix   ablation of wait_seq: each vehicle's station is fixed before the solve by the same
#              argmin, but without adding it to the backlog (isolates the sequential update)
#   wait_seq_h the same, but human drivers are assigned first, each to the station he would choose
#              himself (the nearest), and the AVs are then assigned around them
#   nearest    nearest station (w = 0)
#   available  nearest station with an idle charger, otherwise the nearest station
#   queue      shortest-queue estimate w = max(0, n_s + 1 − N_s)·H/N_s from the vehicle count only
STATION_RULES = ("static", "static_mc", "horizon", "wait", "wait_seq", "wait_fix", "wait_seq_h", "nearest", "available", "queue")


# ---------------------------------------------------------------------------
# Input / output data structures
# ---------------------------------------------------------------------------

@dataclass
class DispatchInput:
    """All inputs required by the SOMFD optimizer at one control step."""

    # Vehicles available for dispatch this step
    vehicles: list[VehicleState]

    # Stations
    stations: list[StationState]

    # Demand zones
    zone_ids: list[str]

    # Forecasted demand for current step: d̂_j(k)
    demand_forecast: dict[str, float]       # zone_id -> trips

    # Forecasted demand for next step: d̂_j(k+1)  (look-ahead)
    demand_forecast_next: dict[str, float]  # zone_id -> trips

    # Per-zone demand uncertainty bounds ξ_j(k)
    demand_uncertainty: dict[str, float]    # zone_id -> bound

    # Pre-computed travel times (minutes)
    travel_time_to_zone: dict[tuple, float]    # (vehicle_id, zone_id) -> min
    travel_time_to_station: dict[tuple, float] # (vehicle_id, station_id) -> min

    # Compliance coefficients (from LogitComplianceModel.effective_supply_coefficients)
    # compliance_coeffs[vehicle_id] = {'pi_zone': {...}, 'pi_station': {...}, 'self_action': ...}
    compliance_coeffs: dict[str, dict] = field(default_factory=dict)

    # Control interval ΔT (minutes)
    delta_t: float = 5.0


@dataclass
class DispatchResult:
    """Output of one SOMFD solve."""
    status: str
    # zone assignments: vehicle_id -> zone_id (None = no zone assignment)
    zone_assignments: dict[str, Optional[str]]
    # station assignments: vehicle_id -> station_id (None = no station assignment)
    station_assignments: dict[str, Optional[str]]
    objective_value: float
    solve_time: float
    unserved_demand: dict[str, float]   # zone_id -> σ_j^d
    station_slack: dict[str, float]     # station_id -> σ_s^c
    has_congestion: bool


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class ERMPCOptimizer:
    """
    Extended Robust MPC optimizer.

    Parameters
    ----------
    gamma_d : float
        Kept for compatibility: any value > 0 means demand_margin = 1, 0 means no margin.
        (The earlier "Γ_d-robust" form with one shared dual ν reduced to d̂_j + ξ_j for every
        Γ_d ≥ 1, so it was a per-zone safety margin, not a budget across zones.)
    demand_margin : float
        m in the coverage target d̂_j + m·ξ_j, where ξ_j = κ·σ_j is the forecast uncertainty.
    gamma_c : float
        Γ_c — charging-side robustness (number of vehicles at worst-case duration).
        0 = deterministic, |V^vac| = full worst-case.
    alpha : float
        Weight on unserved demand penalty.
    beta : float
        Weight on travel time cost.
    delta : float
        Look-ahead incentive weight.
    M : float
        Big-M penalty for charging capacity violation.
    solver : str
        "auto" | "gurobi" | "cbc"
    time_limit : float
        Solver time limit in seconds.
    mip_gap : float
        MIP optimality gap.
    """

    _gurobi_available: Optional[bool] = None

    def __init__(
        self,
        gamma_d: float = 2.0,
        gamma_c: float = 2.0,
        alpha: float = 100.0,
        beta: float = 1.0,
        delta: float = 0.3,
        M: float = 1e5,
        gamma_pi: float = 0.0,
        demand_margin: Optional[float] = None,
        idle_supply: bool = True,
        charge_wait_weight: float = 0.0,
        station_rule: Optional[str] = None,
        expected_cost: bool = True,
        fallback_credit: bool = True,
        pi_discount: bool = True,
        idle_to_fallback: bool = False,
        solver: str = "auto",
        time_limit: float = 30.0,
        mip_gap: float = 0.02,
    ):
        self.gamma_d = gamma_d
        self.gamma_c = gamma_c
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.M = M
        self.demand_margin = (1.0 if gamma_d > 0 else 0.0) if demand_margin is None else demand_margin
        self.gamma_pi = gamma_pi
        self.idle_supply = idle_supply   # count vehicles left in place as supply of their zone
        self.charge_wait_weight = charge_wait_weight   # γ_w on the backlog-predicted wait at a station
        # With compliance coefficients, take travel cost and look-ahead reward in expectation over
        # compliance, as the supply is: a driver who does not follow an order drives to his fallback.
        self.expected_cost = expected_cost
        # Diagnostics (counterfactual ablations of the compliance-aware supply): credit refused
        # zone orders to the fallback zone; discount a zone order by its compliance probability.
        self.fallback_credit = fallback_credit
        self.pi_discount = pi_discount
        # a human driver left without an order takes his fallback action (credit his fallback zone)
        self.idle_to_fallback = idle_to_fallback
        # Station choice of the vehicles below the SOC threshold (see STATION_RULES). The wait rules
        # price a station wait w_is with weight γ_w and drop the capacity constraint; the capacity
        # rules keep it. Every rule except "static" gives each vehicle that must charge a station.
        self.station_rule = station_rule or ("wait" if charge_wait_weight > 0 else "static")
        assert self.station_rule in STATION_RULES, self.station_rule
        self.static_charge_cap = self.station_rule in ("static", "static_mc", "horizon")
        self.must_charge = self.station_rule != "static"
        self.solver = solver
        self.time_limit = time_limit
        self.mip_gap = mip_gap

    # ------------------------------------------------------------------
    # Solver selection
    # ------------------------------------------------------------------

    @classmethod
    def _check_gurobi(cls) -> bool:
        if cls._gurobi_available is not None:
            return cls._gurobi_available
        try:
            import gurobipy
            m = gurobipy.Model()
            m.setParam("OutputFlag", 0)
            v = m.addVar()
            m.setObjective(v)
            m.optimize()
            cls._gurobi_available = True
        except Exception:
            cls._gurobi_available = False
        return cls._gurobi_available

    def _get_solver(self):
        use_gurobi = (
            self.solver == "gurobi"
            or (self.solver == "auto" and self._check_gurobi())
        )
        if use_gurobi:
            try:
                return pulp.GUROBI(msg=0, timeLimit=self.time_limit, mipGap=self.mip_gap,
                                   Threads=int(os.environ.get("ERMPC_GRB_THREADS", "0")))
            except Exception:
                pass
        return pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit, gapRel=self.mip_gap)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, inp: DispatchInput) -> DispatchResult:
        t0 = time.time()

        vehicles = inp.vehicles
        stations = inp.stations
        zones = inp.zone_ids

        if not vehicles:
            return self._empty_result(stations, zones, time.time() - t0)

        v_ids = [v.vehicle_id for v in vehicles]
        s_ids = [s.station_id for s in stations]
        z_ids = zones

        v_map = {v.vehicle_id: v for v in vehicles}
        s_map = {s.station_id: s for s in stations}
        if self.station_rule.startswith("wait_seq"):  # the sequential assignment extends copies of the backlogs
            import copy
            s_map = {k: copy.deepcopy(st) for k, st in s_map.items()}

        prob = pulp.LpProblem("SOMFD", pulp.LpMinimize)

        # ------------------------------------------------------------------
        # Decision variables
        # ------------------------------------------------------------------

        # x[i,j] ∈ {0,1}: vehicle i repositioned to zone j
        x = pulp.LpVariable.dicts("x", [(i, j) for i in v_ids for j in z_ids], cat="Binary")

        # y[i,s] ∈ {0,1}: vehicle i sent to charge at station s
        y = pulp.LpVariable.dicts("y", [(i, s) for i in v_ids for s in s_ids], cat="Binary")

        # σ_j^d ≥ 0: unserved demand slack per zone
        sigma_d = pulp.LpVariable.dicts("sigma_d", z_ids, lowBound=0)

        # σ_s^c ≥ 0: charging overflow slack per station
        sigma_c = pulp.LpVariable.dicts("sigma_c", s_ids, lowBound=0)

        # Γ_π robustness (compliance probabilities): π_ij ∈ [π̂_ij − ε_ij, π̂_ij + ε_ij], at most Γ_π
        # human drivers per zone at their worst case. Duals μ_j ≥ 0, q_ij ≥ 0.
        humans_pi = [i for i in v_ids if v_map[i].vehicle_type != VehicleType.AV
                     and inp.compliance_coeffs.get(i, {}).get("eps_zone")] if self.gamma_pi > 0 else []
        mu_pi = pulp.LpVariable.dicts("mu_pi", z_ids, lowBound=0) if humans_pi else {}
        q_pi = pulp.LpVariable.dicts("q_pi", [(i, j) for i in humans_pi for j in z_ids], lowBound=0) if humans_pi else {}

        # Γ_c robustness dual variables: z_s ≥ 0, p[i,s] ≥ 0
        z_rob = pulp.LpVariable.dicts("z_rob", s_ids, lowBound=0)
        p_rob = pulp.LpVariable.dicts("p_rob", [(i, s) for i in v_ids for s in s_ids], lowBound=0)

        # ------------------------------------------------------------------
        # Compliance-adjusted effective supply  (Eqs. 3.3.7–3.3.8)
        # x̃_{ij} = π_i·x_{ij} + (1−π_i)·𝟏[a_i^self = r_j]
        # ỹ_{is} = π_i·y_{is} + (1−π_i)·𝟏[a_i^self = c_s]
        # ------------------------------------------------------------------

        from chicago_loop.models.compliance import RepositionAction, ChargeAction

        def x_tilde(i: str, j: str):
            v = v_map[i]
            if v.vehicle_type == VehicleType.AV:
                return x[(i, j)]
            coeffs = inp.compliance_coeffs.get(i, {})
            if not coeffs:                      # compliance-blind dispatcher
                return x[(i, j)]
            piz = coeffs.get("pi_zone", {})
            self_action = coeffs.get("self_action", None)
            # x̃_ij = π_ij x_ij + b_ij Σ_k (1 − π_ik) x_ik: a driver who does not follow the zone order
            # he actually receives goes to his fallback zone (b_ij = 1 if that zone is j).
            own = piz.get(j, 1.0) if self.pi_discount else 1.0
            if self.fallback_credit and isinstance(self_action, RepositionAction) and self_action.zone_id == j:
                return own * x[(i, j)] + pulp.lpSum(
                    (1.0 - piz.get(k, 1.0)) * x[(i, k)] for k in z_ids if self.pi_discount or k != j)
            return own * x[(i, j)]

        def y_tilde(i: str, s: str):
            v = v_map[i]
            if v.vehicle_type == VehicleType.AV:
                return y[(i, s)]
            coeffs = inp.compliance_coeffs.get(i, {})
            pi = coeffs.get("pi_station", {}).get(s, 1.0)
            self_action = coeffs.get("self_action", None)
            baseline = (1 - pi) * (1.0 if (isinstance(self_action, ChargeAction)
                                            and self_action.station_id == s) else 0.0)
            return pi * y[(i, s)] + baseline

        # ------------------------------------------------------------------
        # Objective  (Eq. 4.3 in paper)
        # ------------------------------------------------------------------

        unserved = self.alpha * pulp.lpSum(sigma_d[j] for j in z_ids)

        def pi_ij(i, j, station=False):
            """Compliance probability of an order (1 for AVs and the compliance-blind dispatcher)."""
            c = inp.compliance_coeffs.get(i, {})
            if v_map[i].vehicle_type == VehicleType.AV or not c or not self.expected_cost or not self.pi_discount:
                return 1.0
            return c.get("pi_station" if station else "pi_zone", {}).get(j, 1.0)

        def tau_self(i):
            """Travel time to the fallback destination of a human driver."""
            a = inp.compliance_coeffs.get(i, {}).get("self_action")
            if isinstance(a, RepositionAction):
                return inp.travel_time_to_zone.get((i, a.zone_id), 0.0)
            if isinstance(a, ChargeAction):
                return inp.travel_time_to_station.get((i, a.station_id), 0.0)
            return 0.0

        def e_tau(i, dest, station=False):
            tt = (inp.travel_time_to_station if station else inp.travel_time_to_zone).get((i, dest), 0.0)
            p = pi_ij(i, dest, station)
            return tt if p >= 1.0 else p * tt + (1.0 - p) * tau_self(i)

        def wait_at(i, s):
            st = s_map[s]
            if self.station_rule == "nearest":
                return 0.0
            if self.station_rule == "available":
                return 0.0 if st.idle_piles > 0 else UNAVAILABLE_WAIT_MIN
            if self.station_rule == "queue":
                n = len(st.in_service) + sum(1 for eta, _ in st.committed if eta <= 0.0)
                return max(0, n + 1 - st.num_piles) * SESSION_MIN / st.num_piles
            return st.predicted_wait(inp.travel_time_to_station.get((i, s), 0.0))

        def e_wait(i, s):
            p = pi_ij(i, s, station=True)
            if p >= 1.0:
                return wait_at(i, s)
            a = inp.compliance_coeffs[i].get("self_action")
            w_self = wait_at(i, a.station_id) if isinstance(a, ChargeAction) else 0.0
            return p * wait_at(i, s) + (1.0 - p) * w_self

        # Sequential station choice: fix the station of each vehicle that must charge in turn and
        # add it to that station's backlog, so later vehicles of the same step see it.
        seq_station = {}
        if self.station_rule in ("wait_seq", "wait_seq_h", "wait_fix") and s_ids:
            tts = inp.travel_time_to_station
            humans_first = self.station_rule == "wait_seq_h"
            is_h = lambda i: humans_first and v_map[i].vehicle_type != VehicleType.AV
            must = sorted((i for i in v_ids if v_map[i].soc < SOC_THRESHOLD_),
                          key=lambda i: (not is_h(i), min(tts.get((i, s), 0.0) for s in s_ids)))
            for i in must:
                if is_h(i):
                    best = min(s_ids, key=lambda s: tts.get((i, s), 0.0))
                else:
                    best = min(s_ids, key=lambda s: self.beta * e_tau(i, s, station=True)
                               + self.charge_wait_weight * e_wait(i, s))
                seq_station[i] = best
                if self.station_rule == "wait_fix":
                    continue
                # with compliance coefficients, a human driver joins the backlog of the station he
                # most likely drives to (the order, or his fallback station)
                a = inp.compliance_coeffs.get(i, {}).get("self_action")
                dest = (a.station_id if isinstance(a, ChargeAction) and pi_ij(i, best, station=True) < 0.5
                        else best)
                s_map[dest].committed.append((tts.get((i, dest), 0.0), v_map[i].nominal_charge_duration_min()))

        reposition_cost = self.beta * pulp.lpSum(
            e_tau(i, j) * x[(i, j)]
            for i in v_ids for j in z_ids
        )

        charge_cost = self.beta * pulp.lpSum(
            e_tau(i, s, station=True) * y[(i, s)]
            for i in v_ids for s in s_ids
        )
        if not self.static_charge_cap:
            # Station wait of the chosen rule (backlog replay, or one of the heuristic baselines)
            charge_cost += self.charge_wait_weight * pulp.lpSum(
                e_wait(i, s) * y[(i, s)]
                for i in v_ids for s in s_ids if v_map[i].soc < SOC_THRESHOLD_)

        congestion = self.M * pulp.lpSum(sigma_c[s] for s in s_ids)

        # Look-ahead: incentivise sending vehicles to zones reachable within ΔT
        # that have high predicted demand next step
        lookahead = self.delta * pulp.lpSum(
            pi_ij(i, j) * inp.demand_forecast_next.get(j, 0.0) * x[(i, j)]
            for i in v_ids for j in z_ids
            if inp.travel_time_to_zone.get((i, j), float("inf")) <= inp.delta_t
        )

        prob += unserved + reposition_cost + charge_cost + congestion - lookahead

        # ------------------------------------------------------------------
        # Assignment feasibility  (Eq. 3.2.3)
        # ------------------------------------------------------------------

        for i in v_ids:
            prob += (
                pulp.lpSum(x[(i, j)] for j in z_ids)
                + pulp.lpSum(y[(i, s)] for s in s_ids)
                <= 1,
                f"assign_unique_{i}"
            )

        # SOC eligibility: charging only if SOC < e_min  (Eq. 3.2.4)
        from chicago_loop.env.fleet import SOC_THRESHOLD
        for i in v_ids:
            if v_map[i].soc >= SOC_THRESHOLD:
                for s in s_ids:
                    prob += y[(i, s)] == 0, f"soc_elig_{i}_{s}"
            else:                                   # below the threshold a vehicle must charge
                for j in z_ids:
                    prob += x[(i, j)] == 0, f"soc_trip_{i}_{j}"
                if self.must_charge and s_ids:
                    prob += pulp.lpSum(y[(i, s)] for s in s_ids) == 1, f"must_charge_{i}"
                for s in (s_ids if i in seq_station else []):
                    if s != seq_station[i]:
                        prob += y[(i, s)] == 0, f"seq_station_{i}_{s}"

        # ------------------------------------------------------------------
        # Demand coverage with a safety margin and Γ_π compliance protection
        #
        # Σ_i x̃_{ij} + σ_j^d ≥ d̂_j + m·ξ_j + Γ_π·μ_j + Σ_i q_ij    ∀j
        # ------------------------------------------------------------------

        def _self_zone(i):
            a = inp.compliance_coeffs.get(i, {}).get("self_action", None)
            return a.zone_id if isinstance(a, RepositionAction) else None

        def _idle_zone(i):
            if (self.idle_to_fallback and v_map[i].vehicle_type != VehicleType.AV
                    and inp.compliance_coeffs.get(i) and _self_zone(i)):
                return _self_zone(i)
            return v_map[i].node_id

        for j in z_ids:
            effective_supply_j = pulp.lpSum(x_tilde(i, j) for i in v_ids)
            if self.idle_supply:
                # a vehicle left without an order stays vacant in its zone and serves requests there
                # (or, with idle_to_fallback, a human driver goes to his fallback zone)
                effective_supply_j += pulp.lpSum(
                    1 - pulp.lpSum(x[(i, k)] for k in z_ids) - pulp.lpSum(y[(i, s)] for s in s_ids)
                    for i in v_ids if _idle_zone(i) == j)
            d_hat_j = inp.demand_forecast.get(j, 0.0)
            protect_pi = 0
            if humans_pi:
                # Loss of zone-j supply if driver i's compliance probability deviates by ε:
                # assigned to j (and j is not his own choice) -> he complies less;
                # his own choice is j but he is assigned to k -> he complies more and does not come.
                for i in humans_pi:
                    eps = inp.compliance_coeffs[i]["eps_zone"]
                    b = 1.0 if _self_zone(i) == j else 0.0
                    loss = (1.0 - b) * eps.get(j, 0.0) * x[(i, j)] + b * pulp.lpSum(
                        eps.get(k, 0.0) * x[(i, k)] for k in z_ids if k != j)
                    prob += q_pi[(i, j)] + mu_pi[j] >= loss, f"rob_pi_dual_{i}_{j}"
                protect_pi = self.gamma_pi * mu_pi[j] + pulp.lpSum(q_pi[(i, j)] for i in humans_pi)
            prob += (
                effective_supply_j + sigma_d[j]
                >= d_hat_j + self.demand_margin * inp.demand_uncertainty.get(j, 0.0) + protect_pi,
                f"rob_demand_{j}"
            )

        # ------------------------------------------------------------------
        # Γ_c-robust charging capacity  (Eqs. 3.4.8–3.4.10)
        #
        # Σ_i q̄_i·ỹ_{is} + Γ_c·z_s + Σ_i p_{is} ≤ C_s(k) + σ_s^c   ∀s
        # p_{is} + z_s ≥ q̂_i·y_{is}                                    ∀i,s
        # ------------------------------------------------------------------

        from chicago_loop.env.fleet import CHARGE_DURATION_UNCERTAINTY

        for s in (s_ids if self.static_charge_cap else []):
            st = s_map[s]
            if self.station_rule == "horizon":
                busy = sorted(st.in_service)[: st.num_piles]
                C_s = (sum(max(0.0, SESSION_MIN - r) for r in busy) + (st.num_piles - len(busy)) * SESSION_MIN
                       - sum(d for _, d in st.committed))
            else:
                C_s = st.available_capacity_min(inp.delta_t)
            # q_i: the vehicle's own charging time to the target SOC (as in the simulator)
            prob += (
                pulp.lpSum(v_map[i].nominal_charge_duration_min() * y_tilde(i, s) for i in v_ids)
                + self.gamma_c * z_rob[s]
                + pulp.lpSum(p_rob[(i, s)] for i in v_ids)
                <= C_s + sigma_c[s],
                f"rob_charge_{s}"
            )
            for i in v_ids:
                prob += (
                    p_rob[(i, s)] + z_rob[s] >= CHARGE_DURATION_UNCERTAINTY * y[(i, s)],
                    f"rob_charge_dual_{i}_{s}"
                )

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------

        prob.solve(self._get_solver())
        solve_time = time.time() - t0
        status = pulp.LpStatus[prob.status]

        if status != "Optimal":
            logger.warning(f"SOMFD status: {status} in {solve_time:.2f}s")
            return self._empty_result(stations, zones, solve_time, status=status)

        # ------------------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------------------

        zone_assignments: dict[str, Optional[str]] = {i: None for i in v_ids}
        station_assignments: dict[str, Optional[str]] = {i: None for i in v_ids}

        for i in v_ids:
            for j in z_ids:
                if pulp.value(x[(i, j)]) is not None and pulp.value(x[(i, j)]) > 0.5:
                    zone_assignments[i] = j
            for s in s_ids:
                if pulp.value(y[(i, s)]) is not None and pulp.value(y[(i, s)]) > 0.5:
                    station_assignments[i] = s

        unserved_demand = {j: (pulp.value(sigma_d[j]) or 0.0) for j in z_ids}
        station_slack = {s: (pulp.value(sigma_c[s]) or 0.0) for s in s_ids}
        has_congestion = any(v > 0.01 for v in station_slack.values())

        logger.info(
            f"SOMFD solved in {solve_time:.2f}s | obj={pulp.value(prob.objective):.1f} | "
            f"congestion={has_congestion}"
        )

        return DispatchResult(
            status=status,
            zone_assignments=zone_assignments,
            station_assignments=station_assignments,
            objective_value=pulp.value(prob.objective),
            solve_time=solve_time,
            unserved_demand=unserved_demand,
            station_slack=station_slack,
            has_congestion=has_congestion,
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _empty_result(self, stations, zones, solve_time, status="Optimal") -> DispatchResult:
        return DispatchResult(
            status=status,
            zone_assignments={},
            station_assignments={},
            objective_value=0.0,
            solve_time=solve_time,
            unserved_demand={z: 0.0 for z in zones},
            station_slack={s.station_id: 0.0 for s in stations},
            has_congestion=False,
        )
