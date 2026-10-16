"""
shift_sim.py — event-timed simulation of a mixed AV/human electric taxi fleet over a full
day shift in the Chicago Loop.

Differences from experiments/standalone_sim.py:
  * travel distances are shortest paths on the SUMO network (data/network/od_km.json) and
    speeds are the observed hourly taxi speeds in the Loop (data/network/loop_speed_2023.json);
  * times are continuous: requests appear at uniform times within a 5-min control step,
    vehicles arrive at their exact travel time, and waits are measured in minutes;
  * a request that is not picked up within TIMEOUT_MIN is abandoned and counted with a wait
    of TIMEOUT_MIN, so methods that serve fewer passengers do not look faster;
  * a vacant vehicle picks up requests that appear in its own zone without being dispatched;
  * a vehicle below SOC_MUST_CHARGE cannot take trips and must charge; stations hold a FIFO
    queue; a vehicle that reaches SOC 0 is counted as stranded;
  * a human driver who does not follow an assignment takes his self-preferred action:
    the nearest station when he must charge, otherwise the zone with the most pending requests
    per minute of travel (the rule of LogitComplianceModel.self_preferred_action);
  * with alpha_sd > 0, each human driver i has a persistent compliance propensity
    alpha_i ~ N(alpha_mean, alpha_sd^2) that shifts the logit of every compliance probability:
    pi_i = sigmoid(logit(pi) + alpha_i).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from chicago_loop.env.fleet import FleetSnapshot, OperationalStatus, StationState, VehicleState, VehicleType

DATA = Path(__file__).parent.parent / "data"
DT = 5.0                    # control step (min)
BATTERY_KWH = 60.0
CONSUMPTION_KWH_PER_KM = 0.20
CHARGE_KW = 50.0
SOC_MUST_CHARGE = 0.20
SOC_TARGET = 0.80
TIMEOUT_MIN = 10.0
# Pickup inside a zone: the 4.1 km2 Loop has 16 zones of about 0.26 km2 (side a = 0.51 km);
# the mean rectilinear distance between two uniform points in a square is 2a/3 = 0.34 km.
INTRA_ZONE_KM = 0.34


def load_network():
    od = json.load(open(DATA / "network" / "od_km.json"))["od_km"]
    sp = json.load(open(DATA / "network" / "loop_speed_2023.json"))["speed_kmh_by_hour"]
    nm = json.load(open(DATA / "network" / "node_mappings.json"))
    zones = [z["zone_id"] for z in nm["demand_zones"]]
    stations = {s["station_id"]: (int(s.get("piles", 4)), float(s.get("power_kw", CHARGE_KW)))
                for s in nm["charging_stations"]}
    return od, {int(h): v for h, v in sp.items()}, zones, stations


@dataclass
class Veh:
    vid: str
    av: bool
    soc: float
    loc: str
    status: str = "VACANT"          # VACANT | TO_ZONE | OCCUPIED | TO_STATION | QUEUED | CHARGING
    dest: Optional[str] = None
    t_free: float = 0.0             # time at which the current activity ends (or vacant since)
    km_empty: float = 0.0
    km_occ: float = 0.0
    stranded: bool = False
    since: float = 0.0              # start of the current status (min)
    time: dict = field(default_factory=dict)   # minutes spent in each status


@dataclass
class Req:
    zone: str
    t: float                        # creation time (min)


@dataclass
class Stats:
    waits: list = field(default_factory=list)        # (creation time, wait, served)
    charge_waits: list = field(default_factory=list)
    n_charges: int = 0
    n_human_assign: int = 0
    n_complied: int = 0


class ScaledDemand:
    """Taxi demand of the Chicago Loop multiplied by k (ride-hailing pickups in the Loop are about
    3.7 times taxi pickups; data/demand/loop_pickups_taxi_vs_tnp_20230405.json)."""

    def __init__(self, base, k: float = 1.0):
        self.base, self.k = base, k

    def demand_at_bin(self, zone_id, bin_idx, day_type="weekday", scale_to_5min=True):
        return self.k * self.base.demand_at_bin(zone_id, bin_idx, day_type, scale_to_5min)

    def uncertainty_at_bin(self, zone_id, bin_idx, kappa=1.5, day_type="weekday", scale_to_5min=True):
        return self.k * self.base.uncertainty_at_bin(zone_id, bin_idx, kappa, day_type, scale_to_5min)

    def sample_demand(self, zone_id, bin_idx, rng, day_type="weekday"):
        lam = self.demand_at_bin(zone_id, bin_idx, day_type)
        return int(rng.poisson(lam)) if lam > 0 else 0


class ShiftEnv:
    def __init__(self, demand_model, rng: np.random.Generator, fleet_size: int = 80,
                 av_penetration: float = 0.5, start_hour: int = 7, hours: float = 12.0,
                 warmup_min: float = 30.0, soc_init=(0.4, 0.9), network=None, deviation: str = "chase",
                 alpha_sd: float = 0.0, alpha_mean: float = 0.0, idle_humans: str = "stay"):
        self.od, self.speed, self.zone_ids, self.st_cfg = network or load_network()
        self.station_ids = list(self.st_cfg)
        self.dm = demand_model
        self.rng = rng
        self.t0 = start_hour * 60.0
        self.t = self.t0
        self.t_end = self.t0 + hours * 60.0
        self.measure_from = self.t0 + warmup_min
        self.measure_to = self.t_end - TIMEOUT_MIN          # every measured request is resolved
        self.requests: list[Req] = []
        self.piles = {s: [] for s in self.station_ids}      # end times of busy piles
        self.queue = {s: [] for s in self.station_ids}      # vids waiting, FIFO
        self.stats = Stats()
        self.deviation = deviation          # chase | stay | hotspot: what a non-compliant driver does
        n_av = round(fleet_size * av_penetration)
        locs = self.zone_ids
        self.veh = {}
        for i in range(fleet_size):
            v = Veh(f"taxi_{i:03d}", i < n_av, float(rng.uniform(*soc_init)), locs[i % len(locs)])
            v.t_free = v.since = self.t
            self.veh[v.vid] = v
        # hotspot drivers each return to a favourite zone, drawn by daily demand
        w = np.array([sum(demand_model.demand_at_bin(z, b) for b in range(28, 76)) for z in self.zone_ids])
        self.home = {vid: self.zone_ids[int(rng.choice(len(w), p=w / w.sum()))] for vid in self.veh}
        # "mix<p>": each driver is of the stay type with probability p, otherwise of the chase type
        p_stay = float(deviation[3:]) if deviation.startswith("mix") else None
        type_rng = np.random.default_rng(int(rng.integers(2**31)))
        self.dev_type = {vid: (("stay" if type_rng.random() < p_stay else "chase") if p_stay is not None
                               else deviation) for vid in self.veh}
        # persistent driver compliance propensity (drawn from type_rng: the main stream is unchanged)
        a_rng = np.random.default_rng(int(type_rng.integers(2**31)))
        self.alpha_sd = alpha_sd
        self.alpha = {vid: (0.0 if v.av else float(alpha_mean + alpha_sd * a_rng.standard_normal())) for vid, v in self.veh.items()}
        # what a vacant human driver without an order does: "stay" in his zone, or "self": take his
        # self-preferred action, as when he refuses an order
        self.idle_humans = idle_humans
        self.outcomes = []          # (vid, kind, dest, complied) of human orders that differ from the self action

    # ---------------- time and distance ----------------
    def km(self, a, b):
        return self.od[a][b]

    def tt(self, a, b, t=None):
        return self.tt_km(self.km(a, b), t)

    def tt_km(self, d_km, t=None):
        h = int(((self.t if t is None else t) // 60) % 24)
        return d_km / self.speed[h] * 60.0

    @property
    def step(self):
        return int(round((self.t - self.t0) / DT))

    @property
    def done(self):
        return self.t >= self.t_end - 1e-9

    def bin(self, t=None):
        return int(((self.t if t is None else t) // 15) % 96)

    # ---------------- simulation ----------------
    def advance(self):
        """Simulate (t, t + DT]: new requests, arrivals, charging, pickups, abandonment."""
        t1 = self.t + DT
        for z in self.zone_ids:
            lam = self.dm.demand_at_bin(z, self.bin(self.t))
            for _ in range(int(self.rng.poisson(lam)) if lam > 0 else 0):
                self.requests.append(Req(z, self.t + float(self.rng.uniform(0, DT))))
        self.requests.sort(key=lambda r: r.t)
        # vehicle events in time order; vacant vehicles pick up requests in their own zone
        while True:
            busy = [v for v in self.veh.values() if v.status in ("TO_ZONE", "OCCUPIED", "TO_STATION", "CHARGING")
                    and v.t_free <= t1]
            if busy:
                v = min(busy, key=lambda v: (v.t_free, v.vid))
                self._finish(v, v.t_free)
                continue
            picked = False
            for v in sorted(self.veh.values(), key=lambda v: (v.t_free, v.vid)):
                if v.status == "VACANT" and not self._must_charge(v):
                    picked |= self._pickup_here(v, horizon=t1)
            if not picked:
                break
        # abandonment
        keep = []
        for r in self.requests:
            if r.t + TIMEOUT_MIN <= t1:
                self._log_wait(r, TIMEOUT_MIN, False)
            else:
                keep.append(r)
        self.requests = keep
        self.t = t1

    def _mark(self, v: Veh, t):
        """Close the current status interval at time t."""
        v.time[v.status] = v.time.get(v.status, 0.0) + max(0.0, t - v.since)
        v.since = max(v.since, t)

    def _finish(self, v: Veh, t):
        self._mark(v, t)
        if v.status == "TO_ZONE":
            v.status, v.loc, v.dest = "VACANT", v.dest, None
            v.t_free = t
            if not self._must_charge(v):
                self._pickup_here(v, horizon=t + 1e-9, now=t)
        elif v.status == "OCCUPIED":
            v.status, v.loc, v.dest, v.t_free = "VACANT", v.dest, None, t
        elif v.status == "TO_STATION":
            v.loc, v.dest = v.dest, None
            self._join_station(v, t)
        elif v.status == "CHARGING":
            v.soc = max(v.soc, SOC_TARGET)
            v.status, v.t_free = "VACANT", t
            self.piles[v.loc] = [e for e in self.piles[v.loc] if e > t + 1e-9]
            self._start_queued(v.loc, t)

    def _pickup_here(self, v: Veh, horizon: float, now: Optional[float] = None):
        """Serve the oldest request in v's zone created before horizon."""
        for i, r in enumerate(self.requests):
            if r.zone == v.loc and r.t < horizon:
                ready = max(r.t, v.t_free if now is None else now)
                start = ready + self.tt_km(INTRA_ZONE_KM, ready)
                if start - r.t > TIMEOUT_MIN:
                    continue                          # the passenger has already left
                self.requests.pop(i)
                self._mark(v, ready)
                v.time["PICKUP"] = v.time.get("PICKUP", 0.0) + (start - ready)
                v.since = start
                self._drive(v, INTRA_ZONE_KM, occupied=False)
                self._log_wait(r, start - r.t, True)
                dest = self._destination(r.t)
                d = self.km(v.loc, dest)
                self._drive(v, d, occupied=True)
                v.status, v.dest, v.t_free = "OCCUPIED", dest, start + self.tt(v.loc, dest, start)
                return True
        return False

    def _join_station(self, v: Veh, t):
        s = v.loc
        n, _ = self.st_cfg[s]
        self.piles[s] = [e for e in self.piles[s] if e > t + 1e-9]
        v.t_free = t                       # arrival time at the station
        if len(self.piles[s]) < n and not self.queue[s]:
            self._start_charge(v, t)
        else:
            v.status = "QUEUED"
            self.queue[s].append(v.vid)

    def _start_charge(self, v: Veh, t):
        s = v.loc
        _, kw = self.st_cfg[s]
        if t >= self.measure_from:
            self.stats.charge_waits.append(t - v.t_free)
            self.stats.n_charges += 1
        dur = max(0.0, SOC_TARGET - v.soc) * BATTERY_KWH / kw * 60.0
        self._mark(v, t)
        v.status, v.t_free = "CHARGING", t + dur
        self.piles[s].append(t + dur)

    def _start_queued(self, s, t):
        n, _ = self.st_cfg[s]
        while self.queue[s] and len(self.piles[s]) < n:
            self._start_charge(self.veh[self.queue[s].pop(0)], t)

    def _destination(self, t):
        w = np.array([self.dm.demand_at_bin(z, self.bin(t)) for z in self.zone_ids], float)
        if w.sum() <= 0:
            return self.zone_ids[int(self.rng.integers(len(self.zone_ids)))]
        return self.zone_ids[int(self.rng.choice(len(w), p=w / w.sum()))]

    def _drive(self, v: Veh, d_km: float, occupied: bool):
        if occupied:
            v.km_occ += d_km
        else:
            v.km_empty += d_km
        v.soc -= d_km * CONSUMPTION_KWH_PER_KM / BATTERY_KWH
        if v.soc <= 0 and not v.stranded:
            v.stranded = True
        v.soc = max(v.soc, 0.0)

    def _log_wait(self, r: Req, w: float, served: bool):
        if self.measure_from <= r.t < self.measure_to:
            self.stats.waits.append((r.t, w, served))

    def _must_charge(self, v: Veh):
        return v.soc < SOC_MUST_CHARGE

    # ---------------- interface for dispatchers ----------------
    def vacant(self):
        return [v for v in self.veh.values() if v.status == "VACANT"]

    def snapshot(self, vids=None) -> FleetSnapshot:
        snap = FleetSnapshot(step=self.step, sim_time=(self.t - self.t0) * 60.0)
        for v in self.vacant():
            if vids is None or v.vid in vids:
                snap.vehicles[v.vid] = VehicleState(
                    vehicle_id=v.vid, vehicle_type=VehicleType.AV if v.av else VehicleType.HUMAN,
                    node_id=v.loc, soc=v.soc, status=OperationalStatus.VACANT,
                    battery_capacity_kwh=BATTERY_KWH, consumption_rate=CONSUMPTION_KWH_PER_KM,
                    max_charge_power_kw=CHARGE_KW, shift_remaining_min=self.t_end - self.t)
        for s in self.station_ids:
            n, kw = self.st_cfg[s]
            busy = [e - self.t for e in self.piles[s] if e > self.t]
            queued = [max(0.0, SOC_TARGET - self.veh[q].soc) * BATTERY_KWH / kw * 60.0 for q in self.queue[s]]
            enroute = [(v.t_free - self.t,
                        max(0.0, SOC_TARGET - v.soc) * BATTERY_KWH / kw * 60.0)
                       for v in self.veh.values() if v.status == "TO_STATION" and v.dest == s]
            snap.stations[s] = StationState(station_id=s, node_id=s, num_piles=n, power_kw=kw,
                                            occupied_piles=min(n, len(busy) + len(queued)),
                                            remaining_charge_times=sorted(busy) + queued,
                                            in_service=sorted(busy),
                                            committed=[(0.0, d) for d in queued] + enroute)
        return snap

    def pending_demand(self):
        out = {z: 0.0 for z in self.zone_ids}
        for r in self.requests:
            out[r.zone] += 1.0
        return out

    def travel_time_tables(self, vehicles):
        tz, ts = {}, {}
        for vs in vehicles:
            a = self.veh[vs.vehicle_id].loc
            for z in self.zone_ids:
                tz[(vs.vehicle_id, z)] = self.tt(a, z)
            for s in self.station_ids:
                ts[(vs.vehicle_id, s)] = self.tt(a, s)
        return tz, ts

    def self_preferred(self, v: Veh):
        """('station', sid) or ('zone', zid): what the driver does without a recommendation."""
        if self._must_charge(v):
            return "station", min(self.station_ids, key=lambda s: self.tt(v.loc, s))
        kind = self.dev_type[v.vid]
        if kind == "stay":
            return "zone", v.loc
        if kind == "hotspot":
            return "zone", self.home[v.vid]
        pend = self.pending_demand()           # same rule as LogitComplianceModel.self_preferred_action
        if not any(pend.values()):
            return "zone", v.loc
        return "zone", max(self.zone_ids, key=lambda z: pend[z] / max(self.tt(v.loc, z), 0.1))

    def apply(self, zone_assign: dict, station_assign: dict, pi: dict):
        """pi[vid] = (pi_zone dict, pi_station dict) for humans; AVs always comply."""
        handled = set()
        for vid in sorted(set(zone_assign) | set(station_assign)):
            v = self.veh[vid]
            if v.status != "VACANT":
                continue
            handled.add(vid)
            target = (("station", station_assign[vid]) if station_assign.get(vid)
                      else ("zone", zone_assign[vid]) if zone_assign.get(vid) else None)
            if target is None:
                continue
            if target[0] == "zone" and self._must_charge(v):
                target = None                      # infeasible order: treated as not followed
            if not v.av:
                self.stats.n_human_assign += 1
                p = 0.0 if target is None else pi.get(vid, ({}, {}))[0 if target[0] == "zone" else 1].get(target[1], 1.0)
                if 0.0 < p < 1.0 and self.alpha[vid]:
                    p = 1.0 / (1.0 + math.exp(-(math.log(p / (1.0 - p)) + self.alpha[vid])))
                own = self.self_preferred(v)
                ok = self.rng.random() < p
                if target is not None and tuple(target) != own:
                    self.outcomes.append((vid, target[0], target[1], ok))
                if ok:
                    self.stats.n_complied += 1
                else:
                    target = own
            elif target is None:
                target = self.self_preferred(v)
            self._go(v, *target)
        # vehicles that must charge and received no order go to the nearest station
        for v in self.vacant():
            if self._must_charge(v):
                self._go(v, *self.self_preferred(v))
            elif self.idle_humans == "self" and not v.av and v.vid not in handled:
                self._go(v, *self.self_preferred(v))

    def _go(self, v: Veh, kind: str, dest: str):
        self._mark(v, self.t)
        if kind == "zone":
            if dest == v.loc:
                self._pickup_here(v, horizon=self.t, now=self.t)
                return
            self._drive(v, self.km(v.loc, dest), occupied=False)
            v.status, v.dest, v.t_free = "TO_ZONE", dest, self.t + self.tt(v.loc, dest)
        else:
            if dest == v.loc:
                self._join_station(v, self.t)
                return
            self._drive(v, self.km(v.loc, dest), occupied=False)
            v.status, v.dest, v.t_free = "TO_STATION", dest, self.t + self.tt(v.loc, dest)

    # ---------------- metrics ----------------
    def metrics(self, windows=((7, 11), (11, 15), (15, 19))) -> dict:
        for v in self.veh.values():
            self._mark(v, self.t_end)
        tot = {}
        for v in self.veh.values():
            for k, m in v.time.items():
                tot[k] = tot.get(k, 0.0) + m
        fleet_min = sum(tot.values())
        w = np.array([x[1] for x in self.stats.waits]) if self.stats.waits else np.zeros(1)
        served = np.array([x[2] for x in self.stats.waits], bool) if self.stats.waits else np.zeros(1, bool)
        km_e = sum(v.km_empty for v in self.veh.values())
        km_o = sum(v.km_occ for v in self.veh.values())
        out = {
            "n_requests": int(len(self.stats.waits)),
            "wait_all_min": float(w.mean()),            # abandoned requests count as TIMEOUT_MIN
            "wait_p95_min": float(np.percentile(w, 95)),
            "wait_served_min": float(w[served].mean()) if served.any() else float("nan"),
            "service_rate": float(served.mean()),
            "empty_km_share": km_e / max(1e-9, km_e + km_o),
            "n_charges": self.stats.n_charges,
            "charge_wait_min": float(np.mean(self.stats.charge_waits)) if self.stats.charge_waits else 0.0,
            "n_stranded": sum(v.stranded for v in self.veh.values()),
            "compliance_rate": self.stats.n_complied / max(1, self.stats.n_human_assign),
            "soc_end_mean": float(np.mean([v.soc for v in self.veh.values()])),
            # fleet time use over the shift (vehicle-hours) and shares of fleet time
            "veh_h_queue": tot.get("QUEUED", 0.0) / 60.0,
            "veh_h_charging": tot.get("CHARGING", 0.0) / 60.0,
            "veh_h_to_station": tot.get("TO_STATION", 0.0) / 60.0,
            "share_occupied": tot.get("OCCUPIED", 0.0) / fleet_min,
            "share_empty_driving": (tot.get("TO_ZONE", 0.0) + tot.get("PICKUP", 0.0)) / fleet_min,
            "share_idle": tot.get("VACANT", 0.0) / fleet_min,
            "share_charging_total": (tot.get("QUEUED", 0.0) + tot.get("CHARGING", 0.0) + tot.get("TO_STATION", 0.0)) / fleet_min,
        }
        for a, b in windows:
            sel = [x for x in self.stats.waits if a * 60 <= x[0] < b * 60]
            if sel:
                out[f"wait_all_{a:02d}_{b:02d}"] = float(np.mean([x[1] for x in sel]))
                out[f"service_{a:02d}_{b:02d}"] = float(np.mean([x[2] for x in sel]))
        return out
