"""
run_eval.py — evaluation of dispatch methods over a full day shift (env/shift_sim.py).

Every method sees the same demand realization and driver compliance draws for a given seed.
Gurobi is used with one thread per worker process.

  python -m chicago_loop.experiments.run_eval --methods self rob_mpc_nc ermpc --seeds 20
"""

from __future__ import annotations

import os

os.environ.setdefault("ERMPC_GRB_THREADS", "1")

import argparse
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import math

import numpy as np
from scipy import stats

from chicago_loop.env.shift_sim import DT, ScaledDemand, ShiftEnv, load_network
from chicago_loop.env.demand_model import DemandModel, SimpleDemandForecaster
from chicago_loop.env.fleet import VehicleType
from chicago_loop.models.compliance import ChargeAction, ComplianceParams, LogitComplianceModel, RepositionAction
from chicago_loop.models.optimizer import DispatchInput, ERMPCOptimizer

logging.disable(logging.INFO)
MAX_BATCH = 40          # vehicles per MILP (size-limited Gurobi license: 2,000 variables)
THETA_RANGE = (1.0, 2.5)  # behavioural uncertainty behind the compliance uncertainty set
OPT = dict(solver="gurobi", time_limit=10.0, mip_gap=0.02)
STATION_SUFFIX = {"_bc": "wait", "_bs": "wait_seq", "_bh": "wait_seq_h", "_bo": "wait", "_sm": "static_mc", "_sh": "horizon", "_nr": "nearest",
                  "_na": "available", "_sq": "queue"}


def make_opt(method, idle_humans="stay"):
    """(optimizer, uses compliance coefficients).

    Base methods: det_mpc, rob_mpc_nc (compliance-blind), naive_plugin (compliance, no demand
    margin), ermpc (compliance-aware), ermpc_rpi[G] (plus Γ_π = G, default 2). The suffix "_bc"
    replaces the static charging-capacity constraint by the backlog-predicted station wait; "_bo"
    does the same with a wait predicted from the vehicles at the station only (no vehicles en route).
    Station-choice baselines (STATION_SUFFIX): "_sm" the one-interval capacity constraint with every
    vehicle that must charge assigned a station, "_sh" capacity over one full session, "_nr" nearest
    station, "_na" nearest station with an idle charger, "_sq" shortest-queue estimate; "_bs" is the
    backlog wait with the vehicles of one step assigned sequentially, "_bh" the same with human
    drivers sent first to the station they would choose themselves. Without a
    suffix, vehicles that the capacity constraint leaves without a station drive to the nearest one.
    Driver heterogeneity (run_one): ermpc_pool estimates one common compliance intercept online,
    ermpc_ln one intercept per driver, ermpc_or uses the true per-driver propensity (oracle).
    Diagnostics: ermpc_nofb (no fallback credit), ermpc_fbonly (fallback credit, no discount),
    ermpc_flat (each driver's zone-order probabilities replaced by their mean).
    Re-issued orders: ermpc_ms[K] replaces each zone-order compliance probability by the
    probability that the order is followed within K decision epochs (default 3), with the
    re-issue rate r learned online; ermpc_msf[K] assumes r = 1.
    """
    if method == "self":
        return None, False
    rule = STATION_SUFFIX.get(method[-3:])
    base, bc = (method[:-3], True) if rule else (method, False)
    common = dict(delta=0.3, idle_to_fallback=idle_humans == "self", **OPT)
    if bc:
        common.update(station_rule=rule, charge_wait_weight=3.0)
    if base == "det_mpc":
        return ERMPCOptimizer(demand_margin=0.0, gamma_d=0.0, gamma_c=0.0, **{**common, "delta": 0.0}), False
    if base == "det_mpc_pi":                    # deterministic MPC with compliance probabilities
        return ERMPCOptimizer(demand_margin=0.0, gamma_d=0.0, gamma_c=0.0, **{**common, "delta": 0.0}), True
    if base == "rob_mpc_nc":
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=0.0, **common), False
    if base == "rob_mpc_nc_c2":                 # ablation: charging robustness without compliance modelling
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=2.0, **common), False
    if base == "nc_m0":                         # compliance-blind with look-ahead, no demand margin
        return ERMPCOptimizer(demand_margin=0.0, gamma_d=0.0, gamma_c=0.0, **common), False
    if base == "naive_plugin":
        return ERMPCOptimizer(demand_margin=0.0, gamma_d=0.0, gamma_c=0.0, **common), True
    if base == "ermpc":
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=0.0 if bc else 2.0, **common), True
    if base in ("ermpc_nofb", "ermpc_fbonly"):     # diagnostics: drop one part of the expected supply
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=0.0 if bc else 2.0,
                              fallback_credit=base == "ermpc_fbonly", pi_discount=base == "ermpc_nofb", **common), True
    if base in ("ermpc_pool", "ermpc_ln", "ermpc_or", "ermpc_flat") or base.startswith("ermpc_ms"):
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=0.0 if bc else 2.0, **common), True
    if base == "ermpc_c0":                      # ablation: compliance modelling without charging robustness
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=0.0, **common), True
    if base.startswith("ermpc_rpi"):
        g = float(base[9:] or 2)
        return ERMPCOptimizer(demand_margin=1.0, gamma_d=1.0, gamma_c=0.0 if bc else 2.0, gamma_pi=g,
                              **common), True
    raise ValueError(method)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class InterceptLearner:
    """Online MAP estimate of a compliance intercept a in pi = sigmoid(logit(pi_pop) + a), prior
    a ~ N(0, PRIOR_SD^2), from the orders a driver followed or refused. Predictions integrate over
    the Laplace posterior (probit approximation). key(vid) pools drivers ("pool") or not ("ln")."""
    PRIOR_SD = 1.5

    def __init__(self, pooled):
        self.pooled, self.obs, self.post = pooled, {}, {}

    def key(self, vid):
        return "all" if self.pooled else vid

    def add(self, vid, z, y):
        self.obs.setdefault(self.key(vid), []).append((z, y))
        self.post.pop(self.key(vid), None)

    def posterior(self, vid):
        k = self.key(vid)
        if k not in self.post:
            a, h = 0.0, 1.0 / self.PRIOR_SD ** 2
            obs = self.obs.get(k, [])
            for _ in range(20):
                g, h = -a / self.PRIOR_SD ** 2, 1.0 / self.PRIOR_SD ** 2
                for z, y in obs:
                    q = _sig(z + a)
                    g += y - q
                    h += q * (1 - q)
                a += g / h
                if abs(g / h) < 1e-6:
                    break
            self.post[k] = (a, 1.0 / h)
        return self.post[k]

    def adjust(self, vid, pi):
        a, v = self.posterior(vid)
        s = math.sqrt(1.0 + math.pi * v / 8.0)
        return {d: _sig((_logit(p) + a) / s) for d, p in pi.items()}


def multi_step(pi, r, K):
    """P(order followed within K epochs) when a refused order is re-issued with probability r:
    pi * sum_{k<K} ((1 - pi) r)^k."""
    q = (1.0 - pi) * r
    return pi * (K if q >= 1.0 else (1.0 - q ** K) / (1.0 - q))


def parse_ms(method):
    """(K, fixed r) for ermpc_ms[f][K] methods, else None."""
    base = method[:-3] if method[-3:] in STATION_SUFFIX else method
    if not base.startswith("ermpc_ms"):
        return None
    rest = base[8:]
    fixed = rest.startswith("f")
    digits = rest[1:] if fixed else rest
    return int(digits or 3), fixed


def run_one(method, seed, av_rate=0.5, fleet_size=80, theta=1.5, theta_true=None, demand_k=1.0,
            network=None, dm=None, alpha_sd=0.0, alpha_mean=0.0, idle_humans="stay", piles=None, **env_kw):
    rng = np.random.default_rng(seed)
    if piles is not None:                   # charger-scarcity scenario: piles per station
        od_, sp_, z_, st_ = network or load_network()
        network = (od_, sp_, z_, {k: (int(piles), kw) for k, (_, kw) in st_.items()})
    dm = ScaledDemand(dm or DemandModel(), demand_k)
    env = ShiftEnv(dm, rng, fleet_size=fleet_size, av_penetration=av_rate, network=network, alpha_sd=alpha_sd,
                   alpha_mean=alpha_mean, idle_humans=idle_humans,
                   **env_kw)
    start_bin = env.bin()
    fc = SimpleDemandForecaster(dm, env.zone_ids, smoothing=0.2)
    comp = LogitComplianceModel(ComplianceParams(theta=theta))                  # planner's model
    comp_true = LogitComplianceModel(ComplianceParams(theta=theta if theta_true is None else theta_true))
    comp_range = [LogitComplianceModel(ComplianceParams(theta=t)) for t in THETA_RANGE]
    opt, uses_pi = make_opt(method, idle_humans)
    learner = (InterceptLearner(pooled=method.startswith("ermpc_pool")) if method.startswith(("ermpc_pool", "ermpc_ln"))
               else None)
    n_seen = 0              # env.outcomes already passed to the learner
    pop_pi = {}             # vid -> population (pi_zone, pi_station) of its last planned orders
    ms = parse_ms(method)
    dg = dict(n_h=0, same=0, move=0, pi=0.0, tt=0.0, n_av=0, move_av=0)   # zone orders actually issued
    cap = dict(steps=0, capped=0, mc=0, mc_out=0)   # dispatch steps; steps over MAX_BATCH; vehicles that must charge
    reissue = [1.0, 1.0]    # Beta(1, 1) counts: refused zone orders re-issued at the next epoch / not
    refused_prev = set()    # human drivers who refused a zone order at the previous epoch
    batch_rng = np.random.default_rng(seed + 7919)
    solve_s = 0.0
    while not env.done:
        vac = env.vacant()
        if vac:
            if method == "self":
                za, sa = {}, {}
                for v in vac:
                    kind, dest = env.self_preferred(v)
                    (za if kind == "zone" else sa)[v.vid] = dest
                env.apply(za, sa, {v.vid: ({dest: 1.0 for dest in env.zone_ids},
                                           {s: 1.0 for s in env.station_ids}) for v in vac})
            else:
                ids = [v.vid for v in vac]
                cap["steps"] += 1
                if len(ids) > MAX_BATCH:
                    cap["capped"] += 1
                    ids = list(batch_rng.choice(ids, MAX_BATCH, replace=False))
                mc = [v.vid for v in vac if env._must_charge(v)]
                cap["mc"] += len(mc)
                cap["mc_out"] += sum(1 for vid in mc if vid not in ids)
                for vid in refused_prev:
                    reissue[0 if vid in ids else 1] += 1.0
                refused_prev = set()
                snap = env.snapshot(set(ids))
                if method.endswith("_bo"):          # ablation: ignore vehicles driving to the station
                    for st in snap.stations.values():
                        st.committed = [(eta, d) for eta, d in st.committed if eta <= 0.0]
                vs = list(snap.vehicles.values())
                tz, ts = env.travel_time_tables(vs)
                step = env.step
                d_now = env.pending_demand()
                d_hat = fc.forecast(step, start_bin)
                d_next = fc.forecast_next(step, start_bin)
                xi = fc.uncertainty(step, start_bin)
                fc.update(d_now)
                coeffs, true_pi = {}, {}
                for v in vs:
                    kw = dict(vehicle=v, zone_ids=env.zone_ids, station_ids=env.station_ids,
                              zone_demand_density=d_now,
                              travel_times_to_zones={z: tz[(v.vehicle_id, z)] for z in env.zone_ids},
                              travel_times_to_stations={s: ts[(v.vehicle_id, s)] for s in env.station_ids})
                    c = comp.effective_supply_coefficients(**kw)
                    kind, dest = env.self_preferred(env.veh[v.vehicle_id])   # where a deviating driver goes
                    c["self_action"] = RepositionAction(dest) if kind == "zone" else ChargeAction(dest)
                    if method.startswith("ermpc_rpi") and v.vehicle_type == VehicleType.HUMAN:
                        rng_pi = [m.effective_supply_coefficients(**kw)["pi_zone"] for m in comp_range]
                        c["eps_zone"] = {z: max(abs(r[z] - c["pi_zone"][z]) for r in rng_pi) for z in env.zone_ids}
                    if v.vehicle_type == VehicleType.HUMAN:
                        pop_pi[v.vehicle_id] = (dict(c["pi_zone"]), dict(c["pi_station"]))
                        if learner is not None:
                            c["pi_zone"] = learner.adjust(v.vehicle_id, c["pi_zone"])
                            c["pi_station"] = learner.adjust(v.vehicle_id, c["pi_station"])
                        elif method.startswith("ermpc_or"):
                            a = env.alpha[v.vehicle_id]
                            c["pi_zone"] = {d: _sig(_logit(p) + a) for d, p in c["pi_zone"].items()}
                            c["pi_station"] = {d: _sig(_logit(p) + a) for d, p in c["pi_station"].items()}
                    if method.startswith("ermpc_flat") and v.vehicle_type == VehicleType.HUMAN:
                        mz = float(np.mean(list(c["pi_zone"].values())))
                        c["pi_zone"] = {d: mz for d in c["pi_zone"]}
                    if ms is not None and v.vehicle_type == VehicleType.HUMAN:
                        K, fixed = ms
                        r = 1.0 if fixed else reissue[0] / (reissue[0] + reissue[1])
                        c["pi_zone"] = {d: multi_step(p, r, K) for d, p in c["pi_zone"].items()}
                    coeffs[v.vehicle_id] = c
                    t = comp_true.effective_supply_coefficients(**kw)   # population pi; env adds alpha_i
                    true_pi[v.vehicle_id] = (t["pi_zone"], t["pi_station"])
                inp = DispatchInput(vehicles=vs, stations=list(snap.stations.values()), zone_ids=env.zone_ids,
                                    demand_forecast=d_hat, demand_forecast_next=d_next, demand_uncertainty=xi,
                                    travel_time_to_zone=tz, travel_time_to_station=ts,
                                    compliance_coeffs=coeffs if uses_pi else {}, delta_t=DT)
                t0 = time.perf_counter()
                res = opt.solve(inp)
                solve_s += time.perf_counter() - t0
                for v in vs:
                    z = res.zone_assignments.get(v.vehicle_id)
                    if not z or res.station_assignments.get(v.vehicle_id):
                        continue
                    moved = z != env.veh[v.vehicle_id].loc
                    if v.vehicle_type == VehicleType.AV:
                        dg["n_av"] += 1
                        dg["move_av"] += moved
                        continue
                    kind, dest = env.self_preferred(env.veh[v.vehicle_id])
                    dg["n_h"] += 1
                    dg["same"] += kind == "zone" and dest == z
                    dg["move"] += moved
                    dg["pi"] += pop_pi[v.vehicle_id][0][z]
                    dg["tt"] += tz[(v.vehicle_id, z)]
                env.apply(res.zone_assignments, res.station_assignments, true_pi)
                if learner is not None:
                    for vid, kind, dest, ok in env.outcomes[n_seen:]:
                        if vid in pop_pi:
                            learner.add(vid, _logit(pop_pi[vid][0 if kind == "zone" else 1][dest]), float(ok))
                if ms is not None:
                    refused_prev = {vid for vid, kind, _, ok in env.outcomes[n_seen:] if kind == "zone" and not ok}
                n_seen = len(env.outcomes)
        env.advance()
    m = env.metrics()
    m.update(method=method, seed=seed, av_rate=av_rate, fleet_size=fleet_size, theta=theta, demand_k=demand_k,
             theta_true=theta if theta_true is None else theta_true, deviation=env.deviation, alpha_sd=alpha_sd,
             alpha_mean=alpha_mean, idle_humans=idle_humans, piles=piles,
             diag_same_as_fallback=dg["same"] / max(1, dg["n_h"]), diag_move_h=dg["move"] / max(1, dg["n_h"]),
             diag_pi_order=dg["pi"] / max(1, dg["n_h"]), diag_tt_order=dg["tt"] / max(1, dg["n_h"]),
             diag_move_av=dg["move_av"] / max(1, dg["n_av"]), diag_n_orders_h=dg["n_h"],
             solve_s=round(solve_s, 1), n_dispatch_steps=cap["steps"],
             share_steps_capped=cap["capped"] / max(1, cap["steps"]), n_must_charge=cap["mc"],
             n_must_charge_unbatched=cap["mc_out"], reissue_rate=reissue[0] / sum(reissue) if ms is not None else None)
    return m


_CACHE = {}


def _job(args):
    method, seed, kw = args
    if not _CACHE:
        _CACHE["net"], _CACHE["dm"] = load_network(), DemandModel()
    return run_one(method, seed, network=_CACHE["net"], dm=_CACHE["dm"], **kw)


def summarize(R, ref="rob_mpc_nc", keys=("wait_all_min", "service_rate", "wait_served_min", "wait_p95_min",
                                         "n_charges", "charge_wait_min", "n_stranded", "compliance_rate")):
    gk = lambda r: (r["av_rate"], r["fleet_size"], r.get("demand_k", 1.0), r.get("deviation", "chase"), r.get("alpha_sd", 0.0))
    for g in sorted({gk(r) for r in R}):
        sel = [r for r in R if gk(r) == g]
        print(f"\n=== rho={g[0]}, fleet={g[1]}, demand x{g[2]}, deviation={g[3]}, alpha_sd={g[4]} ===")
        print(f"{'method':<14}" + "".join(f"{k[:14]:>15}" for k in keys) + f"{'d_wait vs '+ref:>22}")
        base = {r["seed"]: r["wait_all_min"] for r in sel if r["method"] == ref}
        for m in dict.fromkeys(r["method"] for r in sel):
            x = [r for r in sel if r["method"] == m]
            row = f"{m:<14}" + "".join(f"{np.mean([r[k] for r in x]):15.3f}" for k in keys)
            d = np.array([r["wait_all_min"] - base[r["seed"]] for r in x if r["seed"] in base])
            if len(d) > 1:
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                row += f"{d.mean():+12.2f} ± {h:5.2f}"
            print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["self", "det_mpc", "rob_mpc_nc", "naive_plugin", "ermpc"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--av_rates", nargs="+", type=float, default=[0.5])
    ap.add_argument("--fleet", nargs="+", type=int, default=[80])
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--demand_k", nargs="+", type=float, default=[1.0])
    ap.add_argument("--deviation", nargs="+", default=["chase"],
                    help="chase | stay | hotspot | mix<p> (share p of drivers stay)")
    ap.add_argument("--theta_true", type=float, default=None)
    ap.add_argument("--alpha_sd", nargs="+", type=float, default=[0.0],
                    help="s.d. of the persistent per-driver compliance propensity")
    ap.add_argument("--idle_humans", nargs="+", default=["stay"],
                    help="stay | self: what a human driver without an order does")
    ap.add_argument("--piles", nargs="+", type=int, default=[None], help="chargers per station (default: network)")
    ap.add_argument("--alpha_mean", nargs="+", type=float, default=[0.0],
                    help="common shift of the compliance logit (negative: less compliant drivers)")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()
    jobs = [(m, s, dict(av_rate=a, fleet_size=f, hours=args.hours, demand_k=k, deviation=dv,
                        theta_true=args.theta_true, alpha_sd=sd, alpha_mean=am, idle_humans=ih, piles=pl))
            for pl in args.piles for ih in args.idle_humans for am in args.alpha_mean for sd in args.alpha_sd for dv in args.deviation for k in args.demand_k for a in args.av_rates for f in args.fleet
            for m in args.methods for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        R = pool.map(_job, jobs, chunksize=1)
    out = Path("chicago_loop/results") / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(R, indent=1))
    print(f"saved {len(R)} records to {out}")
    summarize(R)
