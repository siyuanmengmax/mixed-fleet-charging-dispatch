"""
Summary tables of the result files in chicago_loop/results, with paired 95% t-intervals
over seeds. Waits count abandoned requests as 10 min.

  python -m chicago_loop.experiments.summarize_final
"""

from __future__ import annotations

import glob
import os
import json

import numpy as np
from scipy import stats

RES = "chicago_loop/results"
# main runs: human drivers without an order take their own action (prefix "s_");
# PB_PREFIX="" gives the sensitivity runs in which they stay in place
PREFIX = os.environ.get("PB_PREFIX", "s_")


def load(tag):
    files = sorted(glob.glob(f"{RES}/{PREFIX if not tag.startswith(('cf_', 'st_')) else ''}{tag}_2*.json"))
    return json.load(open(files[-1])) if files else []


def paired(R, a, b, key="wait_all_min", **sel):
    x = {r["seed"]: r[key] for r in R if r["method"] == a and all(r.get(k) == v for k, v in sel.items())}
    y = {r["seed"]: r[key] for r in R if r["method"] == b and all(r.get(k) == v for k, v in sel.items())}
    d = np.array([x[s] - y[s] for s in x if s in y])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), h


def mean(R, m, key, **sel):
    return float(np.mean([r[key] for r in R if r["method"] == m and all(r.get(k) == v for k, v in sel.items())]))


def fmt(mu, h):
    return f"{mu:+.2f}±{h:.2f}{'*' if abs(mu) > h else ' '}"


def main_tables():
    R = load("final_main_k3") + load("final_main_k2")
    # The static-capacity variants of these runs used q̄ = 18 min (superseded); station choice is
    # compared in station_table() on the st_* runs.
    methods = ["self", "rob_mpc_nc_bc", "ermpc_bc", "ermpc_rpi_bc"]
    settings = sorted({(r["demand_k"], r["fleet_size"]) for r in R}, reverse=True)
    print("\nMAIN: mean wait (min) / service rate / charging queue (min per session)")
    for k, f in settings:
        for dv in ("chase", "stay", "hotspot"):
            sel = dict(demand_k=k, fleet_size=f, deviation=dv)
            row = f"x{k:g} {f} {dv:7}"
            for m in methods:
                row += f" {m}={mean(R, m, 'wait_all_min', **sel):.2f}/{mean(R, m, 'service_rate', **sel):.3f}/q{mean(R, m, 'charge_wait_min', **sel):.1f}"
            print(row)
    print("\nMAIN: paired differences in mean wait")
    pairs = [("rob_mpc_nc_bc", "self"), ("ermpc_bc", "rob_mpc_nc_bc"), ("ermpc_rpi_bc", "ermpc_bc")]
    for k, f in settings:
        for dv in ("chase", "stay", "hotspot"):
            sel = dict(demand_k=k, fleet_size=f, deviation=dv)
            print(f"x{k:g} {f} {dv:7}" + "".join(f" | {a}-{b} {fmt(*paired(R, a, b, **sel))}" for a, b in pairs))


def misspec_table():
    R = load("final_misspec_t0.5") + load("final_misspec_t3.0")
    if not R:
        return
    print("\nMISSPECIFICATION (planner theta = 1.5; robust set from theta in [1.0, 2.5])")
    for tt in (0.5, 3.0):
        for f in (120, 160):
            for dv in ("chase", "stay"):
                sel = dict(theta_true=tt, fleet_size=f, deviation=dv)
                print(f"theta_true={tt} {f} {dv:5}"
                      + "".join(f" | {a}-{b} {fmt(*paired(R, a, b, **sel))}" for a, b in
                                [("ermpc_bc", "rob_mpc_nc_bc"), ("ermpc_rpi_bc", "ermpc_bc"), ("ermpc_rpi4_bc", "ermpc_bc")]))


def rho_table():
    R = load("final_rho")
    if not R:
        return
    print("\nAV SHARE (x3, 160 vehicles)")
    for dv in ("chase", "stay"):
        for rho in sorted({r["av_rate"] for r in R}):
            sel = dict(av_rate=rho, deviation=dv)
            print(f"{dv:5} rho={rho:.1f} self={mean(R, 'self', 'wait_all_min', **sel):.2f} "
                  f"nc={mean(R, 'rob_mpc_nc_bc', 'wait_all_min', **sel):.2f} er={mean(R, 'ermpc_bc', 'wait_all_min', **sel):.2f}"
                  f" | er-nc {fmt(*paired(R, 'ermpc_bc', 'rob_mpc_nc_bc', **sel))}"
                  f" | nc-self {fmt(*paired(R, 'rob_mpc_nc_bc', 'self', **sel))}")


def mix_table():
    R = load("final_mix")
    if not R:
        return
    print("\nSHARE OF STAY-TYPE DRIVERS (x3)")
    for f in (120, 160):
        for p in ("mix0", "mix0.25", "mix0.5", "mix0.75", "mix1"):
            sel = dict(fleet_size=f, deviation=p)
            print(f"{f} {p:8} nc={mean(R, 'rob_mpc_nc_bc', 'wait_all_min', **sel):.2f} "
                  f"er={mean(R, 'ermpc_bc', 'wait_all_min', **sel):.2f} | er-nc {fmt(*paired(R, 'ermpc_bc', 'rob_mpc_nc_bc', **sel))}")


def het_table(tag="het_main", title="alpha_i ~ N(0, sd^2) (x3, rho = 0.5)"):
    R = load(tag)
    if not R:
        return
    print(f"\nPERSISTENT DRIVER PROPENSITY {title}")
    ms = ["rob_mpc_nc_bc", "ermpc_bc", "ermpc_pool_bc", "ermpc_ln_bc", "ermpc_or_bc"]
    pairs = [("ermpc_bc", "rob_mpc_nc_bc"), ("ermpc_pool_bc", "rob_mpc_nc_bc"), ("ermpc_ln_bc", "rob_mpc_nc_bc"),
             ("ermpc_or_bc", "rob_mpc_nc_bc"), ("ermpc_ln_bc", "ermpc_pool_bc"), ("ermpc_or_bc", "ermpc_pool_bc")]
    for f in sorted({r["fleet_size"] for r in R}):
        for dv in ("chase", "stay"):
            for sd in sorted({r["alpha_sd"] for r in R}):
                sel = dict(fleet_size=f, deviation=dv, alpha_sd=sd)
                print(f"{f} {dv:5} sd={sd:g} " + " ".join(f"{m[:-3]}={mean(R, m, 'wait_all_min', **sel):.2f}" for m in ms)
                      + "".join(f" | {a[:-3]}-{b[:-3]} {fmt(*paired(R, a, b, **sel))}" for a, b in pairs))


def level_table():
    R = load("het_level")
    if not R:
        return
    print("\nCOMPLIANCE LEVEL: common logit shift alpha_mean (x3, 160 vehicles, all human)")
    ms = ["self", "rob_mpc_nc_bc", "ermpc_bc", "ermpc_pool_bc", "ermpc_ln_bc", "ermpc_or_bc"]
    pairs = [("rob_mpc_nc_bc", "self"), ("ermpc_bc", "rob_mpc_nc_bc"), ("ermpc_pool_bc", "rob_mpc_nc_bc"),
             ("ermpc_ln_bc", "rob_mpc_nc_bc"), ("ermpc_or_bc", "rob_mpc_nc_bc")]
    for dv in ("chase", "stay"):
        for am in sorted({r["alpha_mean"] for r in R}):
            sel = dict(deviation=dv, alpha_mean=am)
            print(f"{dv:5} a={am:+g} comply={mean(R, 'rob_mpc_nc_bc', 'compliance_rate', **sel):.2f} "
                  + " ".join(f"{m.replace('_bc', '')}={mean(R, m, 'wait_all_min', **sel):.2f}" for m in ms)
                  + "".join(f" | {a.replace('_bc', '')}-{b.replace('_bc', '')} {fmt(*paired(R, a, b, **sel))}" for a, b in pairs))


def ms_tables():
    """Multi-step compliance (re-issued orders): ms_* result files."""
    nc = "rob_mpc_nc_bc"
    R = load("ms_main_k3") + load("ms_main_k2")
    if R:
        ms = ["ermpc_bc", "ermpc_ms_bc", "ermpc_msf_bc", "ermpc_ms2_bc", "ermpc_ms5_bc"]
        print("\nMULTI-STEP COMPLIANCE: method minus compliance-blind (mean wait, min); learned re-issue rate r")
        for k, f in sorted({(r["demand_k"], r["fleet_size"]) for r in R}, reverse=True):
            for dv in ("chase", "stay", "hotspot"):
                sel = dict(demand_k=k, fleet_size=f, deviation=dv)
                rr = mean(R, "ermpc_ms_bc", "reissue_rate", **sel)
                print(f"x{k:g} {f} {dv:7} nc={mean(R, nc, 'wait_all_min', **sel):.2f} r={rr:.2f}"
                      + "".join(f" | {m[:-3]} {fmt(*paired(R, m, nc, **sel))}" for m in ms)
                      + f" | ms-ermpc {fmt(*paired(R, 'ermpc_ms_bc', 'ermpc_bc', **sel))}")
    R = load("ms_level")
    if R:
        print("\nMULTI-STEP, COMPLIANCE LEVEL (x3, 160 vehicles, all human)")
        for dv in ("chase", "stay"):
            for am in sorted({r["alpha_mean"] for r in R}):
                sel = dict(deviation=dv, alpha_mean=am)
                print(f"{dv:5} a={am:+g} comply={mean(R, nc, 'compliance_rate', **sel):.2f} nc={mean(R, nc, 'wait_all_min', **sel):.2f}"
                      f" r={mean(R, 'ermpc_ms_bc', 'reissue_rate', **sel):.2f}"
                      + "".join(f" | {m[:-3]} {fmt(*paired(R, m, nc, **sel))}" for m in ("ermpc_bc", "ermpc_ms_bc", "ermpc_msf_bc")))
    R = load("ms_mix")
    if R:
        print("\nMULTI-STEP, SHARE OF STAY-TYPE DRIVERS (x3)")
        for f in (120, 160):
            for p in ("mix0.25", "mix0.5", "mix0.75"):
                sel = dict(fleet_size=f, deviation=p)
                print(f"{f} {p:8}" + "".join(f" | {m[:-3]} {fmt(*paired(R, m, nc, **sel))}"
                                            for m in ("ermpc_bc", "ermpc_ms_bc", "ermpc_msf_bc")))


def cf_tables():
    """Counterfactual diagnosis: cf_* result files (idle_humans = stay | self)."""
    R = load("cf_main_k3") + load("cf_main_k2")
    if not R:
        return
    nc = "rob_mpc_nc_bc"
    ms = ["ermpc_bc", "ermpc_nofb_bc", "ermpc_fbonly_bc", "ermpc_flat_bc"]
    print("\nCOUNTERFACTUAL: method minus compliance-blind (mean wait); issued human zone orders per run;"
          " share of orders equal to the fallback")
    for ih in ("stay", "self"):
        print(f"-- human drivers without an order: {ih}")
        for k, f in sorted({(r["demand_k"], r["fleet_size"]) for r in R}, reverse=True):
            for dv in ("chase", "stay", "hotspot"):
                sel = dict(demand_k=k, fleet_size=f, deviation=dv, idle_humans=ih)
                print(f"x{k:g} {f} {dv:7} self={mean(R, 'self', 'wait_all_min', **sel):.2f} nc={mean(R, nc, 'wait_all_min', **sel):.2f}"
                      f" nc-self {fmt(*paired(R, nc, 'self', **sel))}"
                      + "".join(f" | {m[:-3]} {fmt(*paired(R, m, nc, **sel))}" for m in ms)
                      + " | orders " + "/".join(f"{mean(R, m, 'diag_n_orders_h', **sel):.0f}" for m in [nc] + ms[:1])
                      + " same " + "/".join(f"{mean(R, m, 'diag_same_as_fallback', **sel):.2f}" for m in [nc] + ms[:1]))
    R = load("cf_mix")
    if R:
        print("\nCOUNTERFACTUAL, SHARE OF STAY-TYPE DRIVERS (idle_humans = self)")
        for f in (120, 160):
            for p in ("mix0.25", "mix0.5", "mix0.75"):
                sel = dict(fleet_size=f, deviation=p)
                print(f"{f} {p:8} ermpc-nc {fmt(*paired(R, 'ermpc_bc', nc, **sel))}")


def margin_table():
    """Demand margin and compliance with m = 0 (s_margin_* and s_final_main_* runs)."""
    R = load("margin_k3") + load("margin_k2") + load("final_main_k3") + load("final_main_k2")
    if not load("margin_k3"):
        return
    print("\nDEMAND MARGIN: mean wait; differences m=0 minus m=1 and compliance-aware minus blind at m=0")
    for k, f in sorted({(r["demand_k"], r["fleet_size"]) for r in R}, reverse=True):
        for dv in ("chase", "stay", "hotspot"):
            sel = dict(demand_k=k, fleet_size=f, deviation=dv)
            ms = ["self", "det_mpc_bc", "nc_m0_bc", "rob_mpc_nc_bc", "naive_plugin_bc", "ermpc_bc", "det_mpc_pi_bc"]
            print(f"x{k:g} {f} {dv:7} " + " ".join(f"{m}={mean(R, m, 'wait_all_min', **sel):.2f}" for m in ms)
                  + f" | m0-m1 {fmt(*paired(R, 'nc_m0_bc', 'rob_mpc_nc_bc', **sel))}"
                  + f" | aware-blind(m0) {fmt(*paired(R, 'naive_plugin_bc', 'nc_m0_bc', **sel))}"
                  + f" | det+pi-det {fmt(*paired(R, 'det_mpc_pi_bc', 'det_mpc_bc', **sel))}")


STATION = ["self", "rob_mpc_nc", "rob_mpc_nc_sm", "rob_mpc_nc_sh", "rob_mpc_nc_nr", "rob_mpc_nc_na",
           "rob_mpc_nc_sq", "rob_mpc_nc_bo", "rob_mpc_nc_bc", "rob_mpc_nc_bs"]
SHORT = lambda m: m.replace("rob_mpc_nc", "nc")


def to_charger(r):
    """Minutes per session from the station order to the start of charging (drive + queue)."""
    return r["veh_h_to_station"] * 60 / max(1, r["n_charges"]) + r["charge_wait_min"]


def _with_total(R):
    for r in R:
        r["to_charger_min"] = to_charger(r)
    return R


def _settings(R):
    for k in (3.0, 2.0):
        for pl in (4, 3, 2):
            for f in (160, 120):
                for dv in ("chase", "stay", "hotspot"):
                    sel = dict(demand_k=k, piles=pl, fleet_size=f, deviation=dv)
                    if any(all(r.get(a) == b for a, b in sel.items()) for r in R):
                        yield f"x{k:g} {pl}ch {f} {dv:7}", sel


def station_table():
    """Station-choice rules by chargers per station (st_* runs, compliance-blind MPC with m = 1)."""
    R = _with_total(load("st_k3") + load("st_k2") + load("st_bs_k3") + load("st_bs_k2"))
    if not R:
        return
    ref, others = "rob_mpc_nc_bs", ("rob_mpc_nc", "rob_mpc_nc_sm", "rob_mpc_nc_na", "rob_mpc_nc_bc")
    for key, title in (("charge_wait_min", "queue per session (min)"),
                       ("to_charger_min", "drive + queue per session (min)"),
                       ("wait_all_min", "mean passenger wait (min)")):
        print(f"\nSTATION CHOICE: {title}: " + " / ".join(SHORT(m) for m in STATION)
              + f"; then {SHORT(ref)} minus " + ", ".join(SHORT(b) for b in others))
        for lab, sel in _settings(R):
            print(lab + " " + "/".join(f"{mean(R, m, key, **sel):.2f}" for m in STATION) + " |"
                  + "".join(f" {fmt(*paired(R, ref, b, key, **sel))}" for b in others))
    print("\nSTATION CHOICE: fleet queue (veh-h) static / static+mc / backlog / sequential;"
          " deterministic MPC queue static -> backlog; backlog minus occupancy-only (queue); backlog minus nearest idle charger (queue); sequential minus capacity over one session (drive + queue)")
    for lab, sel in _settings(R):
        vh = [mean(R, m, "veh_h_queue", **sel) for m in ("rob_mpc_nc", "rob_mpc_nc_sm", "rob_mpc_nc_bc", ref)]
        print(lab + " veh-h " + "/".join(f"{x:.1f}" for x in vh)
              + f" | det {mean(R, 'det_mpc', 'charge_wait_min', **sel):.2f} -> "
                f"{mean(R, 'det_mpc_bc', 'charge_wait_min', **sel):.2f}"
              + f" | bc-bo {fmt(*paired(R, 'rob_mpc_nc_bc', 'rob_mpc_nc_bo', 'charge_wait_min', **sel))}"
              + f" | bc-na {fmt(*paired(R, 'rob_mpc_nc_bc', 'rob_mpc_nc_na', 'charge_wait_min', **sel))}"
              + f" | bs-sh (d+q) {fmt(*paired(R, ref, 'rob_mpc_nc_sh', 'to_charger_min', **sel))}")
    bs = [r for r in R if r["method"] == ref and r["n_charges"]]
    dur = [r["veh_h_charging"] * 60 / r["n_charges"] for r in bs]
    print(f"\nsequential backlog runs: session {np.mean(dur):.1f} min (range {min(dur):.1f}-{max(dur):.1f}), "
          f"sessions per shift {min(r['n_charges'] for r in bs)}-{max(r['n_charges'] for r in bs)}, "
          f"share of fleet time to station, queued or charging {min(r['share_charging_total'] for r in bs):.3f}-"
          f"{max(r['share_charging_total'] for r in bs):.3f}")

    A = _with_total(load("st_av"))
    if not A:
        return
    A += [r for r in R if r["av_rate"] == 0.5 and r["demand_k"] == 3.0 and r["deviation"] == "chase"
          and r["piles"] in (2, 4) and r["method"] in ("rob_mpc_nc_bc", "rob_mpc_nc_sm", "rob_mpc_nc_na", ref)]
    print("\nAV SHARE AND STATION CHOICE (x3, chase): queue / drive + queue per session for "
          "nc_bc / nc_bs / nc_sm / nc_na; nc_bs minus nc_sm (drive + queue)")
    for pl in (4, 2):
        for f in (160, 120):
            for a in (0.0, 0.5, 1.0):
                sel = dict(av_rate=a, piles=pl, fleet_size=f)
                row = f"{pl}ch {f} rho={a:g} "
                row += " ".join(f"{mean(A, m, 'charge_wait_min', **sel):.2f}/{mean(A, m, 'to_charger_min', **sel):.2f}"
                                for m in ("rob_mpc_nc_bc", ref, "rob_mpc_nc_sm", "rob_mpc_nc_na"))
                print(row + f" | {fmt(*paired(A, ref, 'rob_mpc_nc_sm', 'to_charger_min', **sel))}")


def fix_table():
    """Ablation of the sequential update: stations fixed before the solve by the same per-vehicle
    argmin, without adding each vehicle to the backlog (nc_bf), against nc_bc and nc_bs."""
    R = _with_total(load("st_bf_k3") + load("st_bf_k2"))
    if not R:
        return
    print("\nSEQUENTIAL UPDATE ABLATION: drive + queue per session nc_bc / nc_bf / nc_bs; nc_bf minus nc_bc; "
          "nc_bs minus nc_bf; steps per shift with >= 2 must-charge vehicles in the batch, and their share of "
          "the must-charge vehicles")
    for name, sel in _settings(R):
        bs = [r for r in R if r["method"] == "rob_mpc_nc_bs" and all(r[k] == v for k, v in sel.items())]
        share = sum(r["n_must_charge_multi"] for r in bs) / max(1, sum(r["n_must_charge"] for r in bs))
        print(f"{name} " + "/".join(f"{mean(R, m, 'to_charger_min', **sel):.2f}"
                                     for m in ("rob_mpc_nc_bc", "rob_mpc_nc_bf", "rob_mpc_nc_bs"))
              + f" | {fmt(*paired(R, 'rob_mpc_nc_bf', 'rob_mpc_nc_bc', 'to_charger_min', **sel))}"
              + f" {fmt(*paired(R, 'rob_mpc_nc_bs', 'rob_mpc_nc_bf', 'to_charger_min', **sel))}"
              + f" | {np.mean([r['n_steps_multi_charge'] for r in bs]):.1f} {share:.2f}")


def joint_table():
    """Exact joint assignment (nc_bj) and assignment order (nc_br random, nc_bl lowest SOC first)
    against the sequential rule, nearest first (nc_bs)."""
    R = _with_total(load("st_bj_k3") + load("st_bj_k2"))
    if not R:
        return
    print("\nJOINT ASSIGNMENT AND ORDER: queue / drive + queue per session nc_bs / nc_bj / nc_br / nc_bl; "
          "nc_bj, nc_br, nc_bl minus nc_bs (drive + queue); share of steps with >= 2 must-charge vehicles "
          "in which the joint assignment differed from the sequential one")
    for name, sel in _settings(R):
        bj = [r for r in R if r["method"] == "rob_mpc_nc_bj" and all(r[k] == v for k, v in sel.items())]
        diff = sum(r["joint_differs"] for r in bj) / max(1, sum(r["joint_steps"] for r in bj))
        print(f"{name} " + " ".join(f"{mean(R, m, 'charge_wait_min', **sel):.2f}/{mean(R, m, 'to_charger_min', **sel):.2f}"
                                     for m in ("rob_mpc_nc_bs", "rob_mpc_nc_bj", "rob_mpc_nc_br", "rob_mpc_nc_bl"))
              + " | " + " ".join(fmt(*paired(R, m, "rob_mpc_nc_bs", "to_charger_min", **sel))
                                 for m in ("rob_mpc_nc_bj", "rob_mpc_nc_br", "rob_mpc_nc_bl"))
              + f" | {diff:.2f}")


def aware_table():
    """Compliance-aware station choice (ermpc_bs: a human driver who will likely ignore the order joins
    the backlog of his own station) against ermpc_bc and the compliance-blind sequential rule."""
    if not load("st_ca"):
        return
    R = _with_total(load("st_ca") + load("st_diag"))
    print("\nCOMPLIANCE-AWARE STATION CHOICE (x3, 50% AV): drive + queue per session nc_bs / ca_bc / ca_bs; "
          "ca_bs minus ca_bc; ca_bs minus nc_bs (drive + queue, passenger wait)")
    for name, sel in _settings([r for r in R if r["method"] == "ermpc_bs"]):
        print(f"{name} " + "/".join(f"{mean(R, m, 'to_charger_min', **sel):.2f}"
                                     for m in ("rob_mpc_nc_bs", "ermpc_bc", "ermpc_bs"))
              + f" | {fmt(*paired(R, 'ermpc_bs', 'ermpc_bc', 'to_charger_min', **sel))}"
              + f" | {fmt(*paired(R, 'ermpc_bs', 'rob_mpc_nc_bs', 'to_charger_min', **sel))}"
              + f" {fmt(*paired(R, 'ermpc_bs', 'rob_mpc_nc_bs', 'wait_all_min', **sel))}")


def acceptance_table():
    """Station rules when human drivers accept orders more often: an order-acceptance constant kappa
    (alpha_mean) shifts the logit of every compliance probability (kappa = 0 is st_diag)."""
    A = load("st_acc")
    if not A:
        return
    R = _with_total(load("st_diag") + A)
    rules = ("rob_mpc_nc_bc", "rob_mpc_nc_bs", "rob_mpc_nc_sm", "rob_mpc_nc_na")
    src = ("av", "h_follow", "h_own")
    print("\nORDER ACCEPTANCE (x3, 50% AV, 2 or 4 chargers): per kappa and chargers, means over 6 settings of "
          "the share of human orders followed (nc_bs), the h_own share of human sessions and of queue minutes "
          "(nc_bs), queue / drive + queue for nc_bc / nc_bs / nc_sm / nc_na, and the settings in which nc_bs "
          "is better / worse than nc_sm and nc_na (drive + queue, paired 95%)")
    for kap in (0.0, 1.0, 2.0, 3.0):
        for pl in (4, 2):
            K = [r for r in R if r["alpha_mean"] == kap and r["piles"] == pl]
            bs = [r for r in K if r["method"] == "rob_mpc_nc_bs"]
            n = {s: sum(r["n_charges_" + s] for r in bs) for s in src}
            q = {s: sum(r["charge_wait_" + s] * r["n_charges_" + s] for r in bs if r["n_charges_" + s]) for s in src}
            row = (f"kappa={kap:g} {pl}ch followed {np.mean([r['compliance_rate'] for r in bs]):.2f} "
                   f"h_own {n['h_own'] / max(1, n['h_own'] + n['h_follow']):.2f}/{q['h_own'] / max(1e-9, sum(q.values())):.2f} | ")
            row += " ".join(f"{np.mean([r['charge_wait_min'] for r in K if r['method'] == m]):.2f}/"
                            f"{np.mean([r['to_charger_min'] for r in K if r['method'] == m]):.2f}" for m in rules)
            for other in ("rob_mpc_nc_sm", "rob_mpc_nc_na"):
                better = worse = 0
                for f in (160, 120):
                    for dv in ("chase", "stay", "hotspot"):
                        mu, h = paired(K, "rob_mpc_nc_bs", other, "to_charger_min", fleet_size=f, deviation=dv)
                        better += mu < -h
                        worse += mu > h
                row += f" | vs {other[-2:]} {better}/{worse}"
            print(row)


def source_table():
    """Queue at chargers by who chose the station: AVs, humans who followed the order, and the rest."""
    R = load("st_diag")
    if not R:
        return
    src = ("av", "h_follow", "h_own")
    print("\nQUEUE BY STATION SOURCE (x3, 50% AV, 12 settings with 2 or 4 chargers, pooled sessions)")
    R += load("st_ca")
    for m in ("rob_mpc_nc_bs", "rob_mpc_nc_sm", "rob_mpc_nc_bc", "rob_mpc_nc_na", "ermpc_bs", "ermpc_bc"):
        M = [r for r in R if r["method"] == m]
        n = {s: sum(r["n_charges_" + s] for r in M) for s in src}
        q = {s: sum(r["charge_wait_" + s] * r["n_charges_" + s] for r in M) for s in src}
        row = (f"{m:15s} h_own share of human sessions {n['h_own'] / (n['h_own'] + n['h_follow']):.3f}, "
               f"of queue minutes {q['h_own'] / sum(q.values()):.3f}; queue av+h_follow / h_own:")
        for pl in (4, 2):
            P = [r for r in M if r["piles"] == pl]
            per = [sum(r["charge_wait_" + s] * r["n_charges_" + s] for r in P for s in g)
                   / sum(r["n_charges_" + s] for r in P for s in g) for g in (("av", "h_follow"), ("h_own",))]
            row += f" {pl}ch {per[0]:.2f} / {per[1]:.2f}"
        print(row)


if __name__ == "__main__":
    main_tables()
    misspec_table()
    rho_table()
    mix_table()
    het_table()
    level_table()
    het_table("het_low", "alpha_i ~ N(-2, sd^2) (x3, 160 vehicles, all human)")
    ms_tables()
    cf_tables()
    margin_table()
    station_table()
    fix_table()
    joint_table()
    aware_table()
    acceptance_table()
    source_table()
