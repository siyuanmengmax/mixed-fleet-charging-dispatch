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
    files = sorted(glob.glob(f"{RES}/{PREFIX if not tag.startswith('cf_') else ''}{tag}_2*.json"))
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
    methods = ["self", "det_mpc", "rob_mpc_nc", "rob_mpc_nc_bc", "ermpc", "ermpc_bc", "ermpc_rpi_bc"]
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
    pairs = [("rob_mpc_nc", "self"), ("ermpc_bc", "rob_mpc_nc_bc"), ("ermpc_rpi_bc", "ermpc_bc"),
             ("rob_mpc_nc_bc", "rob_mpc_nc"), ("ermpc_bc", "ermpc")]
    for k, f in settings:
        for dv in ("chase", "stay", "hotspot"):
            sel = dict(demand_k=k, fleet_size=f, deviation=dv)
            print(f"x{k:g} {f} {dv:7}" + "".join(f" | {a}-{b} {fmt(*paired(R, a, b, **sel))}" for a, b in pairs))
    print("\nCHARGING: queue per session (min), fleet queue (veh-h), passenger wait; _bc minus static")
    for k, f in settings:
        for dv in ("chase", "stay"):
            sel = dict(demand_k=k, fleet_size=f, deviation=dv)
            row = f"x{k:g} {f} {dv:5}"
            for a, b in [("rob_mpc_nc_bc", "rob_mpc_nc"), ("ermpc_bc", "ermpc")]:
                row += (f" | {a}: q {fmt(*paired(R, a, b, 'charge_wait_min', **sel))}"
                        f" veh-h {fmt(*paired(R, a, b, 'veh_h_queue', **sel))}"
                        f" wait {fmt(*paired(R, a, b, **sel))}")
            print(row)


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


def charge_table():
    """Station choice by chargers per station (s_charge runs)."""
    R = load("charge")
    if not R:
        return
    print("\nSTATION CHOICE: queue per session (min) self / static / occupancy / backlog; backlog minus static")
    for pl in (4, 3, 2):
        for f in (160, 120):
            for dv in ("chase", "stay", "hotspot"):
                sel = dict(piles=pl, fleet_size=f, deviation=dv)
                q = [mean(R, m, "charge_wait_min", **sel) for m in ("self", "rob_mpc_nc", "rob_mpc_nc_bo", "rob_mpc_nc_bc")]
                print(f"{pl} chargers {f} {dv:7} " + "/".join(f"{x:.2f}" for x in q)
                      + f" | queue {fmt(*paired(R, 'rob_mpc_nc_bc', 'rob_mpc_nc', 'charge_wait_min', **sel))}"
                      + f" | backlog-occupancy {fmt(*paired(R, 'rob_mpc_nc_bc', 'rob_mpc_nc_bo', 'charge_wait_min', **sel))}"
                      + f" | passenger wait {fmt(*paired(R, 'rob_mpc_nc_bc', 'rob_mpc_nc', **sel))}")


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
    charge_table()
