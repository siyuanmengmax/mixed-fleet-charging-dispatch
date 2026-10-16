"""
Figures of the paper from the result files in chicago_loop/results.

  python -m chicago_loop.experiments.make_figures_final   # writes figures/fig_*.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chicago_loop.experiments.summarize_final import load, mean, paired

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = Path("figures")
C = {"chase": "#56B4E9", "hotspot": "#E69F00", "stay": "#D55E00", 120: "#009E73", 160: "#0072B2"}
SETTINGS = [(3, 160), (3, 120), (2, 120)]
LABEL = {"chase": "Chase current requests", "hotspot": "Return to favourite zone", "stay": "Stay in place"}


def fig_fallback(R):
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    w = 0.25
    for k, dv in enumerate(("chase", "hotspot", "stay")):
        mu, h = zip(*[paired(R, "ermpc_bc", "rob_mpc_nc_bc", demand_k=d, fleet_size=f, deviation=dv)
                      for d, f in SETTINGS])
        ax.bar(np.arange(3) + (k - 1) * w, mu, w, yerr=h, color=C[dv], label=LABEL[dv],
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(3), [f"{d}x demand\n{f} vehicles" for d, f in SETTINGS])
    ax.set_ylabel("Change in mean wait (min)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "fig_fallback.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_mix(R):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ps = [0, 0.25, 0.5, 0.75, 1]
    for f in (120, 160):
        mu, h = zip(*[paired(R, "ermpc_bc", "rob_mpc_nc_bc", fleet_size=f, deviation=f"mix{p:g}") for p in ps])
        ax.errorbar(np.array(ps) + (0.01 if f == 160 else -0.01), mu, yerr=h, marker="o", ms=3, lw=1,
                    capsize=2, color=C[f], label=f"{f} vehicles")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Share of drivers who stay in place when not complying")
    ax.set_ylabel("Change in mean wait (min)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mix.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rho(R):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    rhos = sorted({r["av_rate"] for r in R})
    for dv in ("chase", "stay"):
        mu, h = zip(*[paired(R, "ermpc_bc", "rob_mpc_nc_bc", av_rate=a, deviation=dv) for a in rhos])
        ax.errorbar(rhos, mu, yerr=h, marker="o", ms=3, lw=1, capsize=2, color=C[dv], label=LABEL[dv])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Share of autonomous vehicles")
    ax.set_ylabel("Change in mean wait (min)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rho.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_mechanism(R):
    """Counterfactual ablations of the expected supply (cf_* runs), 3x demand, 160 vehicles."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    var = [("ermpc_bc", "Full compliance model", "#D55E00"),
           ("ermpc_fbonly_bc", "Fallback position only", "#E69F00"),
           ("ermpc_nofb_bc", "Compliance discount only", "#56B4E9"),
           ("ermpc_flat_bc", "Zone-independent probability", "#999999")]
    w = 0.2
    for ax, ih, title in zip(axes, ("self", "stay"), ("(a) Drivers without an order act on their own",
                                                      "(b) Drivers without an order stay in place")):
        for k, (m, lab, c) in enumerate(var):
            mu, h = zip(*[paired(R, m, "rob_mpc_nc_bc", demand_k=3.0, fleet_size=160, deviation=dv, idle_humans=ih)
                          for dv in ("chase", "hotspot", "stay")])
            ax.bar(np.arange(3) + (k - 1.5) * w, mu, w, yerr=h, color=c, label=lab, error_kw=dict(lw=0.6, capsize=1.5))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(3), ["Chase current\nrequests", "Return to\nfavourite zone", "Stay in\nplace"])
        ax.set_title(title)
    axes[0].set_ylabel("Change in mean wait (min)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_level(R):
    """Mean wait against the share of orders followed (all-human fleet, s_het_level runs)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=False)
    ams = sorted({r["alpha_mean"] for r in R})
    for ax, dv, title in zip(axes, ("chase", "stay"), ("(a) Chase current requests", "(b) Stay in place")):
        x = [mean(R, "rob_mpc_nc_bc", "compliance_rate", deviation=dv, alpha_mean=a) for a in ams]
        for m, lab, c in (("self", "Self-dispatch", "#999999"), ("rob_mpc_nc_bc", "Compliance-blind", "#0072B2"),
                          ("ermpc_bc", "Compliance-aware", "#D55E00")):
            ax.plot(x, [mean(R, m, "wait_all_min", deviation=dv, alpha_mean=a) for a in ams], marker="o", ms=3,
                    lw=1, color=c, label=lab)
        ax.set_xlabel("Share of orders followed")
        ax.set_title(title)
    axes[0].set_ylabel("Mean wait (min)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_level.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_scarcity(R):
    """Queue at chargers by chargers per station (s_charge runs, 3x demand, chase drivers)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    cols = [("self", "Self-dispatch (nearest station)", "#BBBBBB"),
            ("rob_mpc_nc", "MPC, static capacity", "#999999"),
            ("rob_mpc_nc_bo", "MPC, occupancy wait", "#56B4E9"),
            ("rob_mpc_nc_bc", "MPC, backlog wait", "#D55E00")]
    w = 0.2
    for ax, f in zip(axes, (160, 120)):
        for k, (m, lab, c) in enumerate(cols):
            ax.bar(np.arange(3) + (k - 1.5) * w,
                   [mean(R, m, "charge_wait_min", piles=p, fleet_size=f, deviation="chase") for p in (2, 3, 4)],
                   w, color=c, label=lab)
        ax.set_xticks(range(3), ["2", "3", "4"])
        ax.set_xlabel("Chargers per station")
        ax.set_title(f"({'a' if f == 160 else 'b'}) {f} vehicles")
    axes[0].set_ylabel("Queue at charger (min per session)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig_scarcity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_charging(R):
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    w = 0.2
    cols = [("rob_mpc_nc", "Compliance-blind, static capacity", "#BBBBBB"),
            ("rob_mpc_nc_bc", "Compliance-blind, backlog wait", "#0072B2"),
            ("ermpc", "Compliance-aware, static capacity", "#999999"),
            ("ermpc_bc", "Compliance-aware, backlog wait", "#D55E00")]
    for k, (m, lab, c) in enumerate(cols):
        vals = [mean(R, m, "charge_wait_min", demand_k=d, fleet_size=f, deviation="chase") for d, f in SETTINGS]
        ax.bar(np.arange(3) + (k - 1.5) * w, vals, w, color=c, label=lab)
    ax.set_xticks(range(3), [f"{d}x demand\n{f} vehicles" for d, f in SETTINGS])
    ax.set_ylabel("Queue at charger (min per session)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "fig_charging.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main = load("final_main_k3") + load("final_main_k2")
    fig_fallback(main)
    fig_charging(main)
    fig_mix(load("final_mix"))
    fig_rho(load("final_rho"))
    fig_mechanism(load("cf_main_k3"))
    fig_level(load("het_level"))
    fig_scarcity(load("charge"))
    print("figures written to", OUT)
