# Queue-aware charging-station choice for mixed AV/human electric ride-hailing fleets

Code, data, and results for the paper *Queue-Aware Charging-Station Choice in the Dispatch of Mixed Autonomous and Human-Driven Electric Ride-Hailing Fleets* (Siyuan Meng and Chengbo Ai, University of Massachusetts Amherst; manuscript in preparation).

A rolling-horizon dispatcher repositions vacant vehicles and chooses charging stations for an electric ride-hailing fleet of autonomous vehicles and human drivers in the Chicago Loop. Station choice uses a waiting time predicted from the vehicles in service, queued, and en route at each station; the demand coverage can optionally use the expected supply of human drivers under a Logit compliance model. The repository contains the dispatch model, the full-shift simulator, the network and demand data, and the result files behind every table and figure.

## Contents

| Path | Content |
|---|---|
| `chicago_loop/models/optimizer.py` | Dispatch MILP (demand coverage with margin, predicted station wait or static capacity, expected supply, Γ_π protection) |
| `chicago_loop/models/compliance.py` | Logit compliance model and expected-supply coefficients |
| `chicago_loop/env/shift_sim.py` | Event-timed 12-h shift simulator (requests, abandonment, driver compliance, charging queues) |
| `chicago_loop/env/fleet.py` | Vehicle and station states, including the predicted wait (`StationState.predicted_wait`) |
| `chicago_loop/env/demand_model.py` | Historical zonal demand and the dispatcher's forecaster |
| `chicago_loop/experiments/run_eval.py` | Experiment driver: methods, settings, paired seeds, parallel runs |
| `chicago_loop/experiments/summarize_final.py`, `make_figures_final.py` | Summary tables (paired 95% t-intervals) and figures |
| `chicago_loop/data/network/` | SUMO network of the Loop, zone and station locations, network distances, hourly taxi speeds, and their build scripts |
| `chicago_loop/data/demand/` | Zonal demand baselines and the taxi/ride-hailing pickup comparison, with their query scripts |
| `chicago_loop/scripts/prepare_demand.py` | Builds the demand baselines from the City of Chicago Taxi Trips data |
| `chicago_loop/results/` | Result files (one record per 12-h shift) and `final_summary.txt` |

## Setup

1. Python 3.11 and `pip install -r requirements.txt`.
2. Gurobi: the pip package includes a size-limited license (2,000 variables and constraints), which is enough because each program dispatches at most 40 vehicles.

All commands are run from the repository root.

## Reproducing the results

Summary tables and figures from the included result files:

```
python -m chicago_loop.experiments.summarize_final      # tables (also in chicago_loop/results/final_summary.txt)
python -m chicago_loop.experiments.make_figures_final   # writes figures/fig_*.pdf
```

`bash run_all.sh` re-runs every experiment (set `WORKERS` for the number of processes). A single setting can be run with, for example,

```
python -m chicago_loop.experiments.run_eval --methods rob_mpc_nc rob_mpc_nc_bc --demand_k 3 --fleet 160 \
    --av_rates 0.5 --deviation chase --idle_humans self --piles 2 --seeds 20 --tag test
```

Every method sees the same demand realization and the same compliance draws for a given seed. MILP solutions can vary slightly with the number of solver threads; the paper used one thread per run.

## Method and field names

| Name in code | Method in the paper |
|---|---|
| `self` | Self-dispatch |
| `det_mpc` | Deterministic MPC ($m = 0$, no look-ahead) |
| `nc_m0`, `rob_mpc_nc` | Compliance-blind MPC with $m = 0$ and $m = 1$ |
| `naive_plugin`, `ermpc` | Compliance-aware MPC with $m = 0$ and $m = 1$ |
| `det_mpc_pi` | Deterministic MPC with compliance probabilities |
| `ermpc_rpi`, `ermpc_rpi4` | Robust compliance with $\Gamma_\pi = 2$ and 4 |
| `ermpc_pool`, `ermpc_ln`, `ermpc_or` | Compliance intercept learned for all drivers, learned per driver, and known (oracle) |
| `ermpc_ms[K]`, `ermpc_msf[K]` | Multi-step compliance with learned re-issue rate and with $r = 1$ ($K = 3$ by default) |
| `ermpc_nofb`, `ermpc_fbonly`, `ermpc_flat` | Ablations: no fallback credit, fallback credit only, zone-independent probability |
| suffix `_bc` / `_bo` / none | Station choice by backlog wait / occupancy-only wait / static capacity constraint |

In the result files, `deviation` is the fallback type of drivers who ignore an order (`chase`, `stay`, `hotspot` for the favourite zone, or `mix<p>` with a share `p` of stay-type drivers), `idle_humans` is what a human driver without an order does (`self`: takes the fallback action, the main assumption; `stay`: waits in place), `piles` is the number of chargers per station, and `alpha_mean` and `alpha_sd` set the persistent driver compliance propensity. Waits are in minutes; `wait_all_min` counts abandoned requests as 10 min, and `charge_wait_min` is the queue at chargers per session.

## Data

The road network was built from OpenStreetMap (© OpenStreetMap contributors, ODbL) with SUMO `netconvert`; `build_od.py` computes the network distances. Demand baselines, hourly speeds, and the taxi/ride-hailing pickup comparison were derived from the City of Chicago Data Portal (Taxi Trips, dataset `wrvz-psew`; Transportation Network Providers Trips, dataset `n26f-ihde`) by the scripts in `chicago_loop/data/` and `chicago_loop/scripts/`.

## License

Code: MIT (see `LICENSE`). The network data derive from OpenStreetMap (ODbL); the trip data are published by the City of Chicago under its data portal terms of use.
