#!/usr/bin/env bash
# Re-runs every experiment of the paper (about 3 h with 30 worker processes on a 32-thread
# desktop CPU). Each command writes chicago_loop/results/<tag>_<timestamp>.json.
set -e
R="python -m chicago_loop.experiments.run_eval --seeds 20 --workers ${WORKERS:-30} --idle_humans self"
M7="self det_mpc rob_mpc_nc rob_mpc_nc_bc ermpc ermpc_bc ermpc_rpi_bc"
H5="rob_mpc_nc_bc ermpc_bc ermpc_pool_bc ermpc_ln_bc ermpc_or_bc"
MS="rob_mpc_nc_bc ermpc_bc ermpc_ms_bc ermpc_msf_bc ermpc_ms2_bc ermpc_ms5_bc"
MG="det_mpc_bc det_mpc_pi_bc nc_m0_bc naive_plugin_bc"
CF="self rob_mpc_nc_bc ermpc_bc ermpc_nofb_bc ermpc_fbonly_bc ermpc_flat_bc"

# Main experiment and demand margin
$R --methods $M7 --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay hotspot --tag s_final_main_k3
$R --methods $M7 --demand_k 2 --fleet 120 --av_rates 0.5 --deviation chase stay hotspot --tag s_final_main_k2
$R --methods $MG --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay hotspot --tag s_margin_k3
$R --methods $MG --demand_k 2 --fleet 120 --av_rates 0.5 --deviation chase stay hotspot --tag s_margin_k2

# Station choice: all rules, AV share, and the queue by station source
ST="self det_mpc det_mpc_bc rob_mpc_nc rob_mpc_nc_bc rob_mpc_nc_bo rob_mpc_nc_nr rob_mpc_nc_na rob_mpc_nc_sq rob_mpc_nc_sm rob_mpc_nc_sh"
S4="rob_mpc_nc_bc rob_mpc_nc_bs rob_mpc_nc_sm rob_mpc_nc_na"
$R --methods $ST --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 3 4 --tag st_k3
$R --methods $ST --demand_k 2 --fleet 120 --deviation chase stay hotspot --piles 2 3 4 --tag st_k2
$R --methods rob_mpc_nc_bs --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 3 4 --tag st_bs_k3
$R --methods rob_mpc_nc_bs --demand_k 2 --fleet 120 --deviation chase stay hotspot --piles 2 3 4 --tag st_bs_k2
$R --methods $S4 --demand_k 3 --fleet 120 160 --av_rates 0 1 --deviation chase --piles 2 4 --tag st_av
$R --methods $S4 --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 4 --tag st_diag
# Ablation of the sequential update, compliance-aware station choice, and higher order acceptance
BF="rob_mpc_nc_bc rob_mpc_nc_bf rob_mpc_nc_bs"
$R --methods $BF --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 3 4 --tag st_bf_k3
$R --methods $BF --demand_k 2 --fleet 120 --deviation chase stay hotspot --piles 2 3 4 --tag st_bf_k2
$R --methods ermpc_bc ermpc_bs --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 4 --tag st_ca
$R --methods rob_mpc_nc_bc rob_mpc_nc_bs rob_mpc_nc_sm rob_mpc_nc_na --demand_k 3 --fleet 120 160 --deviation chase stay hotspot \
   --piles 2 4 --alpha_mean 1 2 3 --tag st_acc
BJ="rob_mpc_nc_bs rob_mpc_nc_bj rob_mpc_nc_br rob_mpc_nc_bl"
$R --methods $BJ --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 3 4 --tag st_bj_k3
$R --methods $BJ --demand_k 2 --fleet 120 --deviation chase stay hotspot --piles 2 3 4 --tag st_bj_k2
# Joint rule in the AV-share, diagnostic, compliance-aware and order-acceptance runs (same seeds, merged in the summary)
$R --methods rob_mpc_nc_bj --demand_k 3 --fleet 120 160 --av_rates 0 1 --deviation chase --piles 2 4 --tag st_av_bj
$R --methods rob_mpc_nc_bj --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 4 --tag st_diag_bj
$R --methods ermpc_bj --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 4 --tag st_ca_bj
$R --methods rob_mpc_nc_bj --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 4 --alpha_mean 1 2 3 --tag st_acc_bj
# Joint rule again with the largest must-charge batch per step logged (same results as st_bj)
$R --methods rob_mpc_nc_bj --demand_k 3 --fleet 120 160 --deviation chase stay hotspot --piles 2 3 4 --tag st_nmax_k3
$R --methods rob_mpc_nc_bj --demand_k 2 --fleet 120 --deviation chase stay hotspot --piles 2 3 4 --tag st_nmax_k2
# Distant stations: an access leg of 1.5 or 3 km added to every trip to and from a station
$R --methods rob_mpc_nc_na rob_mpc_nc_sm rob_mpc_nc_bo rob_mpc_nc_bc rob_mpc_nc_bs rob_mpc_nc_bj --demand_k 3 --fleet 120 160 \
   --deviation chase stay hotspot --piles 2 4 --station_km 1.5 3 --tag st_far

# Compliance: mixtures, AV share, compliance level, driver heterogeneity, multi-step, misspecification
$R --methods rob_mpc_nc_bc ermpc_bc --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation mix0 mix0.25 mix0.5 mix0.75 mix1 --tag s_final_mix
$R --methods self rob_mpc_nc_bc ermpc_bc --demand_k 3 --fleet 160 --av_rates 0 0.3 0.5 0.7 0.9 --deviation chase stay --tag s_final_rho
$R --methods self $H5 --demand_k 3 --fleet 160 --av_rates 0 --deviation chase stay --alpha_mean -4 -3 -2 -1 0 --tag s_het_level
$R --methods $H5 --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay --alpha_sd 0 1 2 3 --tag s_het_main
$R --methods $H5 --demand_k 3 --fleet 160 --av_rates 0 --deviation chase stay --alpha_mean -2 --alpha_sd 0 1 2 3 --tag s_het_low
$R --methods $MS --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay hotspot --tag s_ms_main_k3
$R --methods $MS --demand_k 2 --fleet 120 --av_rates 0.5 --deviation chase stay hotspot --tag s_ms_main_k2
for t in 0.5 3.0; do
  $R --methods rob_mpc_nc_bc ermpc_bc ermpc_rpi_bc ermpc_rpi4_bc --demand_k 3 --fleet 120 160 --av_rates 0.5 \
     --deviation chase stay --theta_true $t --tag s_final_misspec_t$t
done

# Ablation of the expected supply, under both assumptions about drivers without an order
C="python -m chicago_loop.experiments.run_eval --seeds 20 --workers ${WORKERS:-30}"
$C --methods $CF --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay hotspot --idle_humans stay self --tag cf_main_k3
$C --methods $CF --demand_k 2 --fleet 120 --av_rates 0.5 --deviation chase stay hotspot --idle_humans stay self --tag cf_main_k2
$C --methods rob_mpc_nc_bc ermpc_bc --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation mix0.25 mix0.5 mix0.75 --idle_humans self --tag cf_mix

python -m chicago_loop.experiments.summarize_final > chicago_loop/results/final_summary.txt
python -m chicago_loop.experiments.make_figures_final
python -m chicago_loop.experiments.time_joint    # timing of the joint assignment
