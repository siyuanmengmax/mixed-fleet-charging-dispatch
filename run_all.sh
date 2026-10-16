#!/usr/bin/env bash
# Re-runs every experiment of the paper (about 1 h with 30 worker processes on a 32-thread
# desktop CPU). Each command writes chicago_loop/results/<tag>_<timestamp>.json.
set -e
R="python -m chicago_loop.experiments.run_eval --seeds 20 --workers ${WORKERS:-30} --idle_humans self"
M7="self det_mpc rob_mpc_nc rob_mpc_nc_bc ermpc ermpc_bc ermpc_rpi_bc"
H5="rob_mpc_nc_bc ermpc_bc ermpc_pool_bc ermpc_ln_bc ermpc_or_bc"
MS="rob_mpc_nc_bc ermpc_bc ermpc_ms_bc ermpc_msf_bc ermpc_ms2_bc ermpc_ms5_bc"
MG="det_mpc_bc det_mpc_pi_bc nc_m0_bc naive_plugin_bc"
CF="self rob_mpc_nc_bc ermpc_bc ermpc_nofb_bc ermpc_fbonly_bc ermpc_flat_bc"

# Main experiment, demand margin, and station choice
$R --methods $M7 --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay hotspot --tag s_final_main_k3
$R --methods $M7 --demand_k 2 --fleet 120 --av_rates 0.5 --deviation chase stay hotspot --tag s_final_main_k2
$R --methods $MG --demand_k 3 --fleet 120 160 --av_rates 0.5 --deviation chase stay hotspot --tag s_margin_k3
$R --methods $MG --demand_k 2 --fleet 120 --av_rates 0.5 --deviation chase stay hotspot --tag s_margin_k2
$R --methods self rob_mpc_nc rob_mpc_nc_bo rob_mpc_nc_bc det_mpc det_mpc_bc --demand_k 3 --fleet 120 160 --av_rates 0.5 \
   --deviation chase stay hotspot --piles 2 3 4 --tag s_charge

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
