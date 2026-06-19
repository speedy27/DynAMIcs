#!/usr/bin/env bash
# BIG BET — close the planning loop. The oracle K-sweep (CPU, oracle_K_sweep.py) MEASURED that the gLV
# task is CONTROLLABLE only with a large action panel: a PERFECT planner reaches all targets at K=24
# (success 1.00, final 0.78) but 0% at K<=18 (final 4.09 -> 2.38), a clean monotone controllability
# curve at a FIXED action budget. So the M3 negative was actuation/controllability, not the model.
#
# This job retrains the world model at K=24 (all species dose-able) with the NON-COLLAPSE (default)
# regularizer (std=1, cov=25, sim_t=1 — good latent geometry, vs the collapse regime used for the IDM
# headline), then re-runs the LEARNED latent-MPPI with the SAME planning protocol as the committed K=6
# negative (seeds 0,1,2; 12 episodes; mpc_steps 20; horizon 6). Two intended changes vs the committed
# run: K 6->24 and regime collapse->default; both are reported explicitly.
#   * If learned-MPPI now reaches the target -> M3 flips to diagnosed-AND-FIXED (a world model you can
#     plan with).
#   * If the oracle/greedy reach it but learned-MPPI still fails -> isolates the bottleneck to the
#     latent cost (diagnosis corr -0.19); a stronger "controllable in principle" result.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_plan_k24.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=glv_plan_k24
#SBATCH --output=glv_plan_k24_%j.out
#SBATCH --error=glv_plan_k24_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
MD=$WORK/checkpoints/microbiome_jepa/plan_model_k24
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ train K=24 world model (DEFAULT reg, idm on, d_model=128, 80ep) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 128 --data.n_candidate 24 \
  --logging.tqdm_silent True

echo "############ plan interventions at K=24 (random / greedy / final_only / mppi) ############"
$PY -m examples.microbiome_jepa.plan_glv --fname $CFG --checkpoint $MD/latest.pth.tar \
  --device cuda --seeds 0,1,2 --n_episodes 12 --mpc_steps 20 --horizon 6 --n_samples 128 --n_iters 3 \
  --out $WORK/checkpoints/microbiome_jepa/planning_k24 \
  --overrides '{"data.n_candidate": 24, "model.d_model": 128}'
echo "PLAN_K24_DONE"
