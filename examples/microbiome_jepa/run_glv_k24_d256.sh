#!/usr/bin/env bash
# d256 weak-reg K=24 world model (80ep, 256 traj) — MATCHED to the d128 weak-reg planning model except
# d_model, to (a) give a clean, un-confounded 3rd readout-fidelity point for the decoded-MPPI trend and
# (b) provide a higher-capacity substrate to test whether more capacity raises the encoder's
# state-retention wall (~0.89 at d128) — and a 2nd substrate for the learned-cost M3 experiment.
# Runs on the 2nd GPU concurrently with the M2-baseline job. Submit: sbatch run_glv_k24_d256.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=glv_k24_d256
#SBATCH --output=glv_k24_d256_%j.out
#SBATCH --error=glv_k24_d256_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
MD=$WORK/checkpoints/microbiome_jepa/plan_model_k24_d256
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
echo "############ train d256 weak-reg K=24 world model (80ep, 256 traj) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 256 --data.n_candidate 24 --data.n_traj 256 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --logging.tqdm_silent True
echo "K24_D256_DONE"
