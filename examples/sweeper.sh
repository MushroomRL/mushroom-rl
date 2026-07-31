#!/usr/bin/env bash
# Sweep script: for each seed, train each env. One process per (seed, env).
# One wandb project per env, all seeds appear as separate runs inside that project.
#
# Usage:
#   ./run_sweep.sh                    # default: seeds 1..5, all four envs
#   SEEDS="1 2 3" ./run_sweep.sh      # custom seeds
#   ENVS="hopper walker_2d" ./run_sweep.sh   # subset of envs

set -euo pipefail

# Defaults; override via env vars
SEEDS="${SEEDS:-1 2 3 4 5}"
ENVS="${ENVS:-hopper walker_2d half_cheetah ant}"
NUM_ENVS="${NUM_ENVS:-4000}"
N_EPOCHS="${N_EPOCHS:-50}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-mushroom}"

# Log directory for stdout/stderr per run
LOG_DIR="./sweep_logs"
mkdir -p "$LOG_DIR"

echo "=== SWEEP CONFIG ==="
echo "  seeds:                $SEEDS"
echo "  envs:                 $ENVS"
echo "  num_envs:             $NUM_ENVS"
echo "  n_epochs:             $N_EPOCHS"
echo "  wandb project prefix: $WANDB_PROJECT_PREFIX"
echo "  log dir:              $LOG_DIR"
echo "====================="
echo

# Outer loop: seeds. Inner loop: envs.
# This matches your intended workflow: seed 1: hopper cheetah walker; seed 2: ...
for seed in $SEEDS; do
    for env in $ENVS; do
        run_id="seed_${seed}_${env}"
        log_file="${LOG_DIR}/${run_id}.log"

        echo "[$(date +%H:%M:%S)] START seed=$seed env=$env"
        echo "  logging to: $log_file"

        # 2>&1 | tee streams to both file and terminal.
        # If you want silent operation, replace with `> "$log_file" 2>&1`.
        python normal_ppo.py \
            --env "$env" \
            --seed "$seed" \
            --num_envs "$NUM_ENVS" \
            --n_epochs "$N_EPOCHS" \
            --wandb_project_prefix "$WANDB_PROJECT_PREFIX" \
            2>&1 | tee "$log_file"

        # Check exit status of python, not tee (which almost always succeeds).
        # PIPESTATUS is bash-only; if using sh switch to a wrapper approach.
        py_status=${PIPESTATUS[0]}
        if [ "$py_status" -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] FAILED seed=$seed env=$env (exit $py_status)"
            echo "  continuing with next run"
        else
            echo "[$(date +%H:%M:%S)] DONE seed=$seed env=$env"
        fi
        echo
    done
done

echo "=== SWEEP COMPLETE ==="
