#!/bin/bash -l
#SBATCH -J prop7_koop_l40s
#SBATCH -p s.geany.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --array=0-79%16

#SBATCH -D /u/alfo/cross_hybrid_deepkoopformer/source
#SBATCH -o /u/alfo/cross_hybrid_deepkoopformer/source/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/cross_hybrid_deepkoopformer/source/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load cuda/12.2 || module load cuda || true
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

mkdir -p logs
mkdir -p results/prop7_ablation

PATCHES=(16 24 32 40)
HORIZONS=(4 8 12 16)
SEEDS=(7 42 123 2025 2026)

IDX=${SLURM_ARRAY_TASK_ID}

SEED_IDX=$(( IDX / 16 ))
REM=$(( IDX % 16 ))
PATCH_IDX=$(( REM / 4 ))
HORIZON_IDX=$(( REM % 4 ))

SEED=${SEEDS[$SEED_IDX]}
PATCH=${PATCHES[$PATCH_IDX]}
HORIZON=${HORIZONS[$HORIZON_IDX]}

echo "Prop7 task ${IDX}: seed=${SEED}, patch=${PATCH}, horizon=${HORIZON}"
echo "Node=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun python koopformer_prop7_ablation.py \
  --file ./df_cleaned_numeric_2.npy \
  --save_dir /u/alfo/cross_hybrid_deepkoopformer/source/results/prop7_ablation/seed${SEED}_patch${PATCH}_h${HORIZON} \
  --seeds "${SEED}" \
  --backbones dlinear,ssm,gatedssm,patchtst,autoformer,informer,itransformer,timesnet \
  --patch_lens "${PATCH}" \
  --horizons "${HORIZON}" \
  --indices 0,1,2,3,4,5 \
  --epochs 4000 \
  --lr 0.0003 \
  --train_frac 0.8 \
  --max_rows 2500 \
  --d_model 96 \
  --num_heads 4 \
  --num_layers 3 \
  --dim_ff 96 \
  --rho_max 0.99 \
  --lyap_weight 0.1
