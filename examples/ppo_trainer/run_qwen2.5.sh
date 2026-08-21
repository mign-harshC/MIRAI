#!/usr/bin/env bash
# Modified for the MIRAI project, 2026.

set -euo pipefail

: "${MIRAI_TRAIN_DATA:?Set MIRAI_TRAIN_DATA to a training parquet file}"
: "${MIRAI_VAL_DATA:?Set MIRAI_VAL_DATA to a validation parquet file}"

MIRAI_MODEL_PATH=${MIRAI_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
MIRAI_CRITIC_PATH=${MIRAI_CRITIC_PATH:-$MIRAI_MODEL_PATH}
MIRAI_LOGGER=${MIRAI_LOGGER:-console}
MIRAI_N_GPUS=${MIRAI_N_GPUS:-8}
MIRAI_TENSOR_PARALLEL=${MIRAI_TENSOR_PARALLEL:-4}
MIRAI_PYTHON_BIN=${MIRAI_PYTHON_BIN:-python3}

if [[ "$MIRAI_LOGGER" == "wandb" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "MIRAI_LOGGER=wandb requires WANDB_API_KEY in the environment." >&2
  exit 2
fi

export VLLM_USE_V1=${VLLM_USE_V1:-1}

exec "$MIRAI_PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files="['${MIRAI_TRAIN_DATA}']" \
  data.val_files="['${MIRAI_VAL_DATA}']" \
  data.train_batch_size=128 \
  data.val_batch_size=512 \
  data.max_prompt_length=3072 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MIRAI_MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$MIRAI_TENSOR_PARALLEL" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.mode=async \
  critic.optim.lr=1e-5 \
  critic.model.path="$MIRAI_CRITIC_PATH" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="[\"${MIRAI_LOGGER}\"]" \
  trainer.project_name=MIRAI \
  trainer.experiment_name=joint_dynamic_optimization \
  trainer.n_gpus_per_node="$MIRAI_N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=78 \
  trainer.test_freq=78 \
  trainer.total_epochs=3 \
  trainer.val_before_train=True \
  data.return_raw_chat=True \
  "$@"
