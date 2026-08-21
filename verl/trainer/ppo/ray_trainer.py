# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for the MIRAI project, 2026.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

import copy
import time
import logging
# logging.disable(logging.CRITICAL)  # 关闭所有日志输出

from qa_manager.compute_metrics import compute_scores
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import random

import asyncio
from tqdm.asyncio import tqdm as async_tqdm  # 异步版 tqdm


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def test_validate(self):
        import asyncio
        from qa_manager.qa import MIRAIManager
        import numpy as np

        print('Testing Begin.')
        start_time = time.time()     # 记录开始时间

        # 唤醒一次模型
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.async_rollout_manager.wake_up()

        # init qa manager
        qa_manager = MIRAIManager(self.tokenizer, self.config)

        all_test_metrics = []

        async def run_batch(test_batch_dict):
            return await self.async_test(
                test_batch_dict,
                qa_manager,
                self.config.data.val_batch_size
            )

        # count = 0
        # for test_batch_dict in self.val_dataloader:
        for test_batch_dict in tqdm(self.val_dataloader, desc="Validating"):
            # count+=1
            try:
                # 如果有运行事件循环（Ray 环境一般是这样）
                loop = asyncio.get_running_loop()
                _, test_reward_metrics = loop.run_until_complete(run_batch(test_batch_dict))
            except RuntimeError:
                # 没有运行事件循环
                _, test_reward_metrics = asyncio.run(run_batch(test_batch_dict))

            all_test_metrics.append(test_reward_metrics)
            
            # if count >= 3:
            #     break

        # 收尾
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.async_rollout_manager.sleep()
        
        end_time = time.time()       # 记录结束时间
        elapsed_time = end_time - start_time  # 秒为单位的时间差
        print(f"Testing耗时: {elapsed_time:.2f} 秒")

        # # 计算平均值
        # final_metrics = {}
        # for key in all_test_metrics[-1].keys():
        #     final_metrics[key] = np.mean([dic[key] for dic in all_test_metrics])

        # 计算平均值
        ## 先获取所有 key（去重）
        all_keys = set()
        for d in all_test_metrics:
            all_keys.update(d.keys())
        ## 再计算平均值
        final_metrics = {}
        for key in all_keys:
            # 确保该 key 在所有字典中都存在
            values = [d[key] for d in all_test_metrics if key in d]
            final_metrics[key] = np.mean(values)

        return final_metrics


    async def async_test(self, batch_dict, qa_manager, max_workers=4):
        """
        异步批量 rollout，采用 asyncio.as_completed 防长尾。
        """
        import torch
        import numpy as np
        import asyncio
        import logging
        from tqdm import tqdm

        questions = [item['question'] for item in batch_dict['extra_info']]  # [:1000]
        answers = [item['answer'] for item in batch_dict['extra_info']]  # [:1000]
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        def select_item_keep_dim(batch_dict, idx):
            item_dict = {}
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    item_dict[key] = value[idx:idx+1]
                elif isinstance(value, np.ndarray):
                    item_dict[key] = value[idx:idx+1]
                elif isinstance(value, list):
                    item_dict[key] = [value[idx]]
                else:
                    try:
                        item_dict[key] = value[idx:idx+1]
                    except Exception:
                        item_dict[key] = value
            return item_dict

        # Metrics 初始化
        final_f1_list, turn_number_list, retrieval_times_list = [], [], []
        stats_counter = {"QDS": 0, "QDP": 0, "Else": 0}
        record_metrics_batch = {
            metric: {agent: [] for agent in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']}
            for metric in ['reward', 'penalty', 'whole', 'times']
        }

        async def process_single_item(item_id, question, answer, semaphore):
            async with semaphore:
                item_dict = select_item_keep_dim(batch_dict, item_id)
                # 在后台线程运行同步方法
                _, record_metrics = await asyncio.to_thread(
                    self.async_rollout_single, qa_manager, item_dict, question, answer, 5, False
                )
                # 收集数据
                final_f1_list.append(record_metrics['final_f1'])
                turn_number_list.append(record_metrics['turn_number'])
                retrieval_times_list.append(record_metrics['retrieval_times'])
                # 统计workflow类型
                workflow_cate = record_metrics['workflow_cate']
                if workflow_cate == 'QDS':
                    stats_counter["QDS"] += 1
                elif workflow_cate == 'QDP':
                    stats_counter["QDP"] += 1
                else:
                    stats_counter["Else"] += 1
                for metric in ['reward', 'penalty', 'whole']:
                    for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                        record_metrics_batch[metric][agent_name].extend(record_metrics[metric][agent_name])
                # times
                for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                    record_metrics_batch['times'][agent_name].append(record_metrics['times'][agent_name])

        # 创建任务
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [
            asyncio.create_task(process_single_item(item_id, q, a, semaphore))
            for item_id, (q, a) in enumerate(zip(questions, answers))
        ]

        # 按完成顺序处理任务，减少长尾等待
        # for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async Rollout"):
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            except Exception as e:
                logging.error(f"Task failed: {e}")

        # # 计算最终 metrics
        test_reward_metrics = {
            "test_metrics/turn_number": np.mean(turn_number_list),
            "test_metrics/retrieval_times": np.mean(retrieval_times_list),
            "test_metrics/final_f1": np.mean(final_f1_list),
            "test_metrics/qds_rate": stats_counter["QDS"] / len(questions),
            "test_metrics/qdp_rate": stats_counter["QDP"] / len(questions),
            "test_metrics/else_rate": stats_counter["Else"] / len(questions)
        }
        for metric in ['reward', 'penalty', 'whole', 'times']:
            for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                if record_metrics_batch[metric][agent_name]:
                    test_reward_metrics[f"test_metrics/{metric}/{agent_name}"] = np.mean(record_metrics_batch[metric][agent_name]) if record_metrics_batch[metric][agent_name] else 0.0

        return batch, test_reward_metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}
    
    def assign_rewards(self, data, reward_value):
        # token and retrieval cost
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # batch是一个turn的, i 是遍历一个turn中batch size条数据
        for i in range(data.batch.batch_size[0]): 
            # get position
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            
            reward_tensor[i, valid_response_length - 1] = reward_value

        return reward_tensor

    def get_reward_tensor(self, data, predict_a, golden_a):
        cur_metrics = compute_scores([predict_a], [golden_a])
        if golden_a in ['yes', 'Yes', 'no', 'No']:
            cur_reward_value = cur_metrics['em']
        else:
            cur_reward_value = cur_metrics['f1']
        cur_reward_tensor = self.assign_rewards(data, cur_reward_value)
        data.batch['reward_tensor'] = cur_reward_tensor

        return data, cur_reward_value
    
    def get_reward_in_metrics(self, data, predict_a, golden_a):
        cur_metrics = compute_scores([predict_a], [golden_a])
        return cur_metrics
    
    def async_inference(self, input_dict, agent_name='DefaultAgent'):
        """
        input_dict - 当前第i条rollout的第j次input的dict
        """

        # 此处的batch实际为(i,j)
        batch: DataProto = DataProto.from_single_dict(input_dict)
        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        
        # async请求第i条rollout中的第j次inference
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        # # TODO: Decouple the DP balancing and mini-batching.
        # if self.config.trainer.balance_batch:
        #     self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # 此处return的是(i,j)单条的batch
        return batch

    # MIRAI multi-agent rollout
    def async_rollout_single(self, qa_manager, target_dict, question, answer, max_iters=5, show_process=False, is_train=True):
        
        def align_item_to_target_format(item_dict, target_dict):
            """
            将 item_dict 对齐到 target_dict 的格式：
            - 保证包含 target_dict 的所有 key
            - 类型与 target_dict 对应的值保持一致
            - 按 target 维度加 batch 维，但不截断数据
            """
            new_item_dict = {}

            for key, target_val in target_dict.items():
                # 没有该 key，直接复制 target
                if key not in item_dict:
                    val = copy.deepcopy(target_val)
                else:
                    val = item_dict[key]

                    # Torch Tensor
                    if isinstance(target_val, torch.Tensor):
                        if not isinstance(val, torch.Tensor):
                            val = torch.as_tensor(val, dtype=target_val.dtype)
                        if val.ndim < target_val.ndim:  # 对齐 batch 维
                            val = val.unsqueeze(0)

                    # NumPy 数组
                    elif isinstance(target_val, np.ndarray):
                        if not isinstance(val, np.ndarray):
                            val = np.array(val, dtype=target_val.dtype)
                        # 如果 target 是 shape=(1,), dtype=object，item也应该包一层批
                        if target_val.ndim > val.ndim:
                            val = np.expand_dims(val, axis=0)

                        # 特别处理 dtype=object且里面是list的情况（如 raw_prompt_ids）
                        if target_val.dtype == object:
                            # target: array([list([...])], dtype=object)
                            # item: array([...], dtype=object)
                            if (target_val.shape[0] == 1 and 
                                isinstance(target_val[0], list) and 
                                not isinstance(val[0], list)):
                                # 包成 list
                                val = np.array([list(val)], dtype=object)

                    # list
                    elif isinstance(target_val, list):
                        if not isinstance(val, list):
                            val = [val]
                        # target 是 batch list: [list([...])]
                        if isinstance(target_val[0], list) and not isinstance(val[0], list):
                            val = [val]

                    # 其他类型，按 target 类型包装 batch
                    else:
                        if isinstance(target_val, np.ndarray) and target_val.ndim > 0:
                            if not isinstance(val, np.ndarray):
                                val = np.array([val], dtype=target_val.dtype)
                        elif isinstance(target_val, list):
                            val = [val]

                new_item_dict[key] = val

            return new_item_dict

        # dataproto列表 [dataproto, penalty, reward(final_f1), whole(final_reward)]
        single_batch_list = []
        # rewards value列表
        record_metrics = {}
        for metric in ['reward', 'penalty', 'whole']:
            record_metrics[metric] = {}
            for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                record_metrics[metric][agent_name] = []
        record_metrics['times'] = {}
        for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
            record_metrics['times'][agent_name] = 0

        if show_process:
            print("=== question===\n{}".format(question))
            print("=== answer===\n{}".format(answer))
            print("\n\n")

        # 记录retrieval times
        record_metrics['retrieval_times'] = 0
        Only_set = {}

        # step 1: 规划模块
        Only_flag = False
        planning_agent_dict = qa_manager.trans_rawprompt_to_ids("PlanningAgent", question)
        planning_agent_dict_ = align_item_to_target_format(planning_agent_dict, target_dict)
        planning_data_ij = self.async_inference(planning_agent_dict_, agent_name='PlanningAgent')
        ## 解析得到planning workflow
        init_output_text = qa_manager.get_answers_text(planning_data_ij)[0]
        output_dict = qa_manager.planning_agent.parse_response(init_output_text)
        workflow_str = output_dict["workflow"]
        is_valid = output_dict["is_valid"]
        penalty = 0.0 if is_valid else -2.0
        single_batch_list.append([planning_data_ij, penalty, 0.0, penalty, 'PlanningAgent'])

        # 记录次数
        record_metrics['times']['PlanningAgent'] += 1

        if show_process:
            print(f"=== PlanningAgent 输出 ===\n{init_output_text}")
            print(f"=== PlanningAgent is_legal_format ===\n{is_valid}\n\n")

        # step 2: 如果 workflow 无效，设为R,AG / QDS
        if not is_valid:
            workflow_str = "QDS"
            
            if show_process:
                print("=== workflow 无效，设为QDS ===\n\n")


        # 为True的时候有DS，为False的时候不加DS
        have_DS = True

        # 统计workflow类型
        if workflow_str == "QDP":
            workflow_cate = 'QDP'
        elif workflow_str == "QDS":
            workflow_cate = 'QDS'
        else:
            workflow_cate = 'ELSE'
        record_metrics['workflow_cate'] = workflow_cate


        # step 3: QDP / QDS 特殊处理
        if workflow_str == "QDP" or workflow_str == "QDS":

            if workflow_str == "QDP":
                qdp_agent_dict = qa_manager.trans_rawprompt_to_ids("QueryDecompositionAgentParallel", question)
                qdp_agent_dict_ = align_item_to_target_format(qdp_agent_dict, target_dict)
                qdp_data_ij = self.async_inference(qdp_agent_dict_, agent_name='QueryDecompositionAgentParallel')
                ## 解析得到sub queries
                init_output_text = qa_manager.get_answers_text(qdp_data_ij)[0]
                output_dict = qa_manager.qdp_agent.parse_response(init_output_text, question)
                sub_queries = output_dict["subquestions"]
                is_valid = output_dict["is_valid"]
                penalty = 0.0 if is_valid else -2.0
                single_batch_list.append([qdp_data_ij, penalty, 0.0, penalty, 'QueryDecompositionAgentParallel'])

                # 记录次数
                record_metrics['times']['QueryDecompositionAgentParallel'] += 1

                if show_process:
                    print(f"=== QueryDecompositionAgentParallel 输出 ===\n{init_output_text}")
                    print(f"=== QueryDecompositionAgentParallel is_legal_format ===\n{is_valid}\n\n")

            else:
                qds_agent_dict = qa_manager.trans_rawprompt_to_ids("QueryDecompositionAgentSerial", question)
                qds_agent_dict_ = align_item_to_target_format(qds_agent_dict, target_dict)
                qds_data_ij = self.async_inference(qds_agent_dict_, agent_name='QueryDecompositionAgentSerial')
                ## 解析得到sub queries
                init_output_text = qa_manager.get_answers_text(qds_data_ij)[0]
                output_dict = qa_manager.qds_agent.parse_response(init_output_text, question)
                sub_queries = output_dict["subquestions"]
                is_valid = output_dict["is_valid"]
                penalty = 0.0 if is_valid else -2.0
                single_batch_list.append([qds_data_ij, penalty, 0.0, penalty, 'QueryDecompositionAgentSerial'])

                # 记录次数
                record_metrics['times']['QueryDecompositionAgentSerial'] += 1

                if show_process:
                    print(f"=== QueryDecompositionAgentSerial 输出 ===\n{init_output_text}")
                    print(f"=== QueryDecompositionAgentSerial is_legal_format ===\n{is_valid}\n\n")

            context = {
                "original_query": question,
                "type": workflow_str,
                "sub-query_docs_a": [
                    {"query": sq, "docs": [], "answer": ""} for sq in sub_queries
                ]
            }

            if show_process:
                print("=== context ===\n{}\n\n".format(context))

            # 需要根据qdp和qds的不同进行修改
            if workflow_str == "QDP":
                record_metrics['turn_number'] = 1 + 1
            else:
                record_metrics['turn_number'] = 1 + len(sub_queries)

            # 循环处理 sub-queries
            for i, sq_item in enumerate(context["sub-query_docs_a"]):
                current_sq = sq_item["query"]

                if workflow_str == "QDS":  # 先 rewrite
                    qr_agent_dict = qa_manager.trans_rawprompt_to_ids("QueryRewriteAgent", current_sq, context)
                    qr_agent_dict_ = align_item_to_target_format(qr_agent_dict, target_dict)
                    qr_data_ij = self.async_inference(qr_agent_dict_, agent_name='QueryRewriteAgent')
                    ## 解析得到sub queries
                    init_output_text = qa_manager.get_answers_text(qr_data_ij)[0]
                    output_dict = qa_manager.query_rewriter.parse_response(init_output_text, question)
                    rewritten_sq = output_dict["query_rewrited"]
                    is_valid = output_dict["is_valid"]
                    penalty = 0.0 if is_valid else -2.0
                    single_batch_list.append([qr_data_ij, penalty, 0.0, penalty, 'QueryRewriteAgent'])

                    # 记录次数
                    record_metrics['times']['QueryRewriteAgent'] += 1

                    # 更新query
                    sq_item['query'] = rewritten_sq

                    if show_process:
                        print(f"=== QueryRewriteAgent 输出 ===\n{init_output_text}")
                        print(f"=== QueryRewriteAgent is_legal_format ===\n{is_valid}\n\n")

                        print("=== context ===\n{}\n\n".format(context))

                # sub-planning
                sub_planning_agent_dict = qa_manager.trans_rawprompt_to_ids("SubPlanningAgent", sq_item['query'])
                sub_planning_agent_dict_ = align_item_to_target_format(sub_planning_agent_dict, target_dict)
                sub_planning_data_ij = self.async_inference(sub_planning_agent_dict_, agent_name='SubPlanningAgent')
                ## 解析得到planning workflow
                init_output_text = qa_manager.get_answers_text(sub_planning_data_ij)[0]
                output_dict = qa_manager.sub_planning_agent.parse_response(init_output_text)
                sp_workflow = output_dict["workflow"]
                sp_valid = output_dict["is_valid"]
                penalty = 0.0 if sp_valid else -2.0
                single_batch_list.append([sub_planning_data_ij, penalty, 0.0, penalty, 'SubPlanningAgent'])

                # 固定为R,DS,AG **************************************** no training planner + trained executors ****************************************
                # RRR
                sp_workflow = "R,DS,AG"
                sp_valid = True

                # 记录次数
                record_metrics['times']['SubPlanningAgent'] += 1

                if show_process:
                    print(f"=== SubPlanningAgent 输出 ===\n{init_output_text}")
                    print(f"=== SubPlanningAgent is_legal_format ===\n{sp_valid}\n\n")


                ## 如果不合法则给penalty后用默认workflow
                if not sp_valid:
                    sp_workflow = "R,AG"  # 默认

                    if show_process:
                        print("=== workflow 无效，设为R,AG ===\n\n")

                # 执行sub-workflow
                module_chain = [m.strip() for m in sp_workflow.split(",")]
                docs = []
                ans = ""

                for module in module_chain:
                    if module == "R":
                        # RetrievalTool
                        retrieval_input = sq_item["query"]
                        top_k = 5
                        docs = qa_manager.retrieval_tool.query(retrieval_input, top_k)
                        sq_item["docs"] = docs

                        record_metrics['retrieval_times'] += 1

                    elif module == "DS" and have_DS:
                        ds_agent_dict = qa_manager.trans_rawprompt_to_ids("DocumentSelectionAgent", sq_item["query"], docs)
                        ds_agent_dict_ = align_item_to_target_format(ds_agent_dict, target_dict)
                        ds_data_ij = self.async_inference(ds_agent_dict_, agent_name='DocumentSelectionAgent')
                        ## 解析得到sub queries
                        init_output_text = qa_manager.get_answers_text(ds_data_ij)[0]
                        output_dict = qa_manager.doc_selector.parse_response(init_output_text, total_docs=len(docs))
                        doc_ids = output_dict["docs_ids"]
                        is_valid = output_dict["is_valid"]
                        penalty = 0.0 if is_valid else -2.0
                        single_batch_list.append([ds_data_ij, penalty, 0.0, penalty, 'DocumentSelectionAgent'])
                        ## 根据doc id筛选docs
                        docs = [docs[idx] for idx in doc_ids]
                        sq_item["docs"] = docs

                        # 记录次数
                        record_metrics['times']['DocumentSelectionAgent'] += 1

                        if show_process:
                            print(f"=== DocumentSelectionAgent 输出 ===\n{init_output_text}")
                            print(f"=== DocumentSelectionAgent is_legal_format ===\n{is_valid}\n\n")

                    elif module == "AG":
                        ag_agent_dict = qa_manager.trans_rawprompt_to_ids("AnswerGenerationAgent", sq_item["query"], docs)
                        ag_agent_dict_ = align_item_to_target_format(ag_agent_dict, target_dict)
                        ag_data_ij = self.async_inference(ag_agent_dict_, agent_name='AnswerGenerationAgent')
                        ## 解析得到sub queries
                        init_output_text = qa_manager.get_answers_text(ag_data_ij)[0]
                        output_dict = qa_manager.answer_gen.parse_response(init_output_text)
                        ans = output_dict["answer"]
                        is_valid = output_dict["is_valid"]
                        penalty = 0.0 if is_valid else -2.0
                        single_batch_list.append([ag_data_ij, penalty, 0.0, penalty, 'AnswerGenerationAgent'])
                        ## 得到这一个sub-query的sub-answer
                        sq_item["answer"] = ans

                        # 记录次数
                        record_metrics['times']['AnswerGenerationAgent'] += 1

                        if show_process:
                            print(f"=== AnswerGenerationAgent 输出 ===\n{init_output_text}")
                            print(f"=== AnswerGenerationAgent is_legal_format ===\n{is_valid}\n\n")

                    if show_process:
                        print("=== context ===\n{}\n\n".format(context))

            # 所有 sub-query 的答案生成后，调用 summarization
            as_agent_dict = qa_manager.trans_rawprompt_to_ids(
                "AnswerSummarizationAgent", 
                question, 
                [sq["query"] for sq in context["sub-query_docs_a"]],
                [sq["answer"] for sq in context["sub-query_docs_a"]]
            )
            as_agent_dict_ = align_item_to_target_format(as_agent_dict, target_dict)
            as_data_ij = self.async_inference(as_agent_dict_, agent_name='AnswerSummarizationAgent')
            ## 解析得到sub queries
            init_output_text = qa_manager.get_answers_text(as_data_ij)[0]
            output_dict = qa_manager.ans_summary.parse_response(init_output_text)
            final_ans = output_dict["answer"]
            is_valid = output_dict["is_valid"]
            penalty = 0.0 if is_valid else -2.0
            single_batch_list.append([as_data_ij, penalty, 0.0, penalty, 'AnswerSummarizationAgent'])

            # 记录次数
            record_metrics['times']['AnswerSummarizationAgent'] += 1
            
            if show_process:
                print(f"=== AnswerSummarizationAgent 输出 ===\n{init_output_text}")
                print(f"=== AnswerSummarizationAgent is_legal_format ===\n{is_valid}\n\n")

        # step 4: 普通模块链执行
        else:
            module_chain = [m.strip() for m in workflow_str.split(",")]
            docs = []
            ans = ""
            record_metrics['turn_number'] = 1

            for module in module_chain:
                if module == "QR":
                    qr_agent_dict = qa_manager.trans_rawprompt_to_ids("QueryRewriteAgent", question, {})
                    qr_agent_dict_ = align_item_to_target_format(qr_agent_dict, target_dict)
                    qr_data_ij = self.async_inference(qr_agent_dict_, agent_name='QueryRewriteAgent')
                    ## 解析得到sub queries
                    init_output_text = qa_manager.get_answers_text(qr_data_ij)[0]
                    output_dict = qa_manager.query_rewriter.parse_response(init_output_text, question)
                    question = output_dict["query_rewrited"]
                    is_valid = output_dict["is_valid"]
                    penalty = 0.0 if is_valid else -2.0
                    single_batch_list.append([qr_data_ij, penalty, 0.0, penalty, 'QueryRewriteAgent'])

                    # 记录次数
                    record_metrics['times']['QueryRewriteAgent'] += 1

                    if show_process:
                        print(f"=== QueryRewriteAgent 输出 ===\n{init_output_text}")
                        print(f"=== QueryRewriteAgent is_legal_format ===\n{is_valid}\n\n")

                elif module == "R":
                    retrieval_input = question
                    top_k = 5
                    docs = qa_manager.retrieval_tool.query(retrieval_input, top_k)

                    record_metrics['retrieval_times'] += 1

                elif module == "DS" and have_DS:
                    ds_agent_dict = qa_manager.trans_rawprompt_to_ids("DocumentSelectionAgent", question, docs)
                    ds_agent_dict_ = align_item_to_target_format(ds_agent_dict, target_dict)
                    ds_data_ij = self.async_inference(ds_agent_dict_, agent_name='DocumentSelectionAgent')
                    ## 解析得到sub queries
                    init_output_text = qa_manager.get_answers_text(ds_data_ij)[0]
                    output_dict = qa_manager.doc_selector.parse_response(init_output_text, total_docs=len(docs))
                    doc_ids = output_dict["docs_ids"]
                    is_valid = output_dict["is_valid"]
                    penalty = 0.0 if is_valid else -2.0
                    single_batch_list.append([ds_data_ij, penalty, 0.0, penalty, 'DocumentSelectionAgent'])
                    ## 根据doc id筛选docs
                    docs = [docs[idx] for idx in doc_ids]

                    # 记录次数
                    record_metrics['times']['DocumentSelectionAgent'] += 1

                    if show_process:
                        print(f"=== DocumentSelectionAgent 输出 ===\n{init_output_text}")
                        print(f"=== DocumentSelectionAgent is_legal_format ===\n{is_valid}\n\n")

                elif module == "AG":
                    ag_agent_dict = qa_manager.trans_rawprompt_to_ids("AnswerGenerationAgent", question, docs)
                    ag_agent_dict_ = align_item_to_target_format(ag_agent_dict, target_dict)
                    ag_data_ij = self.async_inference(ag_agent_dict_, agent_name='AnswerGenerationAgent')
                    ## 解析得到sub queries
                    init_output_text = qa_manager.get_answers_text(ag_data_ij)[0]
                    output_dict = qa_manager.answer_gen.parse_response(init_output_text)
                    final_ans = output_dict["answer"]
                    is_valid = output_dict["is_valid"]
                    penalty = 0.0 if is_valid else -2.0
                    single_batch_list.append([ag_data_ij, penalty, 0.0, penalty, 'AnswerGenerationAgent'])

                    # 记录次数
                    record_metrics['times']['AnswerGenerationAgent'] += 1

                    if show_process:
                        print(f"=== AnswerGenerationAgent 输出 ===\n{init_output_text}")
                        print(f"=== AnswerGenerationAgent is_legal_format ===\n{is_valid}\n\n")

        # reward分配的过程
        if answer in ['yes', 'Yes', 'no', "No"]:
            final_reward = compute_scores([final_ans], [answer])['em']
        else:
            final_reward = compute_scores([final_ans], [answer])['f1']
        for item in single_batch_list:
            item[2] += final_reward
            item[3] += final_reward
        
        # record metrics
        # [dataproto, penalty, final_f1, final_reward, agent_name]
        for item in single_batch_list:
            agent_name = item[4]
            record_metrics['reward'][agent_name].append(item[2])
            record_metrics['penalty'][agent_name].append(item[1])
            record_metrics['whole'][agent_name].append(item[3])
            record_metrics['final_f1'] = final_reward
        
        # assign reward (reward_tensor)
        for item in single_batch_list:
            ## 计算reward_tensor

            # # 无turn number / retrieval times的penalty
            # cur_reward_tensor = self.assign_rewards(item[0], item[3])  # def assign_rewards(self, data, reward_value):

            # 有turn number / retrieval times的penalty
            alpha, beta = 0.0, 0.0
            def scale_score(n):
                # 如果超过3，强制设为3
                if n > 3:
                    n = 3
                # 0~3 映射到 0~1，即除以总区间长度 3
                return n / 3.0
            turn_penalty = scale_score(record_metrics['turn_number']-1)
            retrieval_penalty = scale_score(record_metrics['retrieval_times'])
            whole_reward = item[3] - alpha * turn_penalty - beta * retrieval_penalty

            cur_reward_tensor = self.assign_rewards(item[0], whole_reward)
            item[0].batch['reward_tensor'] = cur_reward_tensor
        
        # # ================= 优化全部模块 =================
        if not Only_flag:
            single_batch_list = [item[0] for item in single_batch_list]
        # ================= 只优化部分模块 =================
        else:
            single_batch_list = [item[0] for item in single_batch_list if item[4] in Only_set]

        # 返回内容为：dataproto list, metric dict
        return single_batch_list, record_metrics


    def async_rollout_batch(self, batch_dict, qa_manager, max_workers=4, parallel=True, show_process=False):
        """
        批量 rollout，支持串行与并行模式。

        Args:
            batch_dict: 输入批量数据
            qa_manager: QA 管理器
            parallel (bool): 是否启用并行模式
            max_workers (int): 并行线程数量（parallel=True 生效）
        """
        questions = [item['question'] for item in batch_dict['extra_info']]
        answers = [item['answer'] for item in batch_dict['extra_info']]
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        if show_process:
            questions = [questions[0]]
            answers = [answers[0]]

        def select_item_keep_dim(batch_dict, idx):
            """
            从 batch_dict 提取指定 idx 的数据，但保留 batch 维度 = 1，
            同时保证格式与原始 batch_dict 相同。
            """
            item_dict = {}
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    item_dict[key] = value[idx:idx+1]
                elif isinstance(value, np.ndarray):
                    item_dict[key] = value[idx:idx+1]
                elif isinstance(value, list):
                    item_dict[key] = [value[idx]]
                else:
                    try:
                        item_dict[key] = value[idx:idx+1]
                    except Exception:
                        item_dict[key] = value
            return item_dict

        # A. 唤醒模型（一次）
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.async_rollout_manager.wake_up()
        
        batch_list, final_f1_list, turn_number_list, retrieval_times_list = [], [], [], []
        stats_counter = {"QDS": 0, "QDP": 0, "Else": 0}
        record_metrics_batch = {metric: {agent: [] for agent in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']} for metric in ['reward', 'penalty', 'whole', 'times']}

        # 包装单条任务逻辑
        def process_single_item(item_id, question, answer):
            item_dict = select_item_keep_dim(batch_dict, item_id)
            item_dataproto = DataProto.from_single_dict(item_dict)
            return self.async_rollout_single(qa_manager, item_dict, question, answer, 5, show_process, True)

        if parallel:
            logging.info(f"Using parallel rollout with max_workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {
                    executor.submit(process_single_item, item_id, question, answer): item_id
                    for item_id, (question, answer) in enumerate(zip(questions, answers))
                }
                # for future in tqdm(as_completed(future_to_id), total=len(questions)):
                for future in as_completed(future_to_id):
                    item_id = future_to_id[future]
                    try:
                        item_batch_list, record_metrics = future.result()
                        # 收集数据
                        batch_list.extend(item_batch_list)
                        final_f1_list.append(record_metrics['final_f1'])
                        turn_number_list.append(record_metrics['turn_number'])
                        retrieval_times_list.append(record_metrics['retrieval_times'])
                        # 统计workflow类型
                        workflow_cate = record_metrics['workflow_cate']
                        if workflow_cate == 'QDS':
                            stats_counter["QDS"] += 1
                        elif workflow_cate == 'QDP':
                            stats_counter["QDP"] += 1
                        else:
                            stats_counter["Else"] += 1
                        'final_f1'
                        for metric in ['reward', 'penalty', 'whole']:
                            for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                                record_metrics_batch[metric][agent_name].extend(record_metrics[metric][agent_name])
                        # times
                        for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                            record_metrics_batch['times'][agent_name].append(record_metrics['times'][agent_name])
                    except Exception as e:
                        logging.error(f"Error in item {item_id}: {e}")
        else:
            logging.info("Using sequential rollout")
            for item_id, (question, answer) in enumerate(tqdm(zip(questions, answers), total=len(questions))):
                item_batch_list, record_metrics = process_single_item(item_id, question, answer)
                batch_list.extend(item_batch_list)
                final_f1_list.append(record_metrics['final_f1'])
                turn_number_list.append(record_metrics['turn_number'])
                retrieval_times_list.append(record_metrics['retrieval_times'])
                for metric in ['reward', 'penalty', 'whole']:
                    for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                        record_metrics_batch[metric][agent_name].extend(record_metrics[metric][agent_name])

        # 计算最终 train metrics
        train_reward_metrics = {
            "train_metrics/turn_number": np.mean(turn_number_list),
            "train_metrics/retrieval_times": np.mean(retrieval_times_list),
            "train_metrics/final_f1": np.mean(final_f1_list),
            "train_metrics/qds_rate": stats_counter["QDS"] / len(questions),
            "train_metrics/qdp_rate": stats_counter["QDP"] / len(questions),
            "train_metrics/else_rate": stats_counter["Else"] / len(questions)
        }
        for metric in ['reward', 'penalty', 'whole', 'times']:
            for agent_name in ['PlanningAgent', 'SubPlanningAgent', 'QueryRewriteAgent', 'QueryDecompositionAgentParallel', 'QueryDecompositionAgentSerial', 'DocumentSelectionAgent', 'AnswerGenerationAgent', 'AnswerSummarizationAgent']:
                if len(record_metrics_batch[metric][agent_name]) > 0:
                    train_reward_metrics[f"train_metrics/{metric}/{agent_name}"] = np.mean(record_metrics_batch[metric][agent_name]) if record_metrics_batch[metric][agent_name] else 0.0
        
        # B. 执行结束后 sleep
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.async_rollout_manager.sleep()

        return batch_list, train_reward_metrics


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            # val_metrics = self._validate()
            val_metrics = self.test_validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        from qa_manager.qa import MIRAIManager
        # init qa manager
        qa_manager = MIRAIManager(self.tokenizer, self.config)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                # timing_raw = {}

                # with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

                # show process
                if self.global_steps == 1 or self.global_steps % 10 == 0:
                    _, _ = self.async_rollout_batch(batch_dict, qa_manager, self.config.data.train_batch_size, True, True)
                

                batch_list, train_reward_metrics = self.async_rollout_batch(batch_dict, qa_manager, self.config.data.train_batch_size, True, False)
                metrics.update(train_reward_metrics)
                for dataproto in batch_list:
                    dataproto.meta_info = {}
                # 需要加入检测：当prompts的tensor的batch_size超过设定的最大值时，从list中抛弃这个dataproto。
                original_len = len(batch_list)
                batch_list[:] = [
                    dp for dp in batch_list 
                    if dp.batch['prompts'][0].size()[0] == self.config.data.max_prompt_length
                ]
                dropped = original_len - len(batch_list)
                if dropped > 0:
                    print(f"Dropped {dropped} dataproto(s) with prompts batch_size != {self.config.data.max_prompt_length}")
                # concat为大的batch
                batch = DataProto.concat(batch_list)

                # 补齐batch与进程数量的差 实现整除
                MIN_SIZE = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                total_size = batch.batch.batch_size[0]  # DataProto.batch 是 TensorDict
                remainder = total_size % MIN_SIZE
                if remainder != 0:
                    extra_needed = MIN_SIZE - remainder
                    # 随机生成 extra_needed 个索引
                    idx = torch.randint(0, total_size, (extra_needed,))
                    # 从原 DataProto 中选出这些索引的子批次
                    # 注意：DataProto 需要支持 __getitem__ 按索引返回子集
                    extra_batch = batch[idx]
                    # 拼接原 batch 与额外样本
                    batch = DataProto.concat([batch, extra_batch])

                # breakpoint()

                is_last_step = self.global_steps >= self.total_training_steps
                # with marked_timer("step", timing_raw):
                # generate a batch
                # with marked_timer("gen", timing_raw, color="red"):
                # if not self.async_rollout_mode:
                #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                # else:
                    
                #     breakpoint()
                #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)


                # # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)

                # if "response_mask" not in batch.batch.keys():
                #     batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # # TODO: Decouple the DP balancing and mini-batching.
                # if self.config.trainer.balance_batch:
                #     self._balance_batch(batch, metrics=metrics)

                # # compute global_valid tokens
                # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # with marked_timer("reward", timing_raw, color="yellow"):
                # # compute reward model score
                # if self.use_rm and "rm_scores" not in batch.batch.keys():
                #     reward_tensor = self.rm_wg.compute_rm_score(batch)
                #     batch = batch.union(reward_tensor)

                # if self.config.reward_model.launch_reward_fn_async:
                #     future_reward = compute_reward_async.remote(
                #         data=batch, config=self.config, tokenizer=self.tokenizer
                #     )
                # else:
                #     reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs
                # with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

                if "rollout_log_probs" in batch.batch.keys():
                    # TODO: we may want to add diff of probs too.
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))

                # breakpoint()
                
                if self.use_reference_policy:
                    # compute reference log_prob
                    # with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    # with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

                # breakpoint()

                # with marked_timer("adv", timing_raw, color="brown"):
                # we combine with rule-based rm
                # reward_extra_infos_dict: dict[str, list]
                # if self.config.reward_model.launch_reward_fn_async:
                #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)



                # 修改的内容get token_level_scores
                # batch.batch["token_level_scores"] = reward_tensor  # 源代码
                batch.batch["token_level_scores"] = batch.batch["reward_tensor"]  # 新修改的



                # if reward_extra_infos_dict:
                #     batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # Compute rollout importance sampling weights centrally (once per batch)
                # This corrects for mismatch between rollout policy and training policy
                # # Also computes mismatch metrics (KL, PPL, etc.)
                # batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                # # IS and mismatch metrics already have mismatch/ prefix
                # metrics.update(is_metrics)

                # compute advantages, executed on the driver process
                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )


                # update critic
                if self.use_critic:
                    # with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    # with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # # Log rollout generations if enabled
                # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                # if rollout_data_dir:
                #     self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    # with marked_timer("testing", timing_raw, color="green"):
                    # val_metrics: dict = self._validate()
                    val_metrics: dict = self.test_validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    # with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

                # with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

                # steps_duration = timing_raw["step"]
                # self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                # n_gpus = self.resource_pool_manager.get_n_gpus()
                # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
