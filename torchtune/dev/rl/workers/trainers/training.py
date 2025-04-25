import functools
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Union

from warnings import warn

import ray
import torch
import torch.distributed
import torch.nn as nn
import torchtune.training as training

from omegaconf import DictConfig, ListConfig, OmegaConf

from ray.util.queue import Full as QueueFull, Queue

from readerwriterlock import rwlock
from tensordict import is_tensorclass, NonTensorData, TensorClass, TensorDict

from torch.optim import Optimizer

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from torchrl.data import LazyStackStorage, RayReplayBuffer
from torchrl.envs import LLMEnv
from torchtune import config, generation, modules, rlhf, utils
from torchtune.dev.rl.datatypes import Trajectory
from torchtune.dev.rl.rewards import batched_rewards
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory
from torchtune.dev.rl.utils import stateless_init_process_group
from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

from torchtune.training import DummyProfiler, PROFILER_KEY

from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

log = utils.get_logger("DEBUG")


@ray.remote(num_cpus=8, num_gpus=1)
class TrainingWorker:
    """
    A Ray actor responsible for training a model using the GRPO (Generalized Reward Policy Optimization)
    algorithm in a distributed setting.
    This class leverages PyTorch's distributed training capabilities with FSDP and interacts
    with a replay buffer and VLLM engines.

    Args:
        cfg (DictConfig): Configuration object containing training parameters.
        environment_variables (dict): Environment variables for distributed training setup.
        replay_buffer: Shared replay buffer for sampling trajectories.
    """

    def __init__(self, cfg, environment_variables, replay_buffer):
        import torch

        self.replay_buffer = replay_buffer
        self.cfg = cfg

        # Device and dtype setup
        self._device = utils.get_device(device=cfg.training.device_type)
        self._dtype = training.get_dtype(cfg.training.dtype, device=self._device)

        device_type = self.cfg.training.device_type
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", True)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, but training does not use cuda. Setting to False."
            )
            self._log_peak_memory_stats = False

        if self._dtype == torch.float16:
            raise ValueError(
                "Full fp16 training is not supported. Use bf16 or fp32 instead."
            )

        # Logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)

        self.fsdp_cpu_offload = cfg.training.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        # Distributed training setup: Simulate torchrun environment
        for var in environment_variables:
            os.environ[var] = str(environment_variables[var])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        world_size = torch.distributed.get_world_size()
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.fsdp_group = torch.distributed.new_group(
            ranks=list(range(self.world_size - 1)), use_local_synchronization=True
        )
        self.device_mesh = torch.distributed.device_mesh.DeviceMesh.from_group(
            self.fsdp_group, device_type="cuda"
        )

        self._is_rank_zero = self.rank == 0

        # Training configuration
        self._resume_from_checkpoint = cfg.training.resume_from_checkpoint
        self._clip_grad_norm = cfg.training.get("clip_grad_norm", None)

        # Activation checkpointing and offloading
        self._enable_activation_checkpointing = cfg.training.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.training.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # Recipe state
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.training.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_dialog_turns = cfg.orchestration.num_steps
        self._epochs_run = 0
        self._rng = torch.Generator(self._device).manual_seed(
            self.seed
        )  # TODO: Verify if needed for GRPO

        # RL parameters
        self._ppo_epochs = cfg.training.ppo_epochs

        # Model and optimizer setup
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._compile = cfg.training.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.training.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._optimizer = self._setup_optimizer(cfg_optimizer=cfg.training.optimizer)
        self._loss_fn = config.instantiate(cfg.training.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        # TODO: generalize this to any chunked loss
        # set num_output_chunks for model
        if self._loss_fn.__class__.__name__ == "GRPOWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        self._tokenizer = config.instantiate(self.cfg.tokenizer)

        # FIXME: need to get _steps_per_epoch when dataloader is no longer per fsdp worker but instead wrapped in vLLM
        # self._lr_scheduler = self._setup_lr_scheduler(
        #     cfg_lr_scheduler=cfg.get("lr_scheduler", None),
        #     num_training_steps=self.total_epochs * self._steps_per_epoch,
        #     last_epoch=self.global_step - 1,
        # )
        self._lr_scheduler = None

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))
        self._steps_before_sync = cfg.training.steps_before_sync

        # Initialize policy version for tracking age of trajectories
        self.policy_version = 0
        self.metric_logger = None  # Placeholder for the logger

        # Debugging configuration
        self.debug_logging_enabled = cfg.get("debug_logging_enabled", True)
        self.debug_num_samples_per_step = cfg.get("debug_num_samples_per_step", 2)

        log.info("Done setup")

    def set_metric_logger(self, logger):
        """Store the MetricLoggerWorker handle for logging metrics."""
        if self._is_rank_zero:
            log.info(f"Setting metric logger {logger} for rank {self.rank}")
            self._metric_logger = logger

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """Set up the profiler based on the configuration. Returns DummyProfiler if not enabled."""
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)
        utils.log_rank_zero(log, f"Profiler config after instantiation: {profiler_cfg}")
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]
        return profiler

    # FIXME: do we need this?
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self._model(*args, **kwargs)

    def init_model_update_group(self, master_address, master_port, rank, world_size):
        """Initialize the model update group for weight synchronization."""
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            torch.device("cuda:0"),  # FIXME: Hardcoded device
        )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """Update recipe state from checkpoint."""
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """
        from torchtune.training import disable_dropout

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        time_setup_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        fsdp_shard_conditions = [
            partial(training.get_shard_conditions, names_to_match=custom_sharded_layers)
        ]
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=True,
            dp_mesh=self.device_mesh,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - time_setup_start:.2f} secs",
        )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero and self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)

        # synchronize before training begins
        torch.distributed.barrier(group=self.fsdp_group)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict=None
    ) -> Optional[Optimizer]:
        """Initialize the optimizer."""
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

        # TODO: does this work?
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """Perform a single GRPO optimization step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (GRPOTrajectory): a batch of trajectories
            context_length (int): the length of the context window

        Returns:
            GRPOStats: Instance of :class:`~torchtune.rlhf.GRPOStats`
        """
        # Create an output mask to avoid computing model.output on tokens we won't train
        # FIXME: when bsz>1, don't we have multiple context_length?
        # FIXME: because of chunked CE, the outout of pi_logits is a chunked list, so masking after the fact is
        # more annoying. Masking before the chunking is easier, but we have to figure out masking for bsz>1
        output_mask = torch.zeros_like(
            trajectory.query_responses, dtype=torch.bool, device=self._device
        )
        output_mask[:, context_length - 1 : -1] = True

        # call model
        with self.activations_handling_ctx:
            pi_logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
                output_mask=output_mask,
            )

        # apply to targets
        targets = trajectory.query_responses[:, context_length:]

        # Compute GRPO loss
        loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs = self._loss_fn(
            pi_logits=pi_logits,
            targets=targets,
            ref_logprobs=trajectory.ref_logprobs,
            advantages=trajectory.advantages,
            padding_masks=~trajectory.response_padding_masks,
        )

        with torch.no_grad():
            mask = ~trajectory.response_padding_masks  # True for non-padded tokens
            approx_policy_kls = (
                0.5 * ((pi_logprobs - trajectory.logprobs)[mask].pow(2)).mean()
            )

        # Handle trajectory return based on debug mode
        metadata = {}
        if self.debug_logging_enabled:
            metadata["pi_logprobs"] = pi_logprobs.detach()

        stats = GRPOStats(
            loss=loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            ratios=ratios,
            clipfrac=clipfrac,
            approx_policy_kls=approx_policy_kls,
            metadata=metadata,
        )

        del pi_logits, pi_logprobs
        torch.cuda.empty_cache()  # TODO: Test if this is needed
        loss.backward()

        return stats

    def set_vllm_engines(self, engines):
        """Set the vLLM engines for weight synchronization."""
        self._vllm_engines = engines

    def cleanup_after_step(
        self, trajectory: GRPOTrajectory, l_grpo_stats: List[GRPOStats]
    ) -> None:
        """Clean up memory after a training step."""
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats

    def _log_metrics(
        self,
        step_idx,
        trajectory,
        grpo_stats,
        total_step_time,
        time_grpo_steps,
        time_waiting_buffer,
        time_weight_sync,
        time_weight_gather,
        number_of_tokens,
        padded_tokens_percentage,
        policy_age,
        train_replay_buffer_size,
    ):
        """Log training metrics, only on rank zero."""
        if not self._is_rank_zero:
            return

        # Stack list[GRPOStats]
        tensor_fields = [
            "loss",
            "policy_loss",
            "kl_loss",
            "ratios",
            "clipfrac",
            "approx_policy_kls",
        ]
        grpo_stats_stacked = GRPOStats(
            **{
                field: torch.stack([getattr(stats, field) for stats in grpo_stats])
                for field in tensor_fields
            }
        )

        log_dict = {}
        if self._log_peak_memory_stats:
            memory_stats = training.get_memory_stats(device=self._device)
            log_dict.update(
                {
                    f"train_actor_performance/memory/{k}": v
                    for k, v in memory_stats.items()
                }
            )

        log_dict.update(
            {
                "train_actor_training/loss": grpo_stats_stacked.loss.mean().item(),
                "train_actor_training/policy_loss": grpo_stats_stacked.policy_loss.mean().item(),
                "train_actor_training/num_stop_tokens": trajectory.response_padding_masks.any(
                    -1
                )
                .sum()
                .item(),
                "train_actor_training/kl_loss": grpo_stats_stacked.kl_loss.mean().item(),
                "train_actor_training/ratios": grpo_stats_stacked.ratios.mean().item(),
                "train_actor_training/clipfrac": grpo_stats_stacked.clipfrac.mean().item(),
                "train_actor_training/approx_policy_kls": grpo_stats_stacked.approx_policy_kls.mean().item(),
                "train_actor_training/response_lengths": trajectory.seq_lens.float()
                .mean()
                .item(),
            }
        )

        log_dict.update(
            {
                "train_actor_performance/total_step_time (s)": total_step_time,
                "train_actor_performance/time_grpo_steps (s)": time_grpo_steps,
                "train_actor_performance/pct_time_grpo_steps (%)": (
                    time_grpo_steps / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/tokens_per_second": (
                    number_of_tokens / total_step_time if total_step_time > 0 else 0
                ),
                "train_actor_performance/time_weight_sync (s)": time_weight_sync,
                "train_actor_performance/pct_time_weight_sync (%)": (
                    time_weight_sync / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/padded_tokens_percentage (%)": padded_tokens_percentage,
                "train_actor_performance/time_waiting_buffer (s)": time_waiting_buffer,
                "train_actor_performance/pct_time_waiting_buffer (%)": (
                    time_waiting_buffer / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "train_actor_performance/time_weight_gather (s)": time_weight_gather,
                "train_actor_performance/pct_time_weight_gather (%)": (
                    time_weight_gather / total_step_time * 100
                    if total_step_time > 0
                    else 0
                ),
                "queues/train_actor_policy_age_mean": policy_age,
                "queues/train_replay_buffer_size": train_replay_buffer_size,
            }
        )

        ray.get(self._metric_logger.log_dict.remote(log_dict, step=step_idx))

    def _log_debug_table(
        self,
        grpo_trajectory: GRPOTrajectory,
        grpo_stats: GRPOStats,
        metadata: Dict[str, Any],
        context_length: int,
    ) -> None:
        """
        Log debugging tables to WandB with per-token and per-sample features using dictionaries.

        ATTENTION:
        - To see multiple tables in the logs check https://github.com/wandb/wandb/issues/6286#issuecomment-2734616342
        - To visualize the columns in wandb, click on 'Columns' in the bottom right, then add them to the graph."

        Args:
            grpo_trajectory (GRPOTrajectory): Object containing sequence data (query_responses, logprobs, etc.).
            grpo_stats (GRPOStats): Object with GRPO-related statistics (loss, policy_loss, etc.).
            metadata (Dict[str, Any]): Dictionary containing rewards, successes, policy_version, etc.
            context_length (int): Integer length of the prompt context.
        """

        def _log_table(data: list, table_name: str) -> None:
            """Helper function to log table data to WandB."""
            if data:
                log.info(f"Logging {table_name} for step {self._steps_run}")
                columns = list(data[0].keys())
                table_data = []
                for row in data:
                    table_data.append([row[col] for col in columns])
                ray.get(
                    self._metric_logger.log_table.remote(
                        table_data, columns, table_name, step=self._steps_run
                    )
                )
            else:
                log.info(f"Failed to log {table_name} for step {self._steps_run}")

        # Determine the number of samples to log
        num_samples = min(
            self.debug_num_samples_per_step, grpo_trajectory.query_responses.size(0)
        )

        # Extract response tokens
        targets = grpo_trajectory.query_responses[:, context_length:]
        per_sample_table_data = []
        per_token_table_data = []

        # Iterate over each sample
        for idx in range(num_samples):
            func_names = metadata["reward_metadata"][idx]["func_names"]
            sequence_id = metadata["sequence_ids"][idx]
            seq_len = grpo_trajectory.seq_lens[idx].item()

            prompt_tokens = grpo_trajectory.query_responses[
                idx, :context_length
            ].tolist()

            response_tokens = grpo_trajectory.query_responses[
                idx, context_length:
            ].tolist()

            prompt = self._tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            response = self._tokenizer.decode(
                response_tokens, skip_special_tokens=False
            )
            decoded_tokens = [
                self._tokenizer.decode([token], skip_special_tokens=False)
                for token in response_tokens
            ]

            ### Per-Sample Data
            per_sample_dict = {}
            per_sample_dict["Sequence ID"] = sequence_id
            per_sample_dict["prompt"] = prompt
            per_sample_dict["response"] = response
            per_sample_dict["answers"] = grpo_trajectory.answers[idx]
            per_sample_dict["policy_version"] = metadata["policy_version"][idx]

            # Add rewards dynamically based on func_names
            rewards = metadata["rewards"][idx].tolist()
            for func_name, reward in zip(func_names, rewards):
                per_sample_dict[f"reward_{func_name}"] = reward

            # Add successes dynamically based on func_names
            successes = metadata["successes"][idx].tolist()
            for func_name, success in zip(func_names, successes):
                per_sample_dict[f"success_{func_name}"] = success

            # Add GRPO statistics, handling per-sample vs. scalar cases
            # TODO: currently has one scalar per batch. We should enable a scalar per sentence.
            # Need to refactor loss reduction to enable that.
            stat_attrs = [
                "loss",
                "policy_loss",
                "kl_loss",
                "ratios",
                "clipfrac",
                "approx_policy_kls",
            ]
            for attr_name in stat_attrs:
                stat = getattr(grpo_stats, attr_name)
                per_sample_dict[attr_name] = (
                    stat[idx].item() if stat.dim() > 0 else stat.item()
                )

            # Add advantages
            per_sample_dict["advantages"] = grpo_trajectory.advantages[idx].item()

            # Add sequence metrics
            per_sample_dict["response_length"] = seq_len
            per_sample_dict["context_length"] = context_length
            per_sample_dict["has_stop_token"] = (
                grpo_trajectory.response_padding_masks[idx].any().item()
            )

            # Check if prompt tokens are included in loss (should be 0)
            per_sample_dict["prompt_masking_is_positive (should be 0)"] = (
                grpo_trajectory.response_padding_masks[idx, :context_length]
                .sum()
                .item()
            )

            # Check if tokens beyond seq_len are included in loss (should be 0)
            if context_length + seq_len < grpo_trajectory.query_responses.shape[1]:
                beyond_seq_len = (
                    grpo_trajectory.response_padding_masks[
                        idx, context_length + seq_len :
                    ]
                    .sum()
                    .item()
                )
            else:
                beyond_seq_len = 0
            per_sample_dict["beyond_seq_len_masking_is_positive (should be 0)"] = (
                beyond_seq_len
            )

            per_sample_dict["num_tokens_response"] = seq_len
            per_sample_dict["step"] = self._steps_run

            # Append the dictionary to the per-sample table data
            per_sample_table_data.append(per_sample_dict)

            ### Per-Token Data
            for pos in range(seq_len):
                per_token_dict = {}
                per_token_dict["Sequence ID"] = sequence_id
                per_token_dict["Token Position"] = pos  # TODO: maybe remove?
                per_token_dict["Token ID"] = targets[idx, pos].item()
                per_token_dict["Decoded Token"] = decoded_tokens[pos]
                per_token_dict["generated_logprob"] = grpo_trajectory.logprobs[
                    idx, pos
                ].item()
                per_token_dict["ref_logprob"] = grpo_trajectory.ref_logprobs[
                    idx, pos
                ].item()
                per_token_dict["pi_logprob"] = (
                    grpo_stats.metadata["pi_logprobs"][idx, pos].item()
                    if grpo_stats.metadata
                    else None
                )
                per_token_dict["abs_diff_pi_ref_logprob"] = abs(
                    per_token_dict["pi_logprob"] - per_token_dict["ref_logprob"]
                )
                per_token_dict["abs_diff_pi_generated_logprob"] = abs(
                    per_token_dict["pi_logprob"] - per_token_dict["generated_logprob"]
                )
                per_token_dict["mask"] = int(
                    ~grpo_trajectory.response_padding_masks[idx, pos]
                )
                per_token_dict["step"] = self._steps_run

                # Append the dictionary
                per_token_table_data.append(per_token_dict)

        # Log tables to WandB
        _log_table(per_sample_table_data, "per_sample_debug_table")
        _log_table(per_token_table_data, "per_token_debug_table")

    def train(self):
        """Execute the GRPO training loop."""
        training.cleanup_before_training()
        self._optimizer.zero_grad()
        self._profiler.start()

        while self._steps_run < self._total_dialog_turns:

            # Memory profiling start
            if (
                self._is_rank_zero
                and self.profiler_profile_memory
                and self._steps_run
                == self.profiler_wait_steps + self.profiler_warmup_steps
            ):
                torch.cuda.memory._record_memory_history()

            time_step_start = time.perf_counter()

            # Fetch trajectory from queue
            time_waiting_buffer_start = time.perf_counter()
            train_replay_buffer_size = None
            if self._is_rank_zero:
                train_replay_buffer_size = len(self.replay_buffer)

            while not len(self.replay_buffer):
                log.info("waiting for replay buffer")
                time.sleep(1.0)

            trajectory = self.replay_buffer.sample().to(self._device)
            time_waiting_buffer = time.perf_counter() - time_waiting_buffer_start
            if self._is_rank_zero:
                log.info(f"{self.rank=} got from queue traj {trajectory}")

            # Prepare trajectory for optimization
            prepared_trajectory, context_length, metadata = self._prepare_trajectory(
                trajectory
            )

            # Perform GRPO optimization
            time_grpo_steps_start = time.perf_counter()
            grpo_stats: list[GRPOStats] = []
            for _ in range(self._ppo_epochs):
                # step
                step_stats = self.grpo_step(
                    prepared_trajectory,
                    context_length,
                )
                grpo_stats.append(step_stats)
                # grad norm
                if self._clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), max_norm=float(self._clip_grad_norm)
                    )

                # optimizer step
                torch.distributed.barrier(group=self.fsdp_group)
                self._optimizer.step()
                torch.distributed.barrier(group=self.fsdp_group)
                self._optimizer.zero_grad(set_to_none=True)

                # scheduler
                self.global_step += 1
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

            log.info(f"{self.rank=} finished step {self._steps_run}")
            time_grpo_steps = time.perf_counter() - time_grpo_steps_start
            self._steps_run += 1

            # Log debug table if enabled, using pi_logprobs from the first epoch
            if (
                self.debug_logging_enabled
                and self._is_rank_zero
                and self._steps_run % self._log_every_n_steps == 0
            ):
                self._log_debug_table(
                    prepared_trajectory, grpo_stats[0], metadata, context_length
                )

            # Synchronize weights
            time_weight_sync = time_weight_gather = 0
            if self._steps_run % self._steps_before_sync == 0:
                utils.log_rank_zero(log, "started weight gather")
                torch.distributed.barrier(group=self.fsdp_group)
                time_weight_gather_start = time.perf_counter()
                new_sd = {
                    k: v.full_tensor() for k, v in self._model.state_dict().items()
                }
                torch.cuda.synchronize()
                time_weight_gather = time.perf_counter() - time_weight_gather_start
                utils.log_rank_zero(log, f"Done gather in {time_weight_gather}")
                time_sync_start = time.perf_counter()
                utils.log_rank_zero(log, "started weight sync")
                self.sync_weights(new_sd)
                del new_sd
                time_weight_sync = time.perf_counter() - time_sync_start
                utils.log_rank_zero(log, f"Done sync in {time_weight_sync}")

            # Log metrics
            total_step_time = time.perf_counter() - time_step_start
            if self._is_rank_zero and self._steps_run % self._log_every_n_steps == 0:
                log.info("logging metrics")
                self._log_metrics(
                    step_idx=self.global_step,
                    trajectory=prepared_trajectory,
                    grpo_stats=grpo_stats,
                    total_step_time=total_step_time,
                    time_grpo_steps=time_grpo_steps,
                    time_waiting_buffer=time_waiting_buffer,
                    time_weight_sync=time_weight_sync,
                    time_weight_gather=time_weight_gather,
                    number_of_tokens=metadata["number_of_tokens"],
                    padded_tokens_percentage=metadata["padded_tokens_percentage"],
                    policy_age=metadata["avg_policy_age"],
                    train_replay_buffer_size=train_replay_buffer_size,
                )
                log.info("done logging metrics")

            self.cleanup_after_step(trajectory, grpo_stats)

            # Memory profiling stop
            self._profiler.step()
            if (
                self._is_rank_zero
                and self.profiler_profile_memory
                and self._steps_run
                == self.profiler_wait_steps
                + self.profiler_warmup_steps
                + self.profiler_active_steps
            ):
                torch.cuda.memory._record_memory_history(enabled=None)

            torch.distributed.barrier(group=self.fsdp_group)

        self._profiler.stop()

    def register_parameter_server(self, parameter_server):
        assert self._is_rank_zero
        self.parameter_server = parameter_server

    def get_model_metadata(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_sd_metadata = {
            k: (v.shape, v.dtype) for k, v in self._model.state_dict().items()
        }
        fake_sd = dict()
        with FakeTensorMode():
            for k, (shape, dtype) in fake_sd_metadata.items():
                fake_sd[k] = torch.empty(shape, dtype=dtype, device="cuda")

        hf_fake_sd = qwen2_tune_to_hf(fake_sd, num_heads=16, num_kv_heads=2, dim=2048)

        return {k: (v.dtype, v.shape) for k, v in hf_fake_sd.items()}

    def sync_weights(self, new_sd):
        self.policy_version += 1
        if self._is_rank_zero:
            new_sd = qwen2_tune_to_hf(new_sd, num_heads=16, num_kv_heads=2, dim=2048)
            ray.get(self.parameter_server.acquire_state_dict_lock.remote())
            self.parameter_server.receive_from_trainer.remote()
            for i, (k, v) in enumerate(new_sd.items()):
                # dst is global rank, can switch to group_dst arg if not 2.5.1
                torch.distributed.send(v, dst=self.world_size - 1)

            ray.get(self.parameter_server.release_state_dict_lock.remote())

    def _prepare_trajectory(
        self, raw_trajectory: Trajectory
    ) -> tuple[GRPOTrajectory, int, Dict[str, Any]]:
        """Process raw trajectory, compute rewards, and prepare for optimization.

        Args:
            raw_trajectory: The trajectory sampled from the replay buffer.

        Returns:
            trajectory (GRPOTrajectory): Processed trajectory for GRPO optimization.
            context_length (int): Length of the context sequence.
            metadata (dict): Metadata for logging
        """
        # Extract components from raw trajectory
        query_responses = raw_trajectory.query_responses
        responses = raw_trajectory.responses
        logprobs = raw_trajectory.logprobs
        ref_logprobs = raw_trajectory.ref_logprobs
        query_response_padding_masks = raw_trajectory.query_response_padding_masks
        seq_lens = raw_trajectory.seq_lens
        advantages = raw_trajectory.advantages
        answers = raw_trajectory.answers

        # Compute padded tokens percentage
        total_tokens = query_responses.numel()
        padded_tokens = (query_responses == self._tokenizer.pad_id).sum().item()
        padded_tokens_percentage = (
            (padded_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        )
        number_of_tokens = seq_lens.sum().item()

        # Truncate sequences at first stop token
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses,
            torch.tensor(self._tokenizer.stop_tokens, device=self._device),
            self._tokenizer.pad_id,
        )

        # Generate masks and position IDs
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        context_length = query_responses.shape[1] - responses.shape[1]
        del query_response_padding_masks

        # Create GRPOTrajectory
        prepared_trajectory = GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
            answers=answers,
        )

        # Metadata for logging
        if isinstance(raw_trajectory.policy_version, list):
            avg_policy_age = self.policy_version - (
                sum(raw_trajectory.policy_version) / len(raw_trajectory.policy_version)
            )
        else:
            avg_policy_age = self.policy_version - raw_trajectory.policy_version

        metadata = {
            "padded_tokens_percentage": padded_tokens_percentage,
            "number_of_tokens": number_of_tokens,
            "avg_policy_age": avg_policy_age,
            "sequence_ids": raw_trajectory.sequence_ids,
            "policy_version": raw_trajectory.policy_version,
            "rewards": raw_trajectory.rewards,
            "successes": raw_trajectory.successes,
            "reward_metadata": raw_trajectory.reward_metadata,
            "query_response_padding_masks": raw_trajectory.query_response_padding_masks,
        }

        return prepared_trajectory, context_length, metadata

    def cleanup(self) -> None:
        """Close the metric logger on rank zero."""
        if self._is_rank_zero:
            self._metric_logger.close.remote()
