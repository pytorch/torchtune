import os
import socket
from functools import partial

import ray
import torch
import torch.nn as nn
import torchtune
import torchtune.training as training

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torchtune import utils
from torchtune.models import qwen2_5
from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

from vllm import LLM, SamplingParams
from vllm.worker.worker import Worker


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class vLLMRayWorker(Worker):
    """vLLM worker for Ray."""

    def init_weight_update_group(self, master_address, master_port, rank, world_size):
        from vllm.distributed.parallel_state import get_world_group

        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=1, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


# Define the worker class
@ray.remote(num_cpus=16, num_gpus=1)
class TrainWorker:
    def __init__(self, environment_variables):
        # Need to setup FSDP and multple workers?
        self.device = utils.get_device(device="cuda")
        self.dtype = training.get_dtype("bf16", device=self.device)
        import os

        for var in environment_variables:
            os.environ[var] = environment_variables[var]

        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        self.rank = os.environ["RANK"]
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
        print(self.device_mesh)
        self.model = self.setup_model(self.device, self.dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_state_dict(self):
        return self.model.state_dict()

    def init_model_update_group(self, master_address, master_port, rank, world_size):
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            # FIXME: hardcoding, not sure if this is right
            torch.device(f"cuda:0"),
        )

    def get_metadata_for_broadcast(self):
        vllm_format_sd = self.new_sd
        new_sd = {}
        for k, v in vllm_format_sd.items():
            new_sd[k] = (v.shape, v.dtype)
        return new_sd

    def all_gather(self):
        new_sd = {}
        for i, (k, v) in enumerate(self.model.state_dict().items()):
            new_sd[k] = v.full_tensor()
            if i == 0:
                print(
                    f"DTensor.local shape {v._local_tensor.shape}, DTensor.full_tensor shape {new_sd[k].shape}"
                )
        new_sd = qwen2_tune_to_hf(new_sd, num_heads=16, num_kv_heads=2, dim=2048)
        # FIXME: is this sus
        self.new_sd = new_sd

    def broadcast_key_to_vllm(self, key):
        self.model_update_group.broadcast(
            self.new_sd[key], src=1, stream=torch.cuda.current_stream()
        )

    def get_rank(self):
        gpu_ids = ray.get_gpu_ids()
        return gpu_ids[0]

    def setup_model(self, device, dtype, compile_model=False, cpu_offload=False):
        with training.set_default_dtype(dtype), torch.device("meta"):
            model = qwen2_5.qwen2_5_3b()

        if compile_model:
            training.compile_model(model, verbose=False)

        # For FSDP sharding
        fsdp_shard_conditions = [partial(training.get_shard_conditions)]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=cpu_offload,
            reshard_after_forward=True,
            dp_mesh=self.device_mesh["fsdp"],
        )

        with training.set_default_dtype(dtype), device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        model_sd = torchtune.training.FullModelHFCheckpointer(
            checkpoint_dir="/tmp/Qwen2.5-3B",
            checkpoint_files=[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            recipe_checkpoint=None,
            output_dir="/tmp/torchtune/qwen2_5_3B/ray_vllm_test",
            model_type="QWEN2",
        ).load_checkpoint()[training.MODEL_KEY]

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_sd,
            device,
            strict=True,
            cpu_offload=cpu_offload,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)
        training.disable_dropout(model)
        return model


class RayGRPORecipe:
    def __init__(self, cfg):
        self.ray_resources = cfg.ray_resources
        self.vllm_tensor_parallel_size = cfg.vllm_tensor_parallel_size
        self.fsdp_world_size = cfg.fsdp_world_size

    def start_ray(self):
        ray.init(
            num_cpus=self.ray_resources.num_cpus, num_gpus=self.ray_resources.num_gpus
        )

    def setup(self):
        # ---- Create models ---- #
        self.vllm_model = _create_vllm_rollout(
            tensor_parallel_size=self.tensor_parallel_size
        )
        addr = get_ip()
        train_workers_port = get_open_port()
        self.pytorch_train_workers = _create_train_workers(
            world_size=self.fsdp_world_size, addr=addr, port=train_workers_port
        )
        # ---- Create PG to sync weights to rollout model ---- #
        # Ensure there is not process group initialized in the main process
        assert not torch.distributed.is_initialized()
        weight_update_port = get_open_port()
        self.vllm_model.collective_rpc.remote(
            "init_weight_update_group",
            args=(addr, weight_update_port, 0, self.vllm_tensor_parallel_size + 1),
        )
        self.pytorch_train_workers[0].init_model_update_group.remote(
            addr,
            weight_update_port,
            self.vllm_tensor_parallel_size,
            self.fsdp_world_size,
        )
        ray.get(handle)  # Wait for the weight update group to be initialized
        # ---- Create dataloader ---- #
        collate_fn = partial(
            _get_component_from_path(cfg.collate_fn),
            padding_idx=self._tokenizer.pad_id,
        )
        self.dataloader = _create_dataloader(
            collate_fn,
            batch_size=cfg.batch_size,
        )
        # ---- Create optimizer ---- #
        self.optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

    def _create_vllm_rollout(tensor_parallel_size: int):
        # Create placement group (still kinda need to figure out what this does)
        pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * tensor_parallel_size)
        ray.get(pg_inference.ready())
        scheduling_inference = PlacementGroupSchedulingStrategy(
            placement_group=pg_inference,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        # Initialize the remote model
        llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=scheduling_inference,
        )(LLM).remote(
            model="Qwen/Qwen2.5-3B",
            enforce_eager=True,
            worker_cls=vLLMRayWorker,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
        )
        return llm

    def _create_train_workers(
        world_size: int, addr: str, port: int
    ) -> List[PyTorchTrainWorker]:
        workers = []
        for i in range(world_size):
            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(i),
                "WG_BACKEND": "ray",
                "MASTER_ADDR": addr,
                "MASTER_PORT": str(port),
            }
            worker = PyTorchTrainWorker.remote(env_vars)
            workers.append(worker)
        return workers

    def _create_dataloader(collate_fn, batch_size, ds, shuffle, rank, world_size):
        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        return dataloader

    def train(self):
        training.cleanup_before_training()
        self._optimizer.zero_grad()
        grad_norm = None
        self._profiler.start()

        rollout_future = None  # Will hold the async rollout result
        next_batch = None  # Will hold the next batch retrieved in advance

        # Preload the very first batch so we can launch its rollout as soon as possible.
        dataloader_iter = iter(self._dataloader)
        next_batch = next(dataloader_iter, None)

        # Launch the first rollout
        tokens = next_batch["tokens"].to(self._device)
        answers = next_batch["answers"]
        rollout_future = self.vllm_model.generate.remote(
            args=(self.generate_trajectory_batched, tokens, answers)
        )

        while next_batch is not None:
            # Wait for the current rollout to finish, then do training steps
            trajectory, context_len = rollout_future.result()
            rollout_future = None  # Clear out so we can queue the next one

            # Fetch the next batch so we can queue its rollout after we finish this one
            next_batch = next(dataloader_iter, None)
            if next_batch is None:
                break

            tokens = next_batch["tokens"].to(self._device)
            answers = next_batch["answers"]
            rollout_future = executor.submit(
                self.generate_trajectory_batched, tokens, answers
            )

            # GRPO updates
            grpo_stats: list[GRPOStats] = []
            for _ in range(self._ppo_epochs):
                step_stats = self.grpo_step(trajectory, context_len)
                grpo_stats.append(step_stats)

                if self._clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=float(self._clip_grad_norm),
                    )

                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

            self._steps_run += 1
            if self._steps_run % self._log_every_n_steps == 0:
                extra_metrics = {
                    "lr": get_lr(self._optimizer),
                }
                if grad_norm is not None:
                    extra_metrics["grad_norm"] = grad_norm
                self.log_metrics(
                    trajectory,
                    GRPOStats(*map(torch.stack, zip(*grpo_stats))),
                    **extra_metrics,
                )

            self.cleanup_after_step(trajectory, grpo_stats)
            pbar.update(1)

            if self._steps_run == self._total_steps:
                break

        self.save_checkpoint(0)
        self._profiler.stop()
    
    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """
        Perform a single GRPO optimization step over a batch of trajectories and corresponding advantages and returns.
        """
        torch.cuda.empty_cache()

        # estimate logprobs from the policy at the current optimisation step
        pi_logits = self._model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )

        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        pi_logprobs = rlhf.batched_logits_to_logprobs(
            pi_logits,
            trajectory.query_responses[:, context_length:],
            self._temperature,
            chunk_size=1,
        )

        pi_logprobs[trajectory.response_padding_masks] = 1.0

        del pi_logits
        torch.cuda.empty_cache()

        # calculate grpo loss
        loss, policy_loss, kl_loss, ratios, clipfrac = self._loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            trajectory.ref_logprobs,
            trajectory.advantages,
            padding_masks=~trajectory.response_padding_masks,
        )

        torch.cuda.empty_cache()
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return GRPOStats(
            loss,
            policy_loss,
            kl_loss,
            ratios,
            clipfrac,
            approx_policy_kls,
        )

    def generate_trajectory_batched(
        self, input_ids: torch.Tensor, answers: List[str]
    ) -> GRPOTrajectory:
        """Generate a batch of ``GRPOTrajectory``."""
        trajectories: List[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_answers = answers[
                    batch_start : batch_start + self._forward_batch_size
                ]
                trajectories.append(
                    self.generate_trajectory(batch_input_ids, batch_answers)
                )
        return GRPOTrajectory(*map(torch.cat, zip(*trajectories)))

    def generate_trajectory(
        self, input_ids: torch.Tensor, answers: List[str]
    ) -> GRPOTrajectory:
        """Generate a ``GRPOTrajectory``."""
        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)  # [B, G, L]
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        # step 1: generate responses, and logits corresponding to the responses using the current policy
        sampling_params = SamplingParams(temperature=self._temperature)
        query_responses = ray.get(
            self.vllm_model.generate.remote(prompts, sampling_params)
        )

        # training._distributed.recursive_reshard(self._model)
        # torch.cuda.empty_cache()

        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy
        logits = self._model(query_responses, input_pos=position_ids, mask=masks)
        logits = logits[:, context_length - 1 :]
        logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
        del logits
        torch.cuda.empty_cache()

        # step 2.1 estimate logprobs of the responses using the reference policy
        ref_logits = self._ref_model(
            query_responses, input_pos=position_ids, mask=masks
        )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
        ref_logprobs = rlhf.batched_logits_to_logprobs(
            ref_logits, responses, self._temperature
        )
        del ref_logits
        torch.cuda.empty_cache()

        # step 4. replace any tokens in the responses after the first stop token (usually EOS token) with padding
        # resulting in truncated responses
        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(  # [B x G, L]
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # Do some reward modelingggggggg
        # responses :: [B x G, L]
        responses = responses.reshape(batch_size, grpo_size, -1)  # [B, G, L]
        rewards, successes = batched_rewards(self._tokenizer, responses, answers)
        rewards = rewards.to(self._device)  # [B, G]
        successes = successes.to(self._device)  # [B, G]

        advantages = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.reshape(batch_size * grpo_size)  # flatten
        del responses
        torch.cuda.empty_cache()

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        return GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards.reshape(batch_size * grpo_size),
            successes=successes.reshape(batch_size * grpo_size),
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
        )

    def _sync_weights():
        handles = []
        for worker in fsdp_workers:
            handle = worker.all_gather.remote()
            handles.append(handle)

        [ray.get(handle) for handle in handles]
        print("done all gather")

        # get metadata for broadcast
        metadata = ray.get(chosen_fsdp_master_rank.get_metadata_for_broadcast.remote())

        # do broadcast key by key
        for name, (shape, dtype) in metadata.items():
            handle = llm.collective_rpc.remote(
                "update_weight", args=(name, dtype, shape)
            )
            chosen_fsdp_master_rank.broadcast_key_to_vllm.remote(name)
            ray.get(handle)

    def stop_ray(self):
        ray.shutdown()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    recipe = RayGRPORecipe(cfg=cfg)
    config.log_config(recipe_name="RayGRPORecipe", cfg=cfg)
    recipe.start_ray()
    recipe.setup()
    recipe.train()
    recipe.cleanup()
    recipe.stop_ray()


if __name__ == "__main__":
    sys.exit(recipe_main())
