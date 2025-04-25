class vLLMRolloutModel(Worker):
    """vLLM Rollout Model worker for Ray."""

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


@ray.remote(num_cpus=32, num_gpus=1)
class PyTorchActorModel:
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

    def estimate_logprobs(self, trajectories):
        padding_mask = trajectories != tokenizer.pad_id
        mask = get_causal_mask_from_padding_mask(padding_mask)
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        logits = self.model(trajectory, input_pos=position_ids, mask=mask)
        return rlhf.batched_logits_to_logprobs(logits, trajectories, temperature)

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
    def start_ray():
        ray.init(num_cpus=192, num_gpus=6)

    def setup():
        self.rollout = _create_vllm_worker(vLLMRolloutModel, tensor_parallel_size=1)
        self.reference = _create_vllm_worker(vLLMReferenceModel, tensor_parallel_size=1)
        self.actor_workers = [worker(env) for i in range(4)]

    def _create_vllm_worker(worker_cls: Worker, tensor_parallel_size: int):
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
            worker_cls=worker_cls,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
        )
        return llm

    def train():
        dataloader = iter(self._dataloader)
        next_batch = next(dataloader, None)

        tokens = next_batch["tokens"]
        answers = next_batch["answers"]
        sampling_params = SamplingParams(temperature=0.0)
        rollout_future = self.vllm_model.generate.remote(tokens, sampling_params)

        while next_batch is not None:
            # Wait for the current rollout to finish, then do training steps
            trajectories = ray.get(rollout_future)
            rewards, advantages = _compute_rewards_and_advantages(trajectories, answers)

            # Fetch the next batch so we can queue its rollout after we finish this one
            next_batch = next(dataloader_iter, None)
            if next_batch is None:
                break

            # Launch the next rollout
            tokens = next_batch["tokens"]
            answers = next_batch["answers"]
            rollout_future = self.vllm_model.generate.remote(tokens, sampling_params)

            # GRPO updates
            for worker in self.actor_workers:
                logprobs = ray.get(
                    worker.estimate_logprobs(trajectory, input_pos, masks).remote()
                )
                ref_logprobs = ray.get(
                    self.reference.estimate_logprobs(
                        trajectory, input_pos, masks
                    ).remote()
                )
                loss = worker.compute_loss_and_take_a_step(
                    logprobs, ref_logprobs, trajectory
                ).remote()
                worker._optimizer.step()
                worker._optimizer.zero_grad(set_to_none=True)

            # Sync weights
            sync_weights()

        save_checkpoint()

    def stop_ray():
        pass
