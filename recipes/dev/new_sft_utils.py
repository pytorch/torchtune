def tp_parallelize_module(model, device_mesh, plan):
    """
    """
    model = training.prepare_mha_for_tp(model, device_mesh["tp"])
    parallelize_module(
        model,
        device_mesh["tp"],
        parallelize_plan=tensor_parallel_plan,
    )


def data_parallelize_module(model, custom_sharded_layers, fsdp_cpu_offload, reshard_after_forward, device_mesh):
    """
    """
    fsdp_shard_conditions = [
        partial(
            training.get_shard_conditions,
            names_to_match=custom_sharded_layers,
        )
    ]
    training.shard_model(
        model=model,
        shard_conditions=fsdp_shard_conditions,
        cpu_offload=fsdp_cpu_offload,
        reshard_after_forward=reshard_after_forward,
        dp_mesh=device_mesh["dp"],
    )


def init_buffers(model, dtype, device):
    """
    """
    with training.set_default_dtype(dtype), device:
        for m in model.modules():
            # Some buffers (e.g. Rope) is not covered in the state dict
            if hasattr(m, "buffer_init"):
                m.buffer_init()


def set_activation_checkpointing(model, auto_wrap_policy, mode=None, option=None)
    if mode is None:
        # original set_activation_checkpointing code
    else:
        apply_selective_activation_checkpointing(model, mode, option)



class Profiler:

    def __init__(self, ...):
        # config.instantiate(cfg_profiler) builder will return this
        # inlcude profiler_cfg values here
        # remove DummyProfiler
        self.torch_profiler = torch_profiler # either torch.profiler.profile or None
        self.start_time = time.perf_counter()
        self.step = step
        # set to -1 when not enabled
        self.cuda_memory_step = self.profiler_wait_step + self.profiler_warmup_steps
        # Start recording on init to capture first step
        if self.step == self.cuda_memory_step:
            self._record_memory_history()

    def _record_memory_history(self):
        if self.rank==0 and self.device.type=="cuda":
            torch.cuda.memory._record_memory_history()

    def _end_record_memory_history(self):
        if self.rank==0 and self.device.type=="cuda":
            torch.cuda.memory._record_memory_history(enabled=None)

    def step(self):
        if torch_profiler is not None:
            self.torch_profiler.step()
        self.start_time = time.perf_counter()
        self.step += 1
        if self.step == self.cuda_memory_step:
            torch.cuda.memory._record_memory_history()
        elif self.step == self.cuda_memory_step + self.profiler_active_steps:
            torch.cuda.memory._record_memory_history(enabled=None)

    @property
    def step_time(self):
        return time.perf_counter() - self.start_time

    def __enter__(self):
        if torch_profiler is not None:
            self.torch_profiler.start()
        return self

    def __exit__(self, type, value, traceback):
        if torch_profiler is not None:
            self.torch_profiler.stop()



class Checkpointer: # formerly CheckpointClient
    ...

    def update_state(self, state):
        # [todo] update this for step only
        # [todo] update step (global_step)
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

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

######################## grad accum ##############

# OPTION 1
class GradientAccumulation:
    def __init__(self, model, steps, dp_size=1):
        self.model = model
        self.steps = steps
        self.dp_size = dp_size
        self.num_tokens = 0
        self.running_loss = 0
        self._count = 0

    def scale_loss(self, loss, num_tokens):
        scaled_loss = loss * num_tokens
        self.num_tokens += num_tokens
        self.running_loss += scale_loss
        self._count += 1
        return scaled_loss

    @property
    def avg_loss(self):
        if self._count > -1:
            raise ValueError(f"avg_loss was accessed before gradient accumulation steps were completed")
        return self.running_loss.item() / self.num_tokens

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._count < self.steps:
            raise ValueError(f"Steps set to {self.steps}, but scale_loss was only called {self._count} times.")
        self._count = -1
        if self.dp_size > 1:
            torch.distributed.all_reduce(self.num_tokens)
            torch.distributed.all_reduce(self.running_loss)
        training.scale_grads(self.model, self.dp_size / self.num_tokens)


with GradientAccumulation(grad_steps, model, dp_size) as grad_acc:
    for _ in range(grad_acc.steps):
        batch = next(dataloader)
        labels = batch.pop("labels")
        num_tokens = (labels != loss.ignore_index).sum()

        with self.activation_offloading:
            logits = state.model(**batch)

        loss = state.loss(logits, labels)
        grad_acc.scale_loss(loss, num_tokens)

        loss.backward()



# Option 2
@dataclass
class AccumulationStats:
    num_token: int = 0
    running_loss: int = 0

    def scale_loss(self, loss, num_tokens):
        scaled_loss = loss * num_tokens
        self.num_tokens += num_tokens
        self.running_loss += scale_loss
        return scaled_loss


def get_gradient_accumulator(steps, model, dp_size=1):
    def grad_accumulator():
        acc_step = AccumulationStats()
        for i in range(steps):
            yield acc_step
        if dp_size > 1:
            torch.distributed.all_reduce(stats.num_tokens)
            torch.distributed.all_reduce(stats.running_loss)
        training.scale_grads(model, dp_size / stats.num_tokens)
    return grad_accumulator


for acc_step in grad_accumulator(grad_steps, model, dp_size):
    batch = next(dataloader)
    labels = batch.pop("labels")
    acc_step.token_count((labels != loss.ignore_index).sum())

    with self.activation_offloading:
        logits = state.model(**batch)

    loss = state.loss(logits, labels)
    acc_step.scale_loss(loss)

    loss.backward()
