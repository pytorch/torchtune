# Distributed torchtune



To make a recipe distributed:
1. `_, rank = training.get_world_size_and_rank(); self._is_rank_zero = rank == 0`
2. Metric logger only in rank 0
3. More options in `_setup_model`
4. Compile loss - only verbose in rank 0
5. `utils.log_rank_zero` instead of `log.info`
6. `_setup_optimizer` has some changes, mostly using `training.load_from_full_optimizer_state_dict`
7. `_setup_data` has `training.get_world_size_and_rank` for a proper distributed sampler
8. No `_loss_step` in distributed? it seems to be inlined, probably unimportant
9. In train:
   1. only track memory on rank 0
   2. 