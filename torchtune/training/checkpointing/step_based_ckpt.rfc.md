## Enabling step-based checkpointing in torchtune

#### What are we currently doing?

We currently only checkpoint at epoch boundaries. That means a fine-tuning run has to iterate through **all data** in a dataset before saving a checkpoint. That's a problem when GPUs (especially interconnected GPUs) can fail frequently, losses can diverge, and datasets keep getting larger and larger.

We provide a tiny amount of flexibility by allowing the user to specify `max_steps_per_epoch`, so they can short-circuit the epoch and save sooner. In addition, it's always possible to split a dataset into chunks and train over them independently, resuming from training to simulate a larger training run.

Both of these "hacks" are not ideal and we've had users continually asking if they can control checkpointing based on number of training steps. (#988, #1107)

#### What does step-based checkpointing look like for the user?

I think the best way to do this would to show an example. Let's take our [Llama3 8B single device full fine-tuning recipe](./recipes/configs/llama3/8B_full_finetune_single_device.yaml), which utilizes the Alpaca dataset. The Alpaca dataset has ~52k samples. Using a batch size of 2 and a gradient accumulation of 16 steps, we can estimate around 1625 steps in this training run. Let's save a checkpointing every 500 steps!

From the config, we can specify:
```yaml
save_every_n_train_steps: 500
```

And in our output directory, we can expect to see something like this:
```
output_dir/
	step_500/
		llama3_8b_single_device.yaml
		config.json
		model-0000-of-0003.safetensors
		...
	step_1000/
		...
	step_1500/
		...
	step_1625/
		...
```

> You'll see that at the end of the training loop, a final checkpoint is saved regardless of how many steps have passed since the last checkpoint was saved.

At this point you might be saying: @joecummings, do you think memory grows on trees? Do you think we all drive Bugattis and smash up Grace Hopper machines for fun? Each Llama3 8B model is roughly 16 GB of memory and we've saved 4 copies of that in addition to the base model we used. That's 80 GB just for checkpoints! Not even to mention if we wanted to save the optimizer states, too...

Introducing:

```yaml
keep_last_n_checkpoints: 1
```

This param will prune all the checkpoints except for the last N specified, leaving you with just the checkpoints you're interested in:
```
output_dir/
	step_1625/
		llama3_8b_single_device.yaml
		config.json
		model-0000-of-0003.safetensors
		...
```

**What about the concept of epochs?**

The concept of epochs will stay as a way to control how long training runs, as will the possibility to shorten training using `max_steps_per_epoch`; however, checkpointing will be entirely handled by a specification of steps.

**Will this slow down training?**

Great question! Checkpointing can take a long time, especially if saving the optimizer state for resuming training at a later date. For single device recipes, this likely isn't a huge issue, but for distributed recipes where the state dict needs to be collected on rank zero before saving, this can be verrrrrrry slow so anything that increases the frequency of checkpointing will increase the time it takes for training to complete. There are two ways to mitigate this:

1) Specify a longer period so you save checkpoints less frequently
2) Look into using DCP checkpointer for any intermediate checkpoints, which drastically reduces the time it takes to save. See #2006 for more information on this.

#### What changes need to be made in code?

**In the recipe:**

```python
num_steps = 0
for curr_epoch in range(self.epochs_run, self.total_epochs):
	steps_run_since_last_saved_ckpt = 0
	for idx, batch in enumerate(self._dataloader):
		# Model forward
		...
		if doing_bwd_pass:
			steps_run_since_last_saved_ckpt += 1
			num_steps += 1

		if steps_run_since_last_saved_ckpt == save_ckpt_every_n_steps:
			self.save_checkpoint(ckpt, step=num_steps)
			steps_run_since_last_saved_ckpt = 0

# One final save
self.save_checkpoint(ckpt, step=num_steps)
```

**And in the checkpointer:**

```python
def save_checkpoint(ckpt, step):
	# Prune old checkpoints if needed
	checkpoints_already_saved = get_all_prev_checkpoint_paths(self.output_dir)
	if len(checkpoints_already_saved) >= self.keep_last_n_checkpoints:
		prune_old_checkpoints(checkpoints_already_saved)

	# Create new folder for path
	new_ckpt_path = output_dir / step_{step}
	new_ckpt_path.mkdir(exist_ok=False)
	# Save new checkpoint
	torch.save(ckpt, new_ckpt_path / "model.bin")
```

#### Inspiration from relevant repositories:**
* [TorchTitan](https://github.com/pytorch/torchtitan)
* [TorchTNT](https://github.com/pytorch/tnt)
