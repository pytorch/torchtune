# Users
To be able to understand how we want to design the abstractions and the repo, it's important that we define how we want our users to interact with the library. One big complaint we heard from a number of users was around the black box nature of the fine tuning libraries they were using. On the other hand we also spoke to users who were very new to ml and were happy just to directly launch scripts without ever looking inside. To address this, I propose we model three levels of users and build our library to allow users to advance through these stages as they grow in requirements.

### User 1:
This user just wants to use a recipes on their own dataset. They may want to play around with a couple of parameter changes but this user would be happiest with cli level control and access to good default values for a particular recipe.

```torchrun llama2_finetune.py --dataset ./my_dataset ```

These users could use huggingface datasets we support or there own. If they used their own dataset they would be responsible for providing the dataset object with included transform.
### User 2:
This user wants to be able to fully customize the recipe but does not want to have to figure out how to build everything from scratch. This user would likely create their own repo with a torchtune dependency and then edit the recipe file themselves.

- copy llama2_finetune.py to my_finetune.py
- edit my_finetune.py directly with no need to change any library files

To enable this user it's very important that our recipes our hackable, self contained, and easily readable. For this reason we encourage copy and pasting the training loop as we add more recipes.
### User 3:
This final user has their own training setup and custom solutions and our just looking for access to some of our components such as models, recipe defaults, or specific trainer utils they don't have. For this user our recipe just acts as an example and they should be able to use our components a la carte in the same way we do in our recipes.

# Training Abstractions

I will include pseudocode here for how we can design our recipes to support the user profiles listed above.

```
parser = argparse.ArgumentParser(...) # support all getters, lr, batchsize, epoch, dtype, num_devices, seed

model = get_model(...)  # get from our models, or our PEFT models
model = tune.distributed(model, num_devices) # FSDP wrap
dataset = get_dataset(...) # one of our datasets, which point to hf datasets
dataloader = Dataloader(dataset, ..., collate_fn=tune.llm_collate())
loss = get_loss(...)

logger = get_logger(...) # wandb, tensorboard, etc
checkpoint = get_checkpoint(...) # checkpointer, sharded_checkpointer, etc
evaluator = get_eval(...) # eval harness to use

opt = get_optimizer(...)
opt = tune.wrapped_optimizer(opt, num_devices) # autocast + distributed logic
device = get_device(num_devices)
dtype = dtype

for e in epoch:
	for step, batch in enumerate(dataloader):
		opt.zero_grad():

		data = data.to(device)

		with tune.cast(dtype):  # choose the appropriate autocast if distributed
			out = model(data)
			l = loss(out, data)

		l.backward()
		opt.step()

		logger(model, l, step, epoch)
		checkpoint(model, opt, dataloader, step, epoch)
		evaluator(model, step, epoch)
		profiler(...)

```

The above example, if we agree on it, should act as a guide for the direction of our recipe code. Initially none of the getter functions and utils will be built so there will be much more boilerplate but we can reduce this with time.
- User 1: call script directly from cli and can change out any of the getter options. We will also need to provide them good defaults for specific combinations. We can do this as python files or yaml.
- User 2: copies just this file and can manually swap out any line or add custom logic directly into the loop. No need to edit our library to use it.
- User 3: They can use the getters they want in their own training code.

The above design roughly approximates a trainer but it's easily editable. It also doesn't require us to make it support every fine tuning concept. For example, the above script might be called finetune_llm.py which can be reused for a lot of recipes. But if a more exotic training setup came along we can just copy it and make a new one for different types of training without complexifying this one.

# Repo Design
Finally a note on repo design. I think all of the components should be groped together so that they serve as a library within the library for the recipes to access and for user 3 to access directly with clean imports. The components should be grouped according to the getter functions.

```
recipes/
	finetune_llm.py

defaults/
	finetune_llama2_on_alpaca.yaml
	lora_llama2_on_alpaca.yaml

tune/
    models/    # including peft wrapped models
    datasets/
    trainer/     # util folder
    ...
```
