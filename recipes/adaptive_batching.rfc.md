<h2>Enable adaptive batching in torchtune</h2>

**Intuition**

It is useful to set a maximum batch size that is not causing OOM for a given compute. Also, it might be interesting to increase batch size gradually during the training process. Both techniques are examples of **adaptive batching**.

**What are we currently doing?**

We don't have an approach to adaptive batching in both offline (set it before training) and online (update it during the training) paradigms.

**How will it look for the user?**

I think the best design at this point is the addition of a new parameter in the recipe. For an offline approach, we might just add a field like this:

`use_maximum_batch_size: True`

It will be exposed with the default parameter set to `False` in every config.

In the case of an online approach, we want a stronger possibility. As the standard **non-empirical** method for adaptive batching does not exist, we might provide the possibility to define the way in which batch size will be increased on users' hands in the following way. Let's add a non-required parameter in the recipe:

`batch_increase_function: ...`

Where a user will provide a function with conditions on increasing batch size and value on which batch size will be increased. Also, we do an offline procedure before training to understand the maximum bound for batch size to handle the case that users' increasing function will cause OOM. By definition, `batch_increase_function` will accept 2 arguments: `epoch` and `current_batch_size`, and it will return a number on what we increase the batch size. An example:

```
def increasing_function(epoch: int, current_batch_size: int) -> int:
    if epoch in [...]:
        return ...
     else:
        return 0
```

In the recipe, on each epoch, the following check will be done:

```

increase_value = batch_increase_function(epoch, batch_size)

if increase_value and  increase_value + batch_size <= max_batch_size:
    ...
```

<h3>On some ways of adapting batching</h3>

**Online**

Currently, only emperical methods has shown efficiency in real tasks. Speaking about "clever" and non-emperical ways, there were no real works that showed greater perfomance then emperical increasing.
Basically, non-emperical ways will use some quailities of optimizer (for example stochastic approximation of upper bound on Breggmans' distance) and addition of such will require thoughtfull consideration.

**Offline**

Main approach is based on idea "Decrease until OOM". The general pipeline looks like this:

```
batch_size = len(batch)
random_state = zero.random.get_state()
loss = None
while chunk_size != 0:
    try:
        zero.random.set_state(random_state)
        optimizer.zero_grad()
        if batch_size <= chunk_size:
            loss = loss_fn(*step(batch))
            loss.backward()
        else:
            loss = None
            for chunk in zero.iter_batches(batch, chunk_size):
                chunk_loss = loss_fn(*step(chunk))
                chunk_loss = chunk_loss * (len(chunk) / batch_size)
                chunk_loss.backward()
                if loss is None:
                    loss = chunk_loss.detach()
                else:
                    loss += chunk_loss.detach()
    except RuntimeError as err:
        if not is_oom_exception(err):
            raise
        chunk_size //= 2
    else:
        break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
```

This is fine approach but not really optimal, better way is to try binary search on answer:

```
def is_oom(batch_size):
     ... # Can be done in different ways

minimum_batch = 0
maximum_batch = 100000
best_batch_size = 0

while minimum_batch <= maximum_batch:
    mid_batch = (minimum_batch + maximum_batch) // 2
    
    if is_oom(mid_batch):
        maximum_batch = mid_batch - 1
    else:
        # If we are not out of memory, record this as the best batch size
        best_batch_size = mid_batch
        minimum_batch = mid_batch + 1
```

Attempt to predict it without OOMs. The most interesting approach is OOM-less attempt to predict it. Probably, we can do some rough rate in following way:

```
def memory_usage(batch_size: int) -> tuple[int]:
    ... # do procedure for given batch_size

    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    allocated_memory = allocated_memory / (1024 ** 2)
    reserved_memory = reserved_memory / (1024 ** 2)

    torch.cuda.empty_cache()
    return allocated_memory, reserved_memory


# In disributed case, for device in devices:
try:
    used_memory = memory_usage(2) - memory_usage(1) # it scales lineary and to handle extra constant consider such difference.
except RuntimeError as err:
    if is_oom_exception(err):
        raise RuntimeError('Not enough memory even for batch_size=1 and batch_size-2')

# Then just divide total_memory on used_memory for each 1 in batch_size

total_memory = torch.cuda.get_device_properties(device).total_memory

final_batch = total_memory // used_memory
```

There other ways to get this rate either, like:

`(vram - model_size) / (forward_back_ward_size)`

For all ways we round to the closest lower power of two.

Obviously, required additions in docs will be done,



