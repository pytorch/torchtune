====================================
Quick Start: Using Your Own Data
====================================

This guide answers the #1 question for TorchTune users: **"How do I use my own data?"**

.. contents:: Table of Contents
   :local:
   :depth: 2

.. note::
   **5-Minute Quick Start**: If you just want to get running immediately, jump to :ref:`quickstart-examples`.

Why This Guide Exists
---------------------
Based on user feedback, finding how to use custom data requires searching through multiple
documentation pages. This guide consolidates everything you need in one place.

.. _quickstart-examples:

Quick Start Examples
--------------------

Example 1: Simple Q&A Dataset (JSON)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Create your data file** (``my_data.json``)::

    [
        {
            "conversations": [
                {"from": "human", "value": "What is machine learning?"},
                {"from": "gpt", "value": "Machine learning is a type of AI that enables computers to learn from data."}
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Explain neural networks"},
                {"from": "gpt", "value": "Neural networks are computing systems inspired by biological neural networks."}
            ]
        }
    ]

**Step 2: Create your config** (``my_config.yaml``)::

    # Dataset configuration
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: ./my_data.json
      template: torchtune.data.AlpacaInstructTemplate
      train_on_input: False

    # Model configuration (using Llama 3.1 8B as example)
    model:
      _component_: torchtune.models.llama3_1.llama3_1_8b

    # Training configuration
    batch_size: 2
    epochs: 3
    max_steps_per_epoch: null
    gradient_accumulation_steps: 8
    learning_rate: 2e-5

    # LoRA configuration
    lora_rank: 64
    lora_alpha: 128
    lora_dropout: 0.1

    # Output
    output_dir: ./my_finetuned_model

    # Checkpointing
    checkpoint_dir: ./checkpoints

**Step 3: Run fine-tuning**::

    tune run lora_finetune_single_device --config my_config.yaml

Example 2: Instruction Following Dataset (CSV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Prepare your CSV** (``instructions.csv``)::

    instruction,response
    "Write a poem about AI","Artificial minds awaken, silicon dreams take flight..."
    "Explain quantum computing","Quantum computing harnesses quantum mechanics principles..."

**Step 2: Convert to TorchTune format**::

    import pandas as pd
    import json

    # Read CSV
    df = pd.read_csv('instructions.csv')

    # Convert to TorchTune format
    data = []
    for _, row in df.iterrows():
        data.append({
            "conversations": [
                {"from": "human", "value": row['instruction']},
                {"from": "gpt", "value": row['response']}
            ]
        })

    # Save as JSON
    with open('instructions.json', 'w') as f:
        json.dump(data, f, indent=2)

**Step 3: Use the same config as Example 1**, just change the source::

    dataset:
      source: ./instructions.json

Example 3: Using HuggingFace Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option A: Public HuggingFace dataset**::

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: "teknium/OpenHermes-2.5"  # HuggingFace dataset ID
      column_map: {"input": "instruction", "output": "response"}

**Option B: Your private HuggingFace dataset**::

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: "your-username/your-dataset"
      split: "train"
      use_auth_token: True  # Will use your HF token

Common Data Formats
-------------------

Chat/Conversation Format (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Best for: Chatbots, assistants, multi-turn conversations

.. code-block:: json

    {
        "conversations": [
            {"from": "human", "value": "User message"},
            {"from": "gpt", "value": "Assistant response"},
            {"from": "human", "value": "Follow-up question"},
            {"from": "gpt", "value": "Follow-up response"}
        ]
    }

Instruction-Response Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Best for: Single-turn tasks, completion

.. code-block:: json

    {
        "instruction": "Summarize this text: [long text here]",
        "response": "Summary: [concise summary]"
    }

Text Completion Format
~~~~~~~~~~~~~~~~~~~~~~
Best for: General text generation, continuing stories

.. code-block:: json

    {
        "text": "Once upon a time in a land far away..."
    }

Step-by-Step Setup Guide
------------------------

Step 1: Prepare Your Data
~~~~~~~~~~~~~~~~~~~~~~~~~

**Data Requirements:**

- **Format**: JSON file with proper structure
- **Size**: Minimum ~100 examples for meaningful fine-tuning
- **Quality**: Clean, diverse examples without excessive repetition
- **Balance**: Mix different types of prompts/responses

**Data Validation Script**::

    import json

    # Load and validate your data
    with open('my_data.json', 'r') as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # Check format
    for i, item in enumerate(data[:5]):
        print(f"\nExample {i+1}:")
        if 'conversations' in item:
            for conv in item['conversations']:
                print(f"  {conv['from']}: {conv['value'][:50]}...")
        else:
            print("  Warning: Missing 'conversations' key!")

Step 2: Choose Your Model and Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Models Available:**

- ``llama3_1_8b`` - Best for general use
- ``llama3_1_70b`` - For complex tasks (requires more memory)
- ``mistral_7b`` - Good balance of size and performance
- ``phi3_mini`` - For resource-constrained environments

**Fine-tuning Methods:**

- **LoRA** (Recommended): Efficient, requires less memory
- **QLoRA**: Even more memory efficient, slightly slower
- **Full Fine-tuning**: Best quality, requires significant resources

Step 3: Create Your Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with a template::

    # Copy a base config
    cp recipes/configs/llama3_1/8B_lora_single_device.yaml my_config.yaml

Key settings to modify::

    # Your data
    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: ./my_data.json

    # Training hyperparameters
    batch_size: 2  # Reduce if OOM
    learning_rate: 2e-5  # Lower for more stable training
    epochs: 3  # Increase for better results

    # LoRA parameters
    lora_rank: 64  # Higher = more capacity but more memory
    lora_alpha: 128  # Usually 2x lora_rank

Step 4: Test with Small Data First
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a test subset::

    import json

    # Load full data
    with open('my_data.json', 'r') as f:
        data = json.load(f)

    # Save first 10 examples
    with open('test_data.json', 'w') as f:
        json.dump(data[:10], f, indent=2)

Test run::

    tune run lora_finetune_single_device --config my_config.yaml \
      dataset.source=./test_data.json \
      epochs=1 \
      max_steps_per_epoch=5

Step 5: Run Full Fine-tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    tune run lora_finetune_single_device --config my_config.yaml

Monitor training::

    # In another terminal
    tail -f torchtune.log

Troubleshooting Common Issues
-----------------------------

"FileNotFoundError: Dataset not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Can't find your data file

**Solutions**:

1. Use absolute path::

    dataset:
      source: /home/user/data/my_data.json

2. Check working directory::

    import os
    print(f"Current directory: {os.getcwd()}")

"KeyError: 'conversations'"
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Data format mismatch

**Solution**: Ensure your data matches expected format::

    # Check your data structure
    import json
    with open('my_data.json', 'r') as f:
        data = json.load(f)
        print(f"Keys in first item: {data[0].keys()}")

"CUDA out of memory"
~~~~~~~~~~~~~~~~~~~~

**Problem**: GPU memory exceeded

**Solutions**:

1. Reduce batch size::

    batch_size: 1  # Start with 1

2. Use gradient accumulation::

    gradient_accumulation_steps: 16  # Increase this

3. Enable gradient checkpointing::

    enable_gradient_checkpointing: True

4. Use QLoRA instead::

    tune run qlora_finetune_single_device --config my_config.yaml

"Loss is NaN or not decreasing"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Training instability

**Solutions**:

1. Lower learning rate::

    learning_rate: 5e-6  # or even 1e-6

2. Check data quality::

    # Look for empty responses, special characters, etc.
    for item in data:
        for conv in item['conversations']:
            if not conv['value'].strip():
                print("Found empty response!")

Advanced Topics
---------------

Using Custom Templates
~~~~~~~~~~~~~~~~~~~~~~

Create custom formatting::

    from torchtune.data import InstructTemplate

    class MyCustomTemplate(InstructTemplate):
        template = "### Human: {instruction}\n### Assistant: {response}"

    # Use in config
    dataset:
      template: my_module.MyCustomTemplate

Multi-Dataset Training
~~~~~~~~~~~~~~~~~~~~~~

Combine multiple sources::

    from torchtune.datasets import ConcatDataset

    dataset:
      _component_: torchtune.datasets.ConcatDataset
      datasets:
        - _component_: torchtune.datasets.instruct_dataset
          source: ./technical_data.json
        - _component_: torchtune.datasets.instruct_dataset
          source: ./creative_data.json

Filtering and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add data filters::

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: ./my_data.json
      filter_fn: |
        lambda x: len(x['conversations'][0]['value']) > 10 and \
                  len(x['conversations'][1]['value']) > 20

Memory-Efficient Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets::

    dataset:
      _component_: torchtune.datasets.instruct_dataset
      source: ./huge_dataset.json
      streaming: True  # Load data on-the-fly
      buffer_size: 1000  # Number of examples to buffer

Monitoring and Evaluation
-------------------------

Track Training Progress
~~~~~~~~~~~~~~~~~~~~~~~

Use TensorBoard::

    # In config
    metric_logger:
      _component_: torchtune.utils.metric_logging.TensorBoardLogger
      log_dir: ./logs

    # View logs
    tensorboard --logdir ./logs

Evaluate Your Model
~~~~~~~~~~~~~~~~~~~

After training::

    # Generate text
    tune run generate --config my_config.yaml \
      prompt="What is machine learning?"

Next Steps
----------

- :doc:`instruct_datasets` - Deep dive into dataset formats
- :doc:`../tutorials/first_finetune_tutorial` - Complete fine-tuning walkthrough
- :doc:`../tutorials/llama3` - Model-specific considerations
- :doc:`../tutorials/evaluation` - Evaluate your fine-tuned model

.. note::
   **Need help?** Join our `Discord <https://discord.gg/torchtune>`_ for community support!
