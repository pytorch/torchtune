import runpy
import sys

TUNE_PATH = "/data/users/ebs/ebs-torchtune/torchtune/_cli/tune.py"

async_cmd_1 = [
    "tune",
    "run",
    "--nnodes",
    "1",
    "--nproc_per_node",
    "2",
    "full_finetune_distributed",
    "--config",
    "llama3/8B_full",
    "batch_size=1",
    "gradient_accumulation_steps=4",
    "output_dir=/tmp/pytest-of-ebs/pytest-20/test_training_state_on_resume_0",
    "checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer",
    "checkpointer.checkpoint_dir='/tmp/test-artifacts'",
    "checkpointer.checkpoint_files=[/tmp/test-artifacts/small-ckpt-tune-llama3-05052024.pt]",
    "checkpointer.output_dir=/tmp/pytest-of-ebs/pytest-20/test_training_state_on_resume_0",
    "checkpointer.model_type=LLAMA3",
    "tokenizer.path='/tmp/test-artifacts/tokenizer_llama3.model'",
    "tokenizer.prompt_template=null",
    "metric_logger.filename=/tmp/pytest-of-ebs/pytest-20/test_training_state_on_resume_0tmppytest-of-ebspytest-20test_training_state_on_resume_0.txt",
    "enable_async_checkpointing=False",
    "dtype=fp32",
    "enable_activation_checkpointing=False",
    "enable_activation_offloading=False",
    "dataset.train_on_input=False",
    "seed=9",
    "epochs=2",
    "max_steps_per_epoch=2",
    "optimizer=torch.optim.AdamW",
    "optimizer.lr=2e-5",
    "log_every_n_steps=1",
    "dataset._component_=torchtune.datasets.alpaca_dataset",
    "dataset.source='json'",
    "dataset.data_files=/data/users/ebs/ebs-torchtune/tests/assets/alpaca_tiny.json",
    "dataset.split='train'",
    "model._component_=torchtune.models.llama3.llama3",
    "model.vocab_size=128_256",
    "model.num_layers=2",
    "model.num_heads=8",
    "model.embed_dim=64",
    "model.max_seq_len=1024",
    "model.norm_eps=1e-5",
    "model.num_kv_heads=4",
    "clip_grad_norm=100",
    "optimizer_in_bwd=False",
]

async_cmd_2 = [
    "tune",
    "run",
    "--nnodes",
    "1",
    "--nproc_per_node",
    "2",
    "full_finetune_distributed",
    "--config",
    "llama3/8B_full",
    "batch_size=1",
    "gradient_accumulation_steps=4",
    "output_dir=/tmp/pytest-of-ebs/pytest-19/test_training_state_on_resume_0",
    "checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer",
    "checkpointer.checkpoint_dir='/tmp/test-artifacts'",
    "checkpointer.checkpoint_files=[/tmp/test-artifacts/small-ckpt-tune-llama3-05052024.pt]",
    "checkpointer.output_dir=/tmp/pytest-of-ebs/pytest-19/test_training_state_on_resume_0",
    "checkpointer.model_type=LLAMA3",
    "tokenizer.path='/tmp/test-artifacts/tokenizer_llama3.model'",
    "tokenizer.prompt_template=null",
    "metric_logger.filename=/tmp/pytest-of-ebs/pytest-19/test_training_state_on_resume_0/resumedtmppytest-of-ebspytest-19test_training_state_on_resume_0resumed.txt",
    "resume_from_checkpoint=True",
    "enable_async_checkpointing=True",
    "dtype=fp32",
    "enable_activation_checkpointing=False",
    "enable_activation_offloading=False",
    "dataset.train_on_input=False",
    "seed=9",
    "epochs=3",
    "max_steps_per_epoch=2",
    "optimizer=torch.optim.AdamW",
    "optimizer.lr=2e-5",
    "log_every_n_steps=1",
    "dataset._component_=torchtune.datasets.alpaca_dataset",
    "dataset.source='json'",
    "dataset.data_files=/data/users/ebs/ebs-torchtune/tests/assets/alpaca_tiny.json",
    "dataset.split='train'",
    "model._component_=torchtune.models.llama3.llama3",
    "model.vocab_size=128_256",
    "model.num_layers=2",
    "model.num_heads=8",
    "model.embed_dim=64",
    "model.max_seq_len=1024",
    "model.norm_eps=1e-5",
    "model.num_kv_heads=4",
    "clip_grad_norm=100",
    "optimizer_in_bwd=False",
    ]

def main():
    
    sys.argv = cmd_1
    runpy.run_path(TUNE_PATH, run_name="__main__")
    print("CMD1 DONE")
    cmd_2 = 
    sys.argv = cmd_2
    runpy.run_path(TUNE_PATH, run_name="__main__")
    print("DONE")


if __name__ == "__main__":
    main()
