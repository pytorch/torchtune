import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchtune.llm.llama2.tokenizer import Tokenizer
from torchtune.llm.llama2.transformer import TransformerDecoder
from torchtune.llm.generation import GenerationUtils

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a native PyTorch LLaMA model on a HuggingFace dataset.')
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, help='HuggingFace dataset name.')
    # Model arguments
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to SentencePiece tokenizer.')
    parser.add_argument('--model-checkpoint', type=str, required=True, help='Path to native PyTorch LLaMA model checkpoint.')
    # Fine-tuning arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for fine-tuning.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for fine-tuning.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for fine-tuning')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW'], help='Optimizer to use for fine-tuning.')
    parser.add_argument('--loss-fn', type=str, default='cross_entropy', choices=['cross_entropy'], help='Loss function to use for fine-tuning')
    parser.add_argument('--output-dir', type=str, default='/tmp/llama-finetune', help='Directory in which to save checkpoints during fine-tuning.')
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokenizer.pad_id = 0 # Original tokenizer has no pad_id, which causes indexing errors when batch training
    print("Tokenizer initialized.")

    # Initialize model
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        max_seq_len=2048,
        norm_eps=1e-5,
    )
    state_dict = torch.load(args.model_checkpoint)
    model.load_state_dict(state_dict)
    print("Model initialized.")

    # Load dataset
    dataset = load_dataset(args.dataset, split="train")
    print("Dataset loaded.")

    # Initialize optimizer
    opt = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # Initialize loss
    loss_fn = getattr(torch.nn.functional, args.loss_fn)

    for epoch in range(args.epochs):
        for i, batch in enumerate(DataLoader(dataset, args.batch_size)):
            opt.zero_grad()

            # ---- Prepare inputs ---- #
            # Grab prompt + instruction + input
            response_tag = "\n\n### Response:\n"
            text = batch["text"]
            instructions_and_inputs = [t.split(response_tag)[0] + response_tag for t in text]

            # Tensorize and encode
            input_ids = tokenizer.to_tensor(
                [tokenizer.encode(t) for t in instructions_and_inputs],
                max_length=model.max_seq_len,
                pad_value=tokenizer.pad_id
            )
            input_ids_seq_len = input_ids.shape[-1]
            labels = tokenizer.to_tensor(
                [tokenizer.encode(output) for output in batch["output"]],
                max_length=model.max_seq_len,
                pad_value=-100
            )
            labels_seq_len = labels.shape[-1]

            # Hack to pad correctly and not use max_seq_len, which is costly
            if input_ids_seq_len > labels_seq_len:
                labels = F.pad(labels, (0, input_ids_seq_len - labels_seq_len), value=-100)
            elif labels_seq_len > input_ids_seq_len:
                input_ids = F.pad(input_ids, (0, labels_seq_len - input_ids_seq_len), value=tokenizer.pad_id)

            # ---- Run forward pass ---- #
            logits = model(input_ids)

            # ---- Compute loss ---- #
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
            shift_labels = shift_labels.view(-1)

            loss = loss_fn(shift_logits, shift_labels)
            print(f"Loss @ step {i} in epoch {epoch}: {loss}")

            loss.backward()
            opt.step()

        # Save checkpoint at end of each epoch (to be changed later)
        torch.save(model.state_dict(), f"{args.output_dir}/model_{epoch}.ckpt")

if __name__ == '__main__':
    main()
