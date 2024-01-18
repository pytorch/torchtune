# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# python -m recipes.evaluate --native-checkpoint-path /tmp/finetune-llm/model_2.ckpt --tokenizer-path ~/llama/tokenizer.model

import argparse
import logging
import time
from typing import List, Optional

import lm_eval

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn.functional as F
from accelerate import find_executable_batch_size
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from torchtune.models.llama2 import llama2_7b, llama2_tokenizer
from torchtune.modules.tokenizer import Tokenizer
from torchtune.modules.transformer import TransformerDecoder
from torchtune.utils.device import _get_device_from_env
from torchtune.utils.env import seed
from torchtune.utils.generation import GenerationUtils
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvalWrapper(LM):
    """
    A wrapper class for TransformerDecoder, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: Tokenizer,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cuda")
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = 512

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        encoded = self._tokenizer.encode(string, add_eos=False)
        return encoded

    def _detect_batch_size(self, requests=None, pos=0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
        else:
            max_length = self.max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            test_batch = torch.ones((batch_size, max_length), device=self.device).long()
            for _ in range(5):
                _ = F.log_softmax(self._model_call(test_batch), dim=-1).cpu()
            return batch_size

        batch_size = forward_batch()
        utils.clear_torch_cache()

        return batch_size

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation

        # automatic batch size detection for vectorization
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # reorder requests by length of context
        re_ord = utils.Reorderer(requests, _collate)

        reordered_requests = re_ord.get_reordered()
        n_reordered_requests = len(reordered_requests)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        def _batch_scheduler(pos):
            sched = pos // int(n_reordered_requests / self.batch_schedule)
            if sched in self.batch_sizes:
                return self.batch_sizes[sched]
            print(
                f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
            )
            self.batch_sizes[sched] = self._detect_batch_size(reordered_requests, pos)
            print(f"Determined largest batch size: {self.batch_sizes[sched]}")
            return self.batch_sizes[sched]

        for chunk in utils.chunks(
            tqdm(reordered_requests, disable=disable_tqdm),
            n=self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0,
            fn=_batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None,
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                inplen = inplen + (
                    logits.shape[0] - padding_length
                )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        logits = self._model(inps)

        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        generations_no_kv_cache, _ = GenerationUtils(
            decoder_lm=self._model,
            eos_id=eos_token_id,
            pad_id=self._tokenizer.pad_id,
        ).generate(
            prompt_tokens=context,
            incremental_decode=False,
            min_gen_len=1,
            max_gen_len=max_length,
            top_k=3,
            device=torch.cuda.current_device(),
        )
        gens = generations_no_kv_cache.tolist()[0]
        return gens

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("`generate_until` is not implemented.")


def main(
    native_checkpoint_path: str,
    tasks: List[str] = ["hellaswag"],  # noqa: B006
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        native_checkpoint_path (str): The path to the model checkpoint file to load.
        tasks (List[str]): The name of the evaluation task or a list of tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    """

    tokenizer = llama2_tokenizer(args.tokenizer_path)

    seed(1234)

    device = _get_device_from_env()
    # --------- Initialize a decoder w/o kv-caching -------- #
    with device:
        model = llama2_7b()

    # Load state_dict into model
    native_state_dict = torch.load(args.native_checkpoint_path, weights_only=True)
    # Note: If using pretrained model, replace native_state_dict["model"] with native_state_dict
    missing, unexpected = model.load_state_dict(
        native_state_dict["model"], strict=False
    )

    model.eval()

    print("Loading model ...")
    t0 = time.time()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.")

    t1 = time.time()
    with torch.no_grad():
        model_eval_wrapper = ModelEvalWrapper(
            model,
            tokenizer,
            max_seq_length,
        )

        lm_eval.tasks.initialize_tasks()
        task_dict = lm_eval.tasks.get_task_dict(tasks)

        eval_results = lm_eval.evaluator.evaluate(
            model_eval_wrapper,
            task_dict,
            limit=10,
        )
    print(f"Time to run eval: {time.time() - t1:.02f} seconds.")
    print(f"For model {args.native_checkpoint_path}")
    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a finetuned model.")

    parser.add_argument(
        "--native-checkpoint-path", type=str, help="Path to native checkpoint file."
    )
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenization file.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["mmlu"],
        help="list of lm-eleuther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="number of samples to evalulate"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="maximum length sequence to evaluate",
    )

    args = parser.parse_args()
    main(
        args.native_checkpoint_path,
        args.tasks,
        args.limit,
        args.max_seq_length,
    )
