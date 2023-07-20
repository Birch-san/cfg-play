# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from transformers.utils import add_start_docstrings
from transformers.utils.logging import get_logger
from transformers import LogitsProcessor
from transformers.generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING


logger = get_logger(__name__)

class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""Logits processor for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
    """

    def __init__(self, guidance_scale):
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # simple check to make sure we have compatible batch sizes between our
        # logits scores (cond + uncond) and input ids (cond only)
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        return scores


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
        smooth_factor (float, **optional**):
            The interpolation weight for CFG Rescale. 1 means no rescaling, 0 reduces to the conditional scores without
            CFG. Turn it lower if the output degenerates.
        use_cache (bool, **optional**):
            Whether to cache key/values during the negative prompt forward pass.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, **optional**):
            Attention mask for unconditional_ids.


    Examples:

    ```python
    >>> # base prompt emphasis example
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> summary_ids = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The dragon flew over Paris, France, landing in Lyon, a city of a few million. Dragon-flying was a new form of
    transport, and the dragon was the first in Europe.

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> summary_ids = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt=neg_inputs["input_ids"])
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The dragon flew over Paris, France, crashing into Notre Dame Cathedral in the French capital killing at least 127
    people and injuring more than 350.
    ```
    """

    def __init__(
        self,
        guidance_scale,
        unconditional_ids,
        model,
        use_cache=True,
        unconditional_attention_mask=None,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.negative_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask or torch.ones_like(unconditional_ids, dtype=torch.long),
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def neg_logits(self, input_ids):
        ctx = self.negative_context

        if ctx["first_pass"]:
            if ctx["input_ids"] is None:
                ctx["input_ids"] = input_ids[:, -1:]
            input_ids = ctx["input_ids"]
            attention_mask = ctx["attention_mask"]
            ctx["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [ctx["attention_mask"], torch.ones_like(input_ids[:, -1:], dtype=torch.long)], dim=1
            )
            if not ctx["use_cache"]:
                input_ids = torch.cat([ctx["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            ctx["input_ids"] = input_ids
            ctx["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=ctx["use_cache"],
            past_key_values=ctx["past_key_values"],
        )
        ctx["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.neg_logits(input_ids)

        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out
