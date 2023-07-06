from transformers import LogitsProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor, cat, tensor
from typing import List, Optional, Protocol, Union

class CausalLM(Protocol):
    def __call__(
        self,
        tokens: LongTensor,
        use_cache=True,
        attention_mask: Union[LongTensor, BoolTensor, None] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
    ) -> CausalLMOutputWithPast: ...

# forked from Guillaume "Vermeille" Sanchez's CFGLogits https://github.com/huggingface/transformers/issues/24536
# modifications to support multi-cond guidance by Alex Birch
class MultiCondLogits(LogitsProcessor):
    model: CausalLM
    cond_scales: FloatTensor
    cond_scales_reduced_ctx: FloatTensor
    ctx_cond: LongTensor
    ctx_cond_mask: Union[LongTensor, BoolTensor, None]
    rescale_factor: float
    out: Optional[CausalLMOutputWithPast]

    r"""Logits processor for Multi-cond guidance.
    This enables us to influence the next token by referring to the predictions from *multiple* conditions.
    More info on multi-cond guidance here:
    https://birchlabs.co.uk/machine-learning

    CFG is a special case of multi-cond guidance:
    https://twitter.com/Birchlabs/status/1627286152087478272
    Two condition weightings: one positive, one negative, summing to 1,
    with the positive being greater in magnitude by 1 than the negative.
    For example, CFG 2.5 could be expressed as two conditions, weighted thusly:
    [3.5, -2.5]

    The processor computes a weighted average across scores from prompt conditional and context logits,
    weighted by cond_scales.
    The context scores are computed internally by prompting `model` with
    the `ctx_cond` tokens.
    Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the LLM CFG paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
        cond_scales (`List[float]` of length `(ctx_conds + 1)`)
            A scale per condition, which must all add together to equal 1.
            The 0th element is the weight of our primary condition.
            Subsequent elements are the weights of the ctx_conds.
            
            In the simplest case (implementing CFG with one ctx_cond), you can express CFG 1.5 like so:
                [2.5, -1.5]
            The primary condition would get a weight of 2.5
            and the ctx_cond (your uncond or negative prompt) would get a weight of -1.5, which sums to 1 as required.

            Multi-cond guidance (a primary cond with two ctx_cond) can be expressed like so:
                [2., 2., -3.]
            The primary condition would get a weight of 2.
            and the two ctx_conds (some condition you wish to boost, and some condition you wish to negate)
            would get weights of 2 and -3 respectively.
            These weights sum to 1, as required.

            You can do multi-cond guidance where all conditions are positive, like so:
                [0.5, 0.25, 0.25]
            The primary condition gets a weight of 0.5,
            and each of the two ctx_conds get a weight of 0.25.
            These weights sum to 1, as required.
        ctx_cond (`torch.LongTensor` of shape `(ctx_conds, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        ctx_cond_mask (`torch.LongTensor` of shape `(ctx_conds, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        rescale_factor (float) *optional*:
            The interpolation weight for multi-cond guidance. 1 means no rescaling, 0 reduces to the conditional scores without
            CFG. Turn it lower if the output degenerates. Lower values allow for higher guidance scale.
    """
    def __init__(
        self,
        model: CausalLM,
        cond_scales: List[float],
        ctx_cond: LongTensor,
        ctx_cond_mask: Union[LongTensor, BoolTensor, None] = None,
        rescale_factor=1.
    ):
        self.model = model
        assert sum(cond_scales) == 1
        self.cond_scales = tensor(cond_scales, device=ctx_cond.device).unsqueeze(-1).unsqueeze(-1)
        self.cond_scales_reduced_ctx = tensor([cond_scales[0], sum(cond_scales[1:])], device=ctx_cond.device).unsqueeze(-1).unsqueeze(-1)
        assert len(cond_scales) == ctx_cond.size(0)+1, f'Expected cond_scales batch dim (0) to be `ctx_cond.size(0)+1` == {ctx_cond.size(0)+1}, instead got {cond_scales.size(0)}. the idea is to have a scale for the primary cond plus a scale per ctx_cond.'
        self.ctx_cond = ctx_cond
        if ctx_cond_mask is not None:
            assert ctx_cond_mask.shape == ctx_cond.shape
        self.ctx_cond_mask = ctx_cond_mask
        self.rescale_factor = rescale_factor
        self.out = None

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        scores = F.log_softmax(scores, dim=-1)
        if self.cond_scales.size(0) == 1:
            # why are you using this logits processor
            return scores

        if self.out is None:
            scales: FloatTensor = self.cond_scales
            self.out = self.model(
                self.ctx_cond,
                attention_mask=self.ctx_cond_mask,
                # avoiding cache because subsequent batch-of-1 lookups go bang due to mismatched batch dim in cache key
                use_cache=False,
            )
        else:
            scales: FloatTensor = self.cond_scales_reduced_ctx
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
        ctx_logits: FloatTensor = F.log_softmax(self.out.logits[:,-1:], dim=-1)
        all_logits: FloatTensor = cat([scores.unsqueeze(0), ctx_logits])
        out: FloatTensor = (all_logits * scales).sum(0)
        out = F.log_softmax(out, dim=-1)
        if self.rescale_factor == 1:
            return out
        return self.rescale_factor * out + (1 - self.rescale_factor) * scores