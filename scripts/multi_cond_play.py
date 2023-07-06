from src.multi_cond_logits import MultiCondLogits
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer

# forked from Grant Celley's https://github.com/huggingface/transformers/issues/24536#issuecomment-1622725548
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

from typing import TypedDict
from torch import LongTensor
import torch

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device=torch.device('cuda:0')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").eval().to(device)

primary_prompt: TokenizerOutput = tokenizer('The culprit was getting away, and', return_tensors='pt').to(device)

# tokenizer.padding_side = 'right'
context_prompts: TokenizerOutput = tokenizer([
  'As a magical girl,',
  'Being the catgirl that I am,',
], padding=True, return_tensors='pt').to(device)

def on_text(message: str, stream_end = False):
  print(message, end='', flush=True)

streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)
outputs = model.generate(
  input_ids=primary_prompt.input_ids,
  attention_mask=primary_prompt.attention_mask,
  max_new_tokens=125,
  logits_processor=LogitsProcessorList([
    MultiCondLogits(
      model=model,
      cond_scales=[0.5, 0.25, 0.25],
      ctx_cond=context_prompts.input_ids,
      ctx_cond_mask=context_prompts.attention_mask,
    ),
    TemperatureLogitsWarper(0.8),
    TopPLogitsWarper(0.95),
  ]),
  do_sample=True,
  streamer=streamer,
)
# print a line break once we've finished streaming
print('')

# no need to print response; we already streamed it out
# print(tokenizer.decode(outputs[0]))