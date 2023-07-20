from src.cfg_logits import UnbatchedClassifierFreeGuidanceLogitsProcessor
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer

# forked from Grant Celley's https://github.com/huggingface/transformers/issues/24536#issuecomment-1622725548
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import torch
from torch import inference_mode

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")

device = torch.device('cuda:0')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b").eval().to(device=device, dtype=torch.float16)

prompt = tokenizer("the shopkeeper could tell that it was time to ", return_tensors='pt')
# either provide a negative prompt:
neg_prompt = tokenizer("because it was night-time, it was time to ", return_tensors='pt')['input_ids']
# or don't:
# neg_prompt = prompt['input_ids'][:, -1:]

def on_text(message: str, stream_end = False):
    print(message, end='', flush=True)

cfg_proc = UnbatchedClassifierFreeGuidanceLogitsProcessor(1.5, neg_prompt.to(device), model)

set_seed(64)

streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)
with inference_mode():
    output = model.generate(
        input_ids=prompt['input_ids'].to(device),
        attention_mask=prompt['attention_mask'].to(device),
        max_new_tokens=125,
        logits_processor=LogitsProcessorList([
            cfg_proc,
            TemperatureLogitsWarper(0.8),
            TopPLogitsWarper(0.95),
        ]),
        do_sample=True,
        streamer=streamer,
    )
    # print a line break once we've finished streaming
    print('')

    # no need to print response; we already streamed it out
    # print(tokenizer.decode(output[0, prompt['input_ids'].size(-1):]))