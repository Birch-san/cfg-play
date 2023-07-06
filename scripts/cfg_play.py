from src.cfg_logits import CFGLogits
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer

# forked from Grant Celley's https://github.com/huggingface/transformers/issues/24536#issuecomment-1622725548
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

device = torch.device('cuda:0')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").eval().to(device)

prompt = tokenizer("Today a dragon flew over Paris, France,", return_tensors='pt')
# either provide a negative prompt:
neg_prompt = tokenizer("Here begins my tale of woe,", return_tensors='pt')['input_ids']
# or don't:
# neg_prompt = prompt['input_ids'][:, -1:]

def on_text(message: str, stream_end = False):
    print(message, end='', flush=True)

streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)
outputs = model.generate(
    input_ids=prompt['input_ids'].to(device),
    attention_mask=prompt['attention_mask'].to(device),
    max_new_tokens=125,
    logits_processor=LogitsProcessorList([
        # inputs_cfg usually is the last token of the prompt but there are
        # possibilities of negative prompting that are explored in the paper
        CFGLogits(1.5, neg_prompt.to(device), model),
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