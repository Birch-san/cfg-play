from src.cfg_logits import CFGLogits

# from https://github.com/huggingface/transformers/issues/24536#issuecomment-1622725548
# by Grant Celley
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")

prompt = tokenizer("Today a dragon flew over Paris, France,", return_tensors='pt')
# either provide a negative prompt:
neg_prompt = tokenizer("A sad event happened,", return_tensors='pt')['input_ids']
# or don't:
# neg_prompt = prompt['input_ids'][:, -1:]

device='cuda:0'
model.to(device)
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
)

print(tokenizer.decode(outputs[0]))