import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import fallback

torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# enable fallback AFTER model loading, to avoid interfering with transformers imports
fallback.enable_fallback()

model_name = "/home/kurt/work/model/qwen2.5-14b"
# model_name = "D:/Code/Phytium/python_ops-main/Qwen2.5-7B/"
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers = 1
config.max_window_layers = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
config._attn_implementation = "sdpa"  # 设在config上，不走from_pretrained传参

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16
)
model = model.to('cuda')

import sys as _sys
_texts = [
    "Hello, how are you",                                              # 0: short
    "I love this product very much, because this is a good product",   # 1: long
]
text = _texts[int(_sys.argv[1])] if len(_sys.argv) > 1 else _texts[0]

inputs = tokenizer(text, return_tensors="pt").to('cuda')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
