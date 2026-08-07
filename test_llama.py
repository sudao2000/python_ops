import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.optim as optim
import fallback
fallback.enable_fallback() # 引入fallback机制

torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 引入blocking机制，方便调试哪个算子出了问题

# model_name = "/home/kurt/work/model/Qwen2.5-0.5B"
model_name = "/media/kurt/mac/model/qwen2.5-14b/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    _attn_implementation='eager',
    torch_dtype=torch.bfloat16
)
model = model.to('cuda')

text = "I love this product"
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
