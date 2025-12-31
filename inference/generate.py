import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from dotenv import load_dotenv

load_dotenv()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained("./models/lora_gpu", cache_dir=os.environ['HUB_CACHE_DIR'])
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    cache_dir=os.environ['HUB_CACHE_DIR']
)
base_model.resize_token_embeddings(len(tokenizer)) # Prevent size mismatch if tokenizer size differs

# Align tokenizer tokens as before
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_model = PeftModel.from_pretrained(base_model, "./models/lora_gpu")

input = input("\nEnter : ")

prompt = f"""
### Instruction:
Translate from English to Hindi

### Input:
{input}

### Output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = peft_model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0], skip_special_tokens=True))