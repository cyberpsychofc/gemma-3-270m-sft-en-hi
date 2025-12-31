import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

# CUDA required for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ['HUB_CACHE_DIR'], token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", cache_dir=os.environ['HUB_CACHE_DIR'], token=os.environ['HF_TOKEN'])