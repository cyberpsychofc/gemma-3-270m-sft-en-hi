import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

model_id = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ['HUB_CACHE_DIR'], token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", cache_dir=os.environ['HUB_CACHE_DIR'], token=os.environ['HF_TOKEN'])