import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ..utils.config import bnb_config
from dotenv import load_dotenv

load_dotenv()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained("./models/lora_gpu", cache_dir=os.environ['HUB_CACHE_DIR'])
model_kwargs = {
    "cache_dir": os.environ['HUB_CACHE_DIR'], 
    "token": os.environ['HF_TOKEN'],
    "device_map": device
}

if device == "cuda:0":
    model_kwargs["quantization_config"] = bnb_config
base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
base_model.resize_token_embeddings(len(tokenizer)) 

os.environ["WANDB_DISABLED"] = "true"

model = PeftModel.from_pretrained(base_model, "./models/lora_gpu")
model = model.merge_and_unload()

model = model.to(dtype=torch.float32, device="cpu")

# save merged CPU-ready model
model.save_pretrained("./models/final_model")
AutoTokenizer.from_pretrained(model_id).save_pretrained("./models/final_model")