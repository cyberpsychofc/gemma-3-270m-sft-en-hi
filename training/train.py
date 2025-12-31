import os
import transformers
import torch
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dotenv import load_dotenv

from ..utils.config import bnb_config, lora_config

load_dotenv()



def flatten(example):
    return {'en': example['translation']['en'], 'hi': example['translation']['hi']}

def formatting_text(example):
    text = (
        "### Instruction:\n"
        "Translate from English to Hindi\n\n"
        "### Input:\n"
        f"{example['en']}\n\n"
        "### Output:\n"
        f"{example['hi']}"
    )
    return text

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "google/gemma-3-270m"
model_kwargs = {
    "cache_dir": os.environ['HUB_CACHE_DIR'], 
    "token": os.environ['HF_TOKEN'],
    "device_map": device
}

if device == "cuda:0":
    model_kwargs["quantization_config"] = bnb_config

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ['HUB_CACHE_DIR'], token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
model.resize_token_embeddings(len(tokenizer))   # Prevent size mismatch if tokenizer size differs

os.environ["WANDB_DISABLED"] = "true"

data = load_dataset("cfilt/iitb-english-hindi", cache_dir=os.environ['DATASET_CACHE_DIR'], split="train[:1000]")
data = data.map(flatten)

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=3,
        warmup_ratio=0.1,
        num_train_epochs=10,
        learning_rate=1e-4,
        bf16= (device == "cuda:0"),
        logging_steps=10,
        output_dir="outputs",
        save_strategy="no",
        optim= "paged_adamw_8bit" if (device == "cuda:0") else "adamw_torch"
    ),
    peft_config=lora_config,
    formatting_func=formatting_text,
)

trainer.train()

trainer.save_model("./models/lora_cpu")
tokenizer.save_pretrained("./models/lora_cpu")