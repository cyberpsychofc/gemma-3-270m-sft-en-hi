import os
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def flatten(example):
    return {'en': example['translation']['en'], 'hi': example['translation']['hi']}

data = load_dataset("cfilt/iitb-english-hindi", cache_dir=os.environ['DATASET_CACHE_DIR'], split="test[:100]")
data = data.map(flatten)

sources = [example["en"] for example in data]
references = [[example["hi"]] for example in data]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "google/gemma-3-270m"
BATCH_SIZE = 1

tokenizer = AutoTokenizer.from_pretrained("./models/lora_gpu", cache_dir=os.environ['HUB_CACHE_DIR'])

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    cache_dir=os.environ['HUB_CACHE_DIR']
)
base_model.resize_token_embeddings(len(tokenizer))  # Prevent size mismatch if tokenizer size differs

peft_model = PeftModel.from_pretrained(base_model, "./models/lora_gpu").to(device)

prompt = """
### Instruction:
Translate from English to Hindi

### Input:
{input}

### Output:
"""

predictions = []
for i in tqdm(range(0, len(sources), BATCH_SIZE)):
    batch_end = min(i + BATCH_SIZE, len(sources))
    batch_sources = sources[i:batch_end]
    
    # Prepare prompts
    prompts = [prompt.format(input=src) for src in batch_sources]
    
    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode predictions, skipping the prompt part
    batch_preds = []
    for j, output in enumerate(outputs):
        pred_tokens = output[len(inputs["input_ids"][j]):]  # Remove input tokens
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
        batch_preds.append(pred_text)
    
    predictions.extend(batch_preds)

print("Computing BLEU score...")
bleu = sacrebleu.corpus_bleu(predictions, references)
print(f"BLEU Score: {bleu.score:.2f}")
print(f"Full BLEU details: {bleu}")