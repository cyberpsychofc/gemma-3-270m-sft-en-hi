import os
from datasets import load_dataset

dataset = load_dataset("cfilt/iitb-english-hindi", cache_dir=os.environ['DATASET_CACHE_DIR'], split= "train[:100]")