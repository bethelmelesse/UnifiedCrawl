
from datasets import load_dataset
import re

LANGUAGE = "amh"
DATASET_PATH = f"./datasets/{LANGUAGE}_dedup/*.arrow"
NEW_DATASET_PATH = f"./datasets/{LANGUAGE}_dedup_filtered/"

dataset = load_dataset("arrow", data_files={DATASET_PATH}, split='train')

def merge_newlines(example):
    return {"text": re.sub(r'\n+', '\n', example["text"]).strip()}

dataset = dataset.map(merge_newlines, num_proc=8)
dataset = dataset.filter(lambda example: len(example["text"]) > 100, num_proc=8)
# shuffle
dataset = dataset.shuffle(seed=42)
dataset = dataset.flatten_indices()
dataset.save_to_disk(NEW_DATASET_PATH)

print()