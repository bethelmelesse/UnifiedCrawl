import os
os.environ['HF_HOME']='/data/bethel/.cache/huggingface'
# os.environ['BNB_CUDA_VERSION']='118'
os.environ['PYTHONUNBUFFERED']='True'
# import time 
# time.sleep(86400)
import torch
from torch import nn 
import numpy as np
import random
from transformers import set_seed
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model")
args = parser.parse_args()
MODEL = args.model

device = "cuda" if torch.cuda.is_available() else "cpu" 

seed_value = 42
random.seed(seed_value) 
np.random.seed(seed_value)
torch.manual_seed(seed_value)
set_seed(seed_value)    # for transformer

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )

'''XGLM MODEL'''
# MODEL = "facebook/xglm-564M"
# MODEL = "facebook/xglm-4.5B"

'''gpt2 base'''
# MODEL = 'gpt2'

'''qLORA Models'''
# MODEL = "models/trained_output/amh/gpt2-74M-epoch20-lr2e-4"

'''Finetuned Models'''
# MODEL="models/finetuned_output/amh/facebook/xglm-564M-epoch1-lr2.5e-4-qlora"
# MODEL="models/finetuned_output/amh/facebook/xglm-4.5B-epoch1-lr1e-4-qlora"

'''Downstream Tuned Models'''
  
# MODEL="models/downstream_output/amh/facebook/xglm-4.5B-epoch6-lr5e-5-qlora"   # xglm4.5b
# MODEL=""           # xglm564M
# MODEL="models/downstream_output/amh/models/finetuned_output/amh/facebook/xglm-564M-epoch1-lr2.5e-4-qlora-epoch6-lr5e-5-qlora"            # xglm564M ours
# MODEL="models/downstream_output/amh/models/finetuned_output/amh/facebook/xglm-4.5B-epoch1-lr1e-4-qlora-epoch6-lr5e-5-qlora"           # xglm4.5b ours
# MODEL="models/downstream_output/amh/models/finetuned_output/amh/facebook/xglm-4.5B-epoch1-lr1e-4-qlora-epoch3-lr3e-5-qlora2"          # xglm4.5b ours 4 accum

# from run_model import MODEL

# MODEL = "models/finetuned_output/amh/facebook/xglm-564M-epoch1-lr2.5e-4"

QUANTIZE_MODEL = True
# QUANTIZE_MODEL = False

DEBUG = False
# DEBUG = False

MAX_LENGTH = 960
if 'gpt' in MODEL:
    MAX_LENGTH=960
if DEBUG:
    MAX_LENGTH = 960


BATCH_SIZE = 4            # can't use more than 1 batch_size for non-xglm due to HF bug in stop criteria
# assert(BATCH_SIZE==1)
NUM_PROMPT_EXAMPLE = 10

if "downstream" in MODEL:
    NUM_PROMPT_EXAMPLE=0
if "downstream" in MODEL:
    MAX_LENGTH=960

print(MAX_LENGTH)

# LANGUAGE = "en"
LANGUAGE = "amh"

QLORA_MODEL = "qlora" in MODEL 
