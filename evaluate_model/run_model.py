import time
# time.sleep(86400)
from hyperparameters import AutoModelForCausalLM, MODEL, QUANTIZE_MODEL, device, quantization_config, LANGUAGE, QLORA_MODEL, DEBUG
from transformers import AutoTokenizer
from evaluate_model import evaluate_model
from predictions import get_prediciton_and_ground_truth
from visualize import print_output_and_scores
from peft import PeftModel, PeftConfig

# export BNB_CUDA_VERSION=118
def get_base_name():
    # This is to detect if this is ours QLoRa model, so that we can automatically load the original model
    if QLORA_MODEL:
        base_model_name = PeftConfig.from_pretrained(MODEL).base_model_name_or_path
    else:
        base_model_name = MODEL

    return base_model_name

def quantize_model(quantize_model):
    base_model_name = get_base_name()

    if quantize_model:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, low_cpu_mem_usage=True, device_map="auto", quantization_config=quantization_config).eval()
        if QLORA_MODEL:
            model = PeftModel.from_pretrained(model, MODEL).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device).eval()
        
    return model

def main():
    start = time.time()
    base_model_name = get_base_name()
    if "llama" in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left', truncation_side='left', truncation=True, max_length=2000, legacy=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    elif "xglm" in MODEL or "bloom" in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left', truncation_side='left', truncation=True, max_length=2000)

    elif "gpt2" in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left', truncation_side='left', truncation=True, max_length=1000)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            
    model = quantize_model(quantize_model=QUANTIZE_MODEL)

    all_predictions, ground_truth, questions = get_prediciton_and_ground_truth(model, tokenizer)

    '''getting score'''
    max_f1_score, max_em_score = evaluate_model(all_predictions=all_predictions, ground_truth=ground_truth)

    '''saving score'''
    print_output_and_scores(questions, ground_truth, all_predictions, max_f1_score, max_em_score)


    end = time.time()
    print("Time took - ", end - start)

main()
# kiml run submit --name run-17 --experiment bethel-xglm --image bethel-xglm --instance-type 1A100-16-MO --source-directory ./evaluate_model "python run_xglm_model.py"
#  kiml run submit --name run-9 --experiment bethel-xglm --image bethel-xglm2 --instance-type 0.14A100-2-MO --source-directory ./evaluate_model "python run_model.py"