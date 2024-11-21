from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
# import transformers


# For now, this wont LORA-fy embeddings and layernorms. Check if they are being trained (we ideally want them to be)
# embed_tokens is an embedding, do we want to finetune it? I dunno. It many. It important.

# MODEL = "facebook/xglm-4.5B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )

# 8bit seems to not work with xglm for qlora, gives some error
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit = True, 
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     llm_int8_has_fp16_weight=True
#     )


# tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left', truncation_side='left', truncation=True, max_length=2000)

# model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, device_map="auto", quantization_config=quantization_config)    # this line is done  
# model.gradient_checkpointing_enable()      # already handled

def model_after_qlora(model):

    model = prepare_model_for_kbit_training(model)

# Later on make this automatic to autmatically add any Linear to target modules, and any layernorms and embeddings to modules_to_save
# Not adding layernomrs as modules to save does not accept regex, and if you finetune xglm embed_tokens, too many to finetune.
# Idea/todo: Only finetune amharic tokens in embed_tokens. Freeze the rest.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=2, 
        lora_alpha=8, 
        lora_dropout=0.1, 
        target_modules=[r".*self_attn\..*", r".*fc.*", "lm_head"], 
        # modules_to_save=["model.embed_tokens"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model




# quantization_config=quantization_config if model_args.finetune_with_qlora else None,