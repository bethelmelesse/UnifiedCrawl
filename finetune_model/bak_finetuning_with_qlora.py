from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
import transformers


# For now, this wont LORA-fy embeddings and layernorms. Check if they are being trained (we ideally want them to be)
# embed_tokens is an embedding, do we want to finetune it? I dunno. It many. It important.

MODEL = "facebook/xglm-4.5B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )


tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left', truncation_side='left', truncation=True, max_length=2000)

model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, device_map="auto", quantization_config=quantization_config)
model.gradient_checkpointing_enable()

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
# Check all models stuff using 
# print([(n, type(m)) for n, m in model.named_modules() if type(m)==torch.nn.Linear])

# model.save_pretrained("output_dir") 

from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# Check bf16
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=2e-4,
        weight_decay=1e-3,
        bf16=True, 
        logging_steps=1,
        output_dir="output_dir",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print()
# trainer.train()
print()