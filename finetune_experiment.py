import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")

# 70 / 15 / 15 split
split_70_30 = dataset.train_test_split(test_size=0.30, seed=42)
train_ds = split_70_30["train"]

split_15_15 = split_70_30["test"].train_test_split(test_size=0.50, seed=42)
val_ds = split_15_15["train"]
test_ds = split_15_15["test"]

val_ds = val_ds.select(range(200)) 

SYSTEM = "You are a medical reasoning assistant. Provide detailed reasoning and final answer."

def format_example(example):
    return {
        "text": f"""<|system|>
{SYSTEM}

<|user|>
{example['Question']}

<|assistant|>
{example['Complex_CoT']}

Final Answer: {example['Response']}
"""
    }

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map={"": 0},
)

# for name, _ in model.named_modules():
#     if "proj" in name or "mlp" in name or "fc" in name:
#         print(name)

# TARGET_MODULES = ["gate_up_projs", "down_projs"] # ["v_proj", "o_proj"]  # <- CHANGE THIS per experiment
TARGET_MODULES = ["v_proj", "o_proj"]

num_layers = model.config.num_hidden_layers

# Code for MLP fine-tuning (all experts) - Uncomment if needed
# for i in range(num_layers):
#     # each layer has ~32 experts → pick ALL
#     for expert_id in range(32):
#         TARGET_MODULES.append(
#             f"model.layers.{i}.mlp.experts.gate_up_projs.{expert_id}"
#         )
#         TARGET_MODULES.append(
#             f"model.layers.{i}.mlp.experts.down_projs.{expert_id}"
#         )

# print("Total target modules:", len(TARGET_MODULES))


model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=TARGET_MODULES,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)



# -------------------------
# TRAIN TOKENIZATION (unchanged)
# -------------------------
def tokenize_train(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        # padding="max_length",
    )


# -------------------------
# VALIDATION TOKENIZATION (MASK CoT)
# -------------------------
def tokenize_val(example):
    text = example["text"]

    split_token = "Final Answer:"
    idx = text.find(split_token)

    # Tokenize full text
    enc = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

    labels = enc["input_ids"].copy()

    if idx != -1:
        # Tokenize prefix separately
        prefix = text[:idx]
        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=2048,
            padding=False,
        )["input_ids"]

        prefix_len = len(prefix_ids)

        # Mask everything before final answer
        labels[:prefix_len] = [-100] * prefix_len

    # If "Final Answer" not found → no masking (fallback)
    enc["labels"] = labels
    return enc

train_tok = train_ds.map(tokenize_train, batched=True)
val_tok = val_ds.map(tokenize_val)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="exp_v_proj_o_proj",  # CHANGE THIS per experiment
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    
    num_train_epochs=1,
    learning_rate=5e-5,
    
    logging_steps=20,
    
    eval_strategy="steps",
    eval_steps=50,
    
    # FIX: match eval strategy
    save_strategy="steps",
    save_steps=50,   # same as eval_steps
    
    save_total_limit=2,
    dataloader_num_workers=4,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    #no code 
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    
    optim="paged_adamw_32bit",
    report_to="none",
    logging_dir="./logs",
    logging_strategy="steps",
)
#Log
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,   # IMPORTANT
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

save_path = f"lora_v_proj_o_proj"  # CHANGE THIS per experiment
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Saved to:", save_path) 