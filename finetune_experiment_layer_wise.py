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

# -------------------------------------------------
# Choose WHICH layers to fine-tune
# -------------------------------------------------
num_layers = model.config.num_hidden_layers

LAYER_MODE = "top_half"   # change to: "top_8", "top_half", or "custom"

if LAYER_MODE == "top_4":
    selected_layers = list(range(max(0, num_layers - 4), num_layers))
elif LAYER_MODE == "top_8":
    selected_layers = list(range(max(0, num_layers - 8), num_layers))
elif LAYER_MODE == "top_half":
    selected_layers = list(range(num_layers // 2, num_layers))
elif LAYER_MODE == "custom":
    # Replace with your knockout-selected layer ids
    selected_layers = [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1]
else:
    raise ValueError(f"Unknown LAYER_MODE: {LAYER_MODE}")

def build_target_modules(model, selected_layers):
    target_modules = []
    for name, _ in model.named_modules():
        for layer_idx in selected_layers:
            if name == f"model.layers.{layer_idx}.self_attn.v_proj":
                target_modules.append(name)
            if name == f"model.layers.{layer_idx}.self_attn.o_proj":
                target_modules.append(name)
    return sorted(set(target_modules))

TARGET_MODULES = build_target_modules(model, selected_layers)

print("Selected layers:", selected_layers)
print("Number of target modules:", len(TARGET_MODULES))
for x in TARGET_MODULES[:10]:
    print(x)

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
# TRAIN TOKENIZATION
# -------------------------
def tokenize_train(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
    )

# -------------------------
# VALIDATION TOKENIZATION
# -------------------------
def tokenize_val(example):
    text = example["text"]

    split_token = "Final Answer:"
    idx = text.find(split_token)

    enc = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

    labels = enc["input_ids"].copy()

    if idx != -1:
        prefix = text[:idx]
        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=2048,
            padding=False,
        )["input_ids"]

        prefix_len = len(prefix_ids)
        labels[:prefix_len] = [-100] * prefix_len

    enc["labels"] = labels
    return enc

train_tok = train_ds.map(tokenize_train, batched=True)
val_tok = val_ds.map(tokenize_val)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=f"exp_v_o_{LAYER_MODE}",

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=1,
    learning_rate=5e-5,

    logging_steps=20,

    eval_strategy="steps",
    eval_steps=50,

    save_strategy="steps",
    save_steps=50,

    save_total_limit=2,
    dataloader_num_workers=4,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    optim="paged_adamw_32bit",
    report_to="none",
    logging_dir="./logs",
    logging_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

save_path = f"lora_v_o_{LAYER_MODE}"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Saved to:", save_path)