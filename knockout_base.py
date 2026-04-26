import torch
import numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel

# -------------------------
# Load small validation set
# -------------------------
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[:100]")

SYSTEM = "You are a medical reasoning assistant."

def format_example(example):
    return {
        "text": f"""<|system|>
{SYSTEM}

<|user|>
{example['Question']}

<|assistant|>
Final Answer: {example['Response']}
"""
    }

dataset = dataset.map(format_example)

# -------------------------
# Load BASE model (NO LoRA)
# -------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    load_in_4bit=True,
    device_map={"": 0},
)

model.eval()

# -------------------------
# Evaluation function
# -------------------------
def evaluate(model, dataset):
    losses = []

    for ex in dataset:
        inputs = tokenizer(ex["text"], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        losses.append(outputs.loss.item())

    return np.mean(losses)

# -------------------------
# Baseline
# -------------------------
base_loss = evaluate(model, dataset)
print("BASE LOSS:", base_loss)

# -------------------------
# Knockout function
# -------------------------
def knockout_and_eval(target_name):
    backups = {}

    for name, param in model.named_parameters():
        if target_name in name:
            backups[name] = param.data.clone()
            param.data.zero_()

    loss = evaluate(model, dataset)

    # restore
    for name, param in model.named_parameters():
        if name in backups:
            param.data.copy_(backups[name])

    return loss

# -------------------------
# Run experiments
# -------------------------
modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

results = {}

for m in modules:
    loss = knockout_and_eval(m)
    delta = loss - base_loss
    results[m] = delta
    print(f"{m}: Δloss = {delta:.4f}")

print("\nFINAL RANKING:")
print(sorted(results.items(), key=lambda x: -x[1]))