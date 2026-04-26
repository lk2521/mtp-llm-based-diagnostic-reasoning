import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel
import evaluate
import re
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

# -------------------------
# CONFIG
# -------------------------
BASE_MODEL = "unsloth/gpt-oss-20b"

# MODELS = {
#     "base": None,
#     "v_o": "lora_v_proj_o_proj",
#     "v": "lora_v_proj",
#     "mlp": "lora_mlp",
# }
MODELS = {
    "top_4": "lora_v_o_top_4",
    "top_8": "lora_v_o_top_8",
    "top_half": None,
}


NUM_SAMPLES = 10

# -------------------------
# DATA
# -------------------------
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")

split_70_30 = dataset.train_test_split(test_size=0.30, seed=42)
split_15_15 = split_70_30["test"].train_test_split(test_size=0.50, seed=42)
test_ds = split_15_15["test"].select(range(NUM_SAMPLES))

# -------------------------
# METRICS
# -------------------------
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# -------------------------
# HELPERS
# -------------------------
def extract_final_answer(text):
    if "Final Answer:" not in text:
        return "NO_ANSWER"
    match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip()


class StopOnFinalAnswer(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return "Final Answer:" in decoded


def generate(model, tokenizer, question):
    prompt = f"""<|system|>
You are a medical reasoning assistant.

<|user|>
{question}

<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([StopOnFinalAnswer(tokenizer)])
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# -------------------------
# QUANT CONFIG (🔥 FIX)
# -------------------------
quant_config = Mxfp4Config()

# -------------------------
# EVALUATION LOOP
# -------------------------
results = {}

for name, lora_path in MODELS.items():
    print(f"\n====================")
    print(f"Evaluating: {name}")
    print(f"====================")

    torch.cuda.empty_cache()

    # 🔥 LOAD BASE MODEL WITH MXFP4
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,   # FIXED
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    # 🔥 LOAD LORA
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    preds, refs = [], []

    for example in tqdm(test_ds):
        question = example["Question"]
        gt_answer = example["Response"]

        output = generate(model, tokenizer, question)

        pred = extract_final_answer(output)
        ref = gt_answer.strip()

        preds.append(pred)
        refs.append(ref)

    # -------------------------
    # METRICS
    # -------------------------
    rouge_score = rouge.compute(predictions=preds, references=refs)

    bleu_score = bleu.compute(
        predictions=[p.split() for p in preds],
        references=[[r.split()] for r in refs]
    )
    bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")

    results[name] = {
        "ROUGE-L": rouge_score["rougeL"],
        "BLEU": bleu_score["bleu"],
        "BERTScore-F1": sum(bert_score["f1"]) / len(bert_score["f1"]),
    }

    # 🔥 CLEANUP
    del model
    torch.cuda.empty_cache()


# -------------------------
# PRINT RESULTS
# -------------------------
print("\n====================")
print("FINAL RESULTS")
print("====================\n")

for k, v in results.items():
    print(k)
    for metric, val in v.items():
        print(f"  {metric}: {val:.4f}")
    print()