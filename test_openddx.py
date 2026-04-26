import pandas as pd
import re
import json
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from peft import PeftModel

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_excel('openDDx.xlsx')

# -------------------------
# CONFIG
# -------------------------
BASE_MODEL = "unsloth/gpt-oss-20b"

MODELS = {
    #"base": None,
    "v": "lora_v_proj",
    "v_o": "lora_v_proj_o_proj",
    #"mlp": "lora_mlp",
}

quant_config = Mxfp4Config()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# -------------------------
# PROMPT BUILDER
# -------------------------
def build_prompt(row):
    return f"""<|system|>
You are a clinical reasoning assistant.

Instructions:
- Think step by step to give the most likely diagnosis based on the patient information.
- Keep reasoning brief (4-5 sentences)

Output Rules (STRICT):
- Follow the format exactly
- Do NOT write anything after the Final Output line
- Stop immediately after writing Final Output
- Do NOT add explanations, notes, or extra text

Answer format:
Reasoning: <your reasoning>
Final Output: <one most likely diagnosis>

<|user|>
PATIENT CASE:
specialty: {row['specialty']}
patient_text: \"\"\"{row['patient_info']}\"\"\"

<|assistant|>
Reasoning:"""
# -------------------------
# JSON FIX
# -------------------------
def fix_and_load_json(text):
    import re
    import json

    # extract ONLY final JSON block
    match = re.search(r'Final Output:\s*(\{.*\})', text, re.DOTALL)

    if not match:
        return None

    json_str = match.group(1).strip()

    # fix trailing commas
    json_str = re.sub(r',\s*\]', ']', json_str)

    try:
        return json.loads(json_str)
    except:
        return None

# -------------------------
# GENERATION
# -------------------------
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # clean assistant output
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[1]

    return text.strip()

# -------------------------
# MAIN LOOP
# -------------------------
for model_name, lora_path in MODELS.items():
    print(f"\nRunning model: {model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        model = base_model

    model.eval()

    preds = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = build_prompt(row)

        try:
            output = generate(model, prompt)
            parsed = fix_and_load_json(output)

            if parsed and "predicted_diseases" in parsed:
                preds.append(str(parsed["predicted_diseases"]))
            else:
                preds.append(None)

        except Exception as e:
            print(f"Error at row {i}: {e}")
            preds.append(None)

    # save predictions for this model
    df[f"predicted_{model_name}"] = preds

    # ✅ SAVE AFTER EACH MODEL
    df.to_csv("openDDx_all_models_results2.csv", index=False)
    print(f"Saved after model: {model_name}")

    # cleanup
    del model
    torch.cuda.empty_cache()