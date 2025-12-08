import torch
import re
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import gc

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATA_PATH = "/home/hice1/mdoutre3/LLM_project_beta/wikifactdiff_converted.csv"

POSSIBLE_LORA_PATHS = [
    "/home/hice1/mdoutre3/scratch/qwen25_individual_loras",
    "~/scratch/qwen25_individual_loras",
    "./qwen25_individual_loras",
    "../qwen25_individual_loras",
]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def find_lora_directory():
    for p in POSSIBLE_LORA_PATHS:
        p = Path(p).expanduser()
        if p.exists():
            print(f"✓ Found LoRA directory: {p}")
            return p
    raise FileNotFoundError("No LoRA folder found!")


def is_gibberish(text):
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.3:
        return True
    if re.search(r'(\b\w+\b)\s+\1\s+\1', text):
        return True
    special = sum(1 for c in text if c in '_-.,;:()[]{}')
        # high punctuation density
    if special / max(len(text), 1) > 0.4:
        return True
    return False


def reshape_to_lora(flat_tensor, shapes):
    """Turn flat vector back into dict of parameter-shaped tensors."""
    lora_dict = {}
    offset = 0
    for name, shape in shapes.items():
        size = 1
        for s in shape:
            size *= s
        lora_dict[name] = flat_tensor[offset:offset+size].reshape(shape)
        offset += size
    return lora_dict


# ---------------------------------------------------------
# Evaluation of ONE real LoRA
# ---------------------------------------------------------
def evaluate_real_lora(fact_idx, lora_dir, df, base_model, tokenizer, lora_shapes, device):
    fact_dir = lora_dir / f"fact_{fact_idx:04d}"
    lora_file = fact_dir / "lora_params.pt"

    if not lora_file.exists():
        return None, None, None, True, f"Missing LoRA for {fact_idx}"

    # Load LoRA
    lora_params = torch.load(lora_file, map_location="cpu", weights_only=True)
    flat = torch.cat([p.flatten().float() for p in lora_params.values()])
    lora_dict = reshape_to_lora(flat, lora_shapes)

    # Attach LoRA to model
    lora_config = LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", inference_mode=True,
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Inject weights
    sd = peft_model.state_dict()
    for name, param in lora_dict.items():
        if name in sd:
            sd[name].copy_(param.to(device))

    # Get question/true answer
    row = df.iloc[fact_idx]
    question = row["question"]
    true_answer = row["new_answer"].lower()

    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        output = peft_model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    gen = tokenizer.decode(output[0], skip_special_tokens=True).lower()

    gib = is_gibberish(gen)
    answer_words = [w for w in true_answer.split() if len(w) > 3]

    correct = any(w in gen for w in answer_words) if not gib else False

    # Cleanup
    del peft_model
    torch.cuda.empty_cache()
    gc.collect()

    return correct, gen, true_answer, gib, None


# ---------------------------------------------------------
# Evaluate many LoRAs
# ---------------------------------------------------------
def evaluate_many_real_loras(lora_dir, df, lora_shapes, model, tokenizer, device, max_examples=50):
    results = []
    for i in range(max_examples):
        correct, gen, true_ans, gib, err = evaluate_real_lora(
            i, lora_dir, df, model, tokenizer, lora_shapes, device
        )

        if err:
            print(f"[{i}] ERROR: {err}")
            continue

        print(f"[{i:04d}] Correct={correct} | Gib={gib}")
        print(f"   Output: {gen}")
        print(f"   Truth : {true_ans}\n")

        results.append(correct)

    acc = sum(results) / len(results) if results else 0
    print("="*80)
    print(f" REAL LORA ACCURACY on {len(results)} examples: {acc*100:.1f}%")
    print("="*80)
    return acc


# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------
if __name__ == "__main__":

    print("\n=== Loading model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    device = model.device

    print("=== Loading CSV ===")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} rows")

    print("=== Locating LoRAs ===")
    lora_dir = find_lora_directory()

    # Load shapes from first LoRA
    test_dir = next(lora_dir.glob("fact_*"))
    test_params = torch.load(test_dir / "lora_params.pt", map_location="cpu", weights_only=True)
    lora_shapes = {name: p.shape for name, p in test_params.items()}

    print("=== Evaluating REAL LoRAs ===")
    evaluate_many_real_loras(
        lora_dir, df, lora_shapes,
        model, tokenizer, device,
        max_examples=50
    )
