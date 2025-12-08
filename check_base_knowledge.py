import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

model_name = "Qwen/Qwen2-7B-Instruct"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

df = pd.read_csv("/home/hice1/mdoutre3/LLM_project_beta/wikifactdiff_converted.csv")

# first 1000 examples
num_test = min(1000, len(df))
print(f"Testing first {num_test} examples...")

results = []

for idx in tqdm(range(num_test), desc="Testing base model"):
    row = df.iloc[idx]
    q = row["question"]
    old = row["old_answer"]
    new = row["new_answer"]
    
    prompt = f"As of today, {q.lower()}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=15, do_sample=False)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("Answer:")[-1].strip().split("\n")[0].split(".")[0].strip()
    
    # Check what the model says
    answer_lower = answer.lower()
    old_lower = old.lower()
    new_lower = new.lower()
    
    has_old = old_lower in answer_lower
    has_new = new_lower in answer_lower
    has_both = has_old and has_new
    has_neither = not has_old and not has_new
    
    results.append({
        "fact_idx": idx,
        "question": q,
        "old_answer": old,
        "new_answer": new,
        "model_answer": answer,
        "has_old": has_old,
        "has_new": has_new,
        "has_both": has_both,
        "has_neither": has_neither,
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv("/home/hice1/mdoutre3/scratch/base_model_knowledge_check.csv", index=False)

# Calculate statistics
print("BASE MODEL KNOWLEDGE ANALYSIS")
print(f"Total tested: {len(results_df)}")
print()

old_count = results_df["has_old"].sum()
new_count = results_df["has_new"].sum()
both_count = results_df["has_both"].sum()
neither_count = results_df["has_neither"].sum()

print(f"Model says OLD answer:     {old_count:4d} / {num_test} ({100*old_count/num_test:.1f}%)")
print(f"Model says NEW answer:     {new_count:4d} / {num_test} ({100*new_count/num_test:.1f}%)")
print(f"Model says BOTH:           {both_count:4d} / {num_test} ({100*both_count/num_test:.1f}%)")
print(f"Model says NEITHER:        {neither_count:4d} / {num_test} ({100*neither_count/num_test:.1f}%)")
print()

# Exclusive categories
only_old = results_df[results_df["has_old"] & ~results_df["has_new"]].shape[0]
only_new = results_df[results_df["has_new"] & ~results_df["has_old"]].shape[0]

print("Exclusive categories:")
print(f"ONLY old (not new):        {only_old:4d} / {num_test} ({100*only_old/num_test:.1f}%)")
print(f"ONLY new (not old):        {only_new:4d} / {num_test} ({100*only_new/num_test:.1f}%)")
print()

print("INTERPRETATION:")
print(f"only_new = {only_new} new vs only_odd = {only_old}")

print()
print(f"Results saved to: /home/hice1/mdoutre3/scratch/base_model_knowledge_check.csv")

print("EXAMPLE CASES")

# Show 3 examples where model says NEW
new_examples = results_df[results_df["has_new"] & ~results_df["has_old"]].head(3)
if len(new_examples) > 0:
    print("\n--- Examples where model says NEW answer ---")
    for _, row in new_examples.iterrows():
        print(f"\nQ: {row['question']}")
        print(f"   Old: {row['old_answer']} | New: {row['new_answer']}")
        print(f"   Model: {row['model_answer']}")

# Show 3 examples where model says OLD
old_examples = results_df[results_df["has_old"] & ~results_df["has_new"]].head(3)
if len(old_examples) > 0:
    print("\n--- Examples where model says OLD answer ---")
    for _, row in old_examples.iterrows():
        print(f"\nQ: {row['question']}")
        print(f"   Old: {row['old_answer']} | New: {row['new_answer']}")
        print(f"   Model: {row['model_answer']}")

# Show 3 examples where model says NEITHER
neither_examples = results_df[results_df["has_neither"]].head(3)
if len(neither_examples) > 0:
    print("\n--- Examples where model says NEITHER ---")
    for _, row in neither_examples.iterrows():
        print(f"\nQ: {row['question']}")
        print(f"   Old: {row['old_answer']} | New: {row['new_answer']}")
        print(f"   Model: {row['model_answer']}")