import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import sys
from contextlib import contextmanager

#os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Configuration
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Use 2.0, not 2.5
BASE_OUTPUT_DIR = "/home/hice1/mdoutre3/scratch/qwen25_individual_loras"
DATA_PATH = "/home/hice1/mdoutre3/LLM_project_beta/wikifactdiff_converted.csv"

# carefully chosen
LORA_CONFIG = {
    "r": 2,                     
    "lora_alpha": 4,            # Alpha: 2*r
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "init_lora_weights": "gaussian",
}

TRAINING_CONFIG = {
    "learning_rate": 1e-4,              
    "num_steps": 30,               
    "max_seq_length": 256, 
    "gradient_clip": 0.5,
}


def load_base_model():
    """Load base model and tokenizer once"""
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32,  # FP32 to avoid NaN issues
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, tokenizer


def create_lora_model(base_model):
    """Create a fresh LoRA model from base"""
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(base_model, lora_config)
    return model


def format_training_text(row, tokenizer):
    """Format single fact for training - train on NEW facts (not old)"""
    prompt = f"As of today, {row['question'].lower()}\nAnswer: {row['new_answer']}"
    return prompt


def train_single_lora(fact_idx, fact_row, base_model, tokenizer, output_dir, pbar=None):
    """Train LoRA adapter for a single fact using manual training loop"""
    import torch.optim as optim
    import torch.nn as nn
    
    fact_dir = Path(output_dir) / f"fact_{fact_idx:04d}"
    fact_dir.mkdir(parents=True, exist_ok=True)
    
    text = format_training_text(fact_row, tokenizer)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      max_length=TRAINING_CONFIG["max_seq_length"]).to(base_model.device)
    
    if pbar:
        pbar.set_description(f"Fact {fact_idx}: {fact_row['question'][:50]}...")
    
    model = create_lora_model(base_model)
    model.train()
    
    # Check initial LoRA weights
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"NaN/Inf in initial {name}")
    
    # Setup optimizer, only LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    optimizer = optim.AdamW(lora_params, lr=TRAINING_CONFIG["learning_rate"], 
                           weight_decay=0.01, eps=1e-8)
    

    for step in range(TRAINING_CONFIG["num_steps"]):
        optimizer.zero_grad()
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n Fact {fact_idx} step {step}: Invalid loss = {loss.item()}")
            # Check logits
            with torch.no_grad():
                test_outputs = model(**inputs)
                logits = test_outputs.logits
                print(f"  Logits: min={logits.min().item()}, max={logits.max().item()}, " +
                      f"has_nan={torch.isnan(logits).any()}, has_inf={torch.isinf(logits).any()}")
            raise ValueError(f"Invalid loss at step {step}")
        
        loss.backward()
        
        # Check gradients
        max_grad = 0.0
        for p in lora_params:
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())
                if torch.isnan(p.grad).any():
                    raise ValueError(f"NaN in gradient at step {step}")
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(lora_params, TRAINING_CONFIG["gradient_clip"])
        
        optimizer.step()
    
    model.eval()
    
    # Verify NaNs in final weights
    for name, param in model.named_parameters():
        if "lora" in name and (torch.isnan(param).any() or torch.isinf(param).any()):
            raise ValueError(f"NaN/Inf in final {name}")
    
    # Extract LoRA parameters (only save tensors, not full PEFT model)
    lora_params_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            # Save in bfloat16 to reduce size (4 bytes -> 2 bytes)
            lora_params_dict[name] = param.detach().cpu().to(torch.bfloat16)
    
    # Compress and save
    torch.save(lora_params_dict, fact_dir / "lora_params.pt", 
               _use_new_zipfile_serialization=True)  # Better compression ++
    
    metadata = {
        "fact_idx": fact_idx,
        "question": fact_row['question'],
        "answer": fact_row.get('new_answer', ''),  # Only save what we trained on
        "num_params": sum(p.numel() for p in lora_params_dict.values()),
    }
    
    with open(fact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, separators=(',', ':'))  # Compact JSON
    
    # Clean up
    del model, optimizer
    torch.cuda.empty_cache()
    
    return lora_params_dict, metadata


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train individual LoRA adapters per fact")
    parser.add_argument("--num_facts", type=int, default=None, 
                        help="Number of facts to train (default: all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index (default: 0)")
    #parser.add_argument("--single_file", action="store_true",
    #                    help="Save all LoRAs in single compressed file (saves space)")
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    output_dir = Path(BASE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Checking for existing LoRAs...")
    existing_facts = set()
    for fact_dir in output_dir.glob("fact_*"):
        if (fact_dir / "lora_params.pt").exists():
            idx = int(fact_dir.name.split("_")[1])
            existing_facts.add(idx)
    
    print(f"Found {len(existing_facts)} already trained LoRAs")
    
    if args.num_facts is not None:
        end_idx = min(args.start_idx + args.num_facts, len(df))
        df_slice = df.iloc[args.start_idx:end_idx]
        print(f"Training on facts {args.start_idx} to {end_idx-1} ({len(df_slice)} total)")
    elif args.start_idx > 0:
        df_slice = df.iloc[args.start_idx:]
        print(f"Training on facts {args.start_idx} to {len(df)-1} ({len(df_slice)} total)")
    else:
        df_slice = df
        print(f"Training on ALL {len(df)} facts")
    
    # Filter out already trained
    total_to_train = len(df_slice)
    remaining = sum(1 for idx in df_slice.index if idx not in existing_facts)
    print(f"Remaining to train: {remaining}/{total_to_train} (skipping {total_to_train - remaining} already done)")
    
    df = df_slice
    
    # Load base model
    base_model, tokenizer = load_base_model()
    
    global_config = {
        "model_name": MODEL_NAME,
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "num_facts": len(df),
        "start_idx": args.start_idx if args.num_facts else 0,
        "end_idx": (args.start_idx + len(df)) if args.num_facts else len(df),
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(global_config, f, indent=2)
    
    # Train LoRA for each fact with progress bar
    all_metadata = []
    failed_facts = []
    skipped_count = 0
    
    print(f"\nTraining individual LoRA adapters...")
    print(f"Output: {output_dir}\n")
    
    with tqdm(total=len(df), desc="Training LoRAs", unit="fact") as pbar:
        for idx, row in df.iterrows():
            # Check if already trained
            if idx in existing_facts:
                skipped_count += 1
                pbar.set_postfix({"skipped": skipped_count, "failed": len(failed_facts)})
                pbar.update(1)
                continue
            
            try:
                lora_params, metadata = train_single_lora(
                    fact_idx=idx,
                    fact_row=row,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    pbar=pbar,
                )
                all_metadata.append(metadata)
                pbar.update(1)
                
            except Exception as e:
                failed_facts.append({"idx": idx, "question": row['question'], "error": str(e)})
                pbar.set_postfix({"skipped": skipped_count, "failed": len(failed_facts)})
                pbar.update(1)
                continue
    
    summary_df = pd.DataFrame(all_metadata)
    summary_df.to_csv(output_dir / "training_summary.csv", index=False)
    
    if failed_facts:
        failed_df = pd.DataFrame(failed_facts)
        failed_df.to_csv(output_dir / "failed_facts.csv", index=False)
    



    print(f"Training complete!")
    print(f"Successfully trained: {len(all_metadata)} new LoRAs")
    print(f"Skipped: {skipped_count} (already trained)")
    if failed_facts:
        print(f"✗ Failed: {len(failed_facts)} facts (see failed_facts.csv)")
    print(f"Total LoRAs in directory: {len(existing_facts) + len(all_metadata)}")
    print(f"Output directory: {output_dir}")



if __name__ == "__main__":
    main()