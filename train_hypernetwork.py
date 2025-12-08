import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from pathlib import Path
import pandas as pd
import gc
import re
import random
from glob import glob

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

NUM_EXAMPLES = 5000
TRAIN_SPLIT = 0.8         # 80% train
RANDOM_SEED = 42

POSSIBLE_LORA_PATHS = [
    "/home/hice1/mdoutre3/scratch/qwen25_individual_loras",
    "~/scratch/qwen25_individual_loras",
    "./qwen25_individual_loras",
    "../qwen25_individual_loras",
]

DATA_PATH = "/home/hice1/mdoutre3/LLM_project_beta/wikifactdiff_converted.csv"

def find_lora_directory():
    for path_str in POSSIBLE_LORA_PATHS:
        path = Path(path_str).expanduser()
        if path.exists():
            print(f"Found LoRA directory: {path}")
            return path

    print("Searching for LoRA directory...")
    home = Path.home()
    for pattern in ["**/qwen25_individual_loras", "**/individual_loras"]:
        matches = list(home.glob(pattern))
        if matches:
            print(f"Found LoRA directory: {matches[0]}")
            return matches[0]
    
    raise FileNotFoundError("Could not find LoRA directory. Please update POSSIBLE_LORA_PATHS")

def discover_all_loras(lora_dir):
    print(f"\nScanning {lora_dir} for LoRA files...")

    fact_dirs = sorted(lora_dir.glob("fact_*"))
    
    if not fact_dirs:
        raise FileNotFoundError(f"No fact_* directories found in {lora_dir}")

    available_loras = []
    for fact_dir in fact_dirs:
        try:
            idx = int(fact_dir.name.split('_')[1])
            lora_file = fact_dir / "lora_params.pt"
            if lora_file.exists():
                available_loras.append((idx, fact_dir))
        except (ValueError, IndexError):
            print(f"Warning: Skipping invalid directory name: {fact_dir.name}")


    available_loras.sort(key=lambda x: x[0])

    print(f"Found {len(available_loras)} valid LoRA files")
    print(f"  Index range: {available_loras[0][0]} to {available_loras[-1][0]}")
    
    return available_loras

def load_data_flexible(lora_dir, df, num_examples, tokenizer, embedding_layer, device):
    available_loras = discover_all_loras(lora_dir)

    if num_examples == 'all':
        num_to_load = len(available_loras)
    else:
        num_to_load = min(num_examples, len(available_loras))
    
    print(f"\nLoading {num_to_load} examples out of {len(available_loras)} available...")
    
    lora_stats = []
    all_data = []
    
    for i, (fact_idx, fact_dir) in enumerate(available_loras[:num_to_load]):
        if i % 100 == 0 and i > 0:
            print(f"  Loaded {i}/{num_to_load}...")
        
        lora_file = fact_dir / "lora_params.pt"
        
        try:
            lora_params = torch.load(lora_file, map_location="cpu", weights_only=True)
            target_flat = torch.cat([p.flatten().float() for p in lora_params.values()])
            
            lora_stats.append({
                'std': target_flat.std().item(),
                'abs_mean': target_flat.abs().mean().item(),
            })

            if fact_idx < len(df):
                row = df.iloc[fact_idx]
            else:
                print(f"Warning: No CSV entry for fact_{fact_idx:04d}, skipping...")
                continue
            
            q_inputs = tokenizer(row['question'], return_tensors="pt", truncation=True, max_length=64).to(device)
            a_inputs = tokenizer(row['new_answer'], return_tensors="pt", truncation=True, max_length=64).to(device)
            
            with torch.no_grad():
                q_emb = embedding_layer(q_inputs["input_ids"]).mean(dim=1)
                a_emb = embedding_layer(a_inputs["input_ids"]).mean(dim=1)
            
            text_emb = torch.cat([q_emb, a_emb], dim=-1).squeeze(0).cpu()
            
            all_data.append({
                'text_emb': text_emb,
                'target': target_flat,
                'question': row['question'],
                'answer': row['new_answer'],
                'fact_idx': fact_idx,
            })
        
        except Exception as e:
            print(f"Warning: Error loading fact_{fact_idx:04d}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded successfully!")
    
    avg_std = sum(s['std'] for s in lora_stats) / len(lora_stats)
    avg_abs_mean = sum(s['abs_mean'] for s in lora_stats) / len(lora_stats)

    print(f"\nSuccessfully loaded {len(all_data)} examples")
    print(f"  Target LoRA stats: std={avg_std:.6f}, abs_mean={avg_abs_mean:.6f}")
    
    return all_data, avg_std, avg_abs_mean


print(f"Configuration:")
print(f"  Total examples: {NUM_EXAMPLES}")
print(f"  Train split: {int(NUM_EXAMPLES * TRAIN_SPLIT if NUM_EXAMPLES != 'all' else 'TBD')}")
print(f"  Val split: {int(NUM_EXAMPLES * (1-TRAIN_SPLIT) if NUM_EXAMPLES != 'all' else 'TBD')}")
print(f"  Random seed: {RANDOM_SEED}")

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

LORA_DIR = find_lora_directory()

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32,
    device_map="auto"
)
embedding_layer = model.get_input_embeddings()
device = model.device

print(f"\nLoading CSV from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"CSV has {len(df)} rows")

all_data, avg_std, avg_abs_mean = load_data_flexible(
    LORA_DIR, df, NUM_EXAMPLES, tokenizer, embedding_layer, device
)

random.shuffle(all_data)
split_idx = int(len(all_data) * TRAIN_SPLIT)
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

print(f"\nData split:")
print(f"  Train: {len(train_data)} examples")
print(f"  Val: {len(val_data)} examples")


def evaluate_base_model(base_model, tokenizer, data, max_examples=None):
    eval_data = data[:max_examples] if max_examples else data
    results = []
    
    for d in eval_data:
        prompt = f"Q: {d['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(base_model.device)

        with torch.no_grad():
            output = base_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted_answer = text.lower()
        true_answer = d['answer'].lower()

        answer_words = [w for w in true_answer.split() if len(w) > 3]
        correct = any(word in predicted_answer for word in answer_words)

        results.append(correct)

    base_accuracy = sum(results) / len(results) if results else 0
    return base_accuracy

print("\nEvaluating BASE MODEL (no LoRA)...")
base_acc_train = evaluate_base_model(model, tokenizer, train_data, max_examples=50)
base_acc_val = evaluate_base_model(model, tokenizer, val_data, max_examples=50)

print(f"Base model accuracy on TRAIN: {base_acc_train*100:.1f}%")
print(f"Base model accuracy on VAL:   {base_acc_val*100:.1f}%")

class ImprovedHypernetwork(nn.Module):
    def __init__(self, input_dim, output_dim, target_std=0.4):
        super().__init__()
        self.target_std = target_std
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, output_dim),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        out = self.net(x)
        out = torch.clamp(out, -1.5, 1.5)
        
        current_std = out.std()
        if current_std > 1e-6:
            scale_factor = self.target_std / current_std
            scale_factor = torch.clamp(scale_factor, 0.5, 2.0)
            out = out * scale_factor
        
        return out

input_dim = train_data[0]['text_emb'].shape[0]
output_dim = train_data[0]['target'].shape[0]

hypernetwork = ImprovedHypernetwork(input_dim, output_dim, target_std=avg_std).to(device)
print(f"\nHypernetwork: {sum(p.numel() for p in hypernetwork.parameters()):,} params")

optimizer = optim.AdamW(
    hypernetwork.parameters(), 
    lr=2e-4,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=15,
    verbose=True
)

fact_dir = LORA_DIR / "fact_0000"
lora_params = torch.load(fact_dir / "lora_params.pt", map_location="cpu", weights_only=True)
lora_shapes = {name: param.shape for name, param in lora_params.items()}

def reshape_to_lora(flat_tensor, shapes):
    lora_dict = {}
    offset = 0
    for name, shape in shapes.items():
        size = shape[0] * shape[1] if len(shape) == 2 else shape[0]
        lora_dict[name] = flat_tensor[offset:offset+size].reshape(shape)
        offset += size
    return lora_dict

def is_gibberish(text):
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.3:
        return True
    if re.search(r'(\b\w+\b)\s+\1\s+\1', text):
        return True
    special = sum(1 for c in text if c in '_-.,;:()[]{}')
    if special / max(len(text), 1) > 0.4:
        return True
    return False

def test_functional(hypernetwork, data_sample, base_model, tokenizer, lora_shapes, device):
    with torch.no_grad():
        text_emb = data_sample['text_emb'].unsqueeze(0).to(device)
        generated_flat = hypernetwork(text_emb)[0]
    
    lora_dict = reshape_to_lora(generated_flat, lora_shapes)
    
    has_nan = any(torch.isnan(p).any() for p in lora_dict.values())
    has_inf = any(torch.isinf(p).any() for p in lora_dict.values())
    
    if has_nan or has_inf:
        return False, False, "ERROR", "ERROR", True
    
    lora_config = LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", inference_mode=True
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    state_dict = peft_model.state_dict()
    for name, param in lora_dict.items():
        if name in state_dict:
            state_dict[name].copy_(param.to(device))
    
    prompt = f"Q: {data_sample['question']}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with peft_model.disable_adapter():
            base_out = peft_model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        lora_out = peft_model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    
    base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
    lora_text = tokenizer.decode(lora_out[0], skip_special_tokens=True)
    
    del peft_model
    torch.cuda.empty_cache()
    gc.collect()
    
    lora_gibberish = is_gibberish(lora_text)
    changed = base_text != lora_text
    
    answer_lower = data_sample['answer'].lower()
    lora_lower = lora_text.lower()
    answer_words = [w for w in answer_lower.split() if len(w) > 3]
    correct = any(word in lora_lower for word in answer_words) if not lora_gibberish else False
    
    return changed, correct, base_text, lora_text, lora_gibberish

def evaluate_set(hypernetwork, data, base_model, tokenizer, lora_shapes, device, max_examples=None):
    hypernetwork.eval()
    eval_data = data[:max_examples] if max_examples else data
    results = []
    
    for d in eval_data:
        changed, correct, base, lora, gib = test_functional(hypernetwork, d, base_model, tokenizer, lora_shapes, device)
        results.append({'changed': changed, 'correct': correct, 'gib': gib})
    
    change_rate = sum(r['changed'] for r in results) / len(results) if results else 0
    accuracy = sum(r['correct'] for r in results) / len(results) if results else 0
    gib_rate = sum(r['gib'] for r in results) / len(results) if results else 0
    
    return change_rate, accuracy, gib_rate

best_val_acc = 0
patience_counter = 0
max_patience = 40

for epoch in range(150):
    hypernetwork.train()
    epoch_loss = 0
    
    for d in train_data:
        text_emb = d['text_emb'].unsqueeze(0).to(device)
        target = d['target'].unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        pred = hypernetwork(text_emb)
        
        mse_loss = nn.functional.mse_loss(pred, target)
        pred_std = pred.std()
        target_std = target.std()
        std_loss = (pred_std - target_std) ** 2
        extreme_loss = torch.mean(torch.abs(pred[torch.abs(pred) > 1.5])) if (torch.abs(pred) > 1.5).any() else torch.tensor(0.0, device=device)
        
        loss = mse_loss + 0.1 * std_loss + 0.5 * extreme_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), 0.5)
        optimizer.step()
        
        epoch_loss += mse_loss.item()
    
    avg_train_loss = epoch_loss / len(train_data)
    scheduler.step(avg_train_loss)
    
    if epoch % 20 == 0 or epoch == 299:
        print(f"\nEpoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        val_change, val_acc, val_gib = evaluate_set(hypernetwork, val_data, model, tokenizer, lora_shapes, device, max_examples=min(10, len(val_data)))
        train_change, train_acc, train_gib = evaluate_set(hypernetwork, train_data, model, tokenizer, lora_shapes, device, max_examples=10)
        
        print(f"TRAIN:  Change={train_change*100:.0f}% | Acc={train_acc*100:.0f}% | Gib={train_gib*100:.0f}%")
        print(f"VAL:    Change={val_change*100:.0f}% | Acc={val_acc*100:.0f}% | Gib={val_gib*100:.0f}%")
        
        if train_acc - val_acc > 0.3:
            print(f"Overfitting: gap = {(train_acc - val_acc)*100:.0f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Best val acc: {val_acc*100:.0f}%")
            torch.save({'epoch': epoch, 'model_state_dict': hypernetwork.state_dict(),
                       'val_acc': val_acc}, 'best_hypernetwork.pt')
        else:
            patience_counter += 1

        if val_acc >= 0.5 and val_gib == 0:
            print(f"\nSUCCESS! Val acc={val_acc*100:.0f}%")
            break
        
        if patience_counter >= max_patience:
            print(f"\nEarly stopping")
            break

print(f"Best validation accuracy: {best_val_acc*100:.0f}%")