import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from pathlib import Path
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
LORA_DIR = Path("/home/hice1/mdoutre3/scratch/qwen25_individual_loras")
DATA_PATH = "/home/hice1/mdoutre3/LLM_project_beta/wikifactdiff_converted.csv"

# LoRA config ( match training)
LORA_R = 2
LORA_ALPHA = 4
TARGET_MODULES = ["q_proj"]


class SimpleHyperNetwork(nn.Module):
    """
    Simple hypernetwork that takes (question, answer) and generates LoRA weights
    
    Architecture:
    - Input: Concatenated embeddings of question + answer
    - Hidden layers: MLP with LayerNorm and Dropout
    - Output: LoRA weights (A and B matrices for each layer)
    """
    def __init__(self, input_dim, hidden_dim, lora_shapes):
        super().__init__()
        self.lora_shapes = lora_shapes  # Dict of {param_name: shape}
        
        total_params = sum(shape[0] * shape[1] for shape in lora_shapes.values())
        
        # MLP to generate LoRA weights -> match training architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, total_params),
        )
        
    def forward(self, x):
        """
        x: [batch_size, input_dim] - concatenated question + answer embeddings
        returns: dict of {param_name: tensor} - generated LoRA weights
        """
        flat_weights = self.network(x)  # [batch_size, total_params]
        
        # Reshape into LoRA parameter dict
        lora_params = {}
        offset = 0
        for name, shape in self.lora_shapes.items():
            size = shape[0] * shape[1]
            param = flat_weights[:, offset:offset+size].reshape(-1, *shape)
            lora_params[name] = param[0]  # Remove batch dim
            offset += size
        
        return lora_params


def load_base_model_and_tokenizer():
    """Load the base model and tokenizer"""
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    return model, tokenizer


def get_lora_shapes_from_saved(fact_idx=0):
    """Get LoRA parameter shapes from a saved adapter"""
    lora_params = torch.load(LORA_DIR / f"fact_{fact_idx:04d}" / "lora_params.pt", 
                            weights_only=True, map_location="cpu")
    return {name: tuple(param.shape) for name, param in lora_params.items()}


def embed_text(text, tokenizer, model):
    """Get embedding for text by averaging token embeddings"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        avg_embedding = embeddings.mean(dim=1)  # [1, hidden_dim]
    
    return avg_embedding


def test_hypernetwork(hypernetwork, base_model, tokenizer, test_facts):
    """
    Test the hypernetwork on generating LoRAs for new facts
    
    Args:
        hypernetwork: Trained hypernetwork model
        base_model: Base LLM
        tokenizer: Tokenizer
        test_facts: List of (question, answer) tuples to test
    
    Returns:
        Dictionary with test results
    """
    hypernetwork.eval()
    results = []
    
    print("\nTesting hypernetwork on generating LoRAs...")
    for question, answer in tqdm(test_facts, desc="Testing"):
        # Generate LoRA weights using hypernetwork
        q_embed = embed_text(question, tokenizer, base_model)
        a_embed = embed_text(answer, tokenizer, base_model)
        input_embed = torch.cat([q_embed, a_embed], dim=-1)
        
        with torch.no_grad():
            generated_lora_params = hypernetwork(input_embed)
        
        # Apply generated LoRA to base model
        lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],  # Match training config
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        lora_model = get_peft_model(base_model, lora_config)
        
        # Load generated weights
        lora_state_dict = lora_model.state_dict()
        for name, param in generated_lora_params.items():
            if name in lora_state_dict:
                lora_state_dict[name].copy_(param.to(base_model.device))
        
        lora_model.eval()
        
        # 3. Test if model outputs the correct answer
        prompt = f"As of today, {question.lower()}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        
        # Base model answer
        with torch.no_grad():
            base_outputs = base_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        base_answer = base_text.split("Answer:")[-1].strip().split("\n")[0].strip()
        
        # LoRA model answer
        with torch.no_grad():
            lora_outputs = lora_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        lora_text = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        lora_answer = lora_text.split("Answer:")[-1].strip().split("\n")[0].strip()
        
        # Check if correct ...
        correct = answer.lower() in lora_answer.lower()
        changed = base_answer.lower() != lora_answer.lower()
        
        results.append({
            "question": question,
            "target_answer": answer,
            "base_answer": base_answer,
            "lora_answer": lora_answer,
            "correct": correct,
            "changed": changed,
        })
        
        # Clean up
        del lora_model
        torch.cuda.empty_cache()
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained hypernetwork checkpoint")
    parser.add_argument("--num_test", type=int, default=5,
                       help="Number of facts to test")
    args = parser.parse_args()
    
    base_model, tokenizer = load_base_model_and_tokenizer()
    
    lora_shapes = get_lora_shapes_from_saved()
    print(f"\nLoRA parameter shapes:")
    for name, shape in lora_shapes.items():
        print(f"  {name}: {shape}")
    
    # Get model hidden dimension
    hidden_dim = base_model.config.hidden_size
    input_dim = hidden_dim * 2  # Concatenated question + answer embeddings
    
    print(f"\nModel hidden dimension: {hidden_dim}")
    print(f"Hypernetwork input dimension: {input_dim}")
    
    if args.checkpoint:
        print(f"\nLoading trained hypernetwork from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=base_model.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        checkpoint_hidden_dim = config.get('hidden_dim', 1024)
        checkpoint_lora_shapes = {k: tuple(v) for k, v in config.get('lora_shapes', {}).items()}
        
        if checkpoint_lora_shapes:
            lora_shapes = checkpoint_lora_shapes
            print(f"  Using LoRA shapes from checkpoint")
        
        hypernetwork = SimpleHyperNetwork(
            input_dim=input_dim,
            hidden_dim=checkpoint_hidden_dim,
            lora_shapes=lora_shapes
        ).to(base_model.device)
        
        hypernetwork.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded trained hypernetwork")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
    else:
        hypernetwork = SimpleHyperNetwork(
            input_dim=input_dim,
            hidden_dim=1024,
            lora_shapes=lora_shapes
        ).to(base_model.device)
        print("\n No checkpoint provided - using RANDOM initialization")
        print("   Use --checkpoint path/to/best_hypernetwork.pt to load trained model")
    
    print(f"\nHypernetwork parameters: {sum(p.numel() for p in hypernetwork.parameters()):,}")
    
    # Load test data
    df = pd.read_csv(DATA_PATH)
    
    # Test on first N facts
    test_facts = [
        (df.iloc[i]["question"], df.iloc[i]["new_answer"]) 
        for i in range(args.num_test)
    ]
    
    if not args.checkpoint:
        print("NOTE: This hypernetwork is RANDOMLY INITIALIZED")
        print("To test a trained model:")
        print("  python test_hypernetwork.py --checkpoint path/to/best_hypernetwork.pt")
    
    # Test the hypernetwork
    results = test_hypernetwork(hypernetwork, base_model, tokenizer, test_facts)
    
    # Print results
    print("TEST RESULTS")
    
    for i, result in enumerate(results):
        print(f"\n--- Test {i+1} ---")
        print(f"Q: {result['question']}")
        print(f"Target: {result['target_answer']}")
        print(f"Base:   {result['base_answer']}")
        print(f"HyperLoRA: {result['lora_answer']}")
        print(f"Changed: {result['changed']} | Correct: {result['correct']}")
    
    # Summary
    correct_count = sum(r["correct"] for r in results)
    changed_count = sum(r["changed"] for r in results)
    
    print("SUMMARY")
    print(f"Total tested: {len(results)}")
    print(f"Changed output: {changed_count}/{len(results)} ({100*changed_count/len(results):.0f}%)")
    print(f"Correct answer: {correct_count}/{len(results)} ({100*correct_count/len(results):.0f}%)")
    if args.checkpoint:
        print("✓ Testing TRAINED hypernetwork")
        print(f"  Actual: {100*correct_count/len(results):.0f}%")
    else:
        print("  No network found, testing RANDOM hypernetwork")
        print("  Train with: python train_hypernetwork.py")


if __name__ == "__main__":
    main()