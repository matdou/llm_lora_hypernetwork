# LoRA Hypernetwork for Fact Learning

This project trains a hypernetwork to generate LoRA adapters for teaching new facts to language models.

## What it does

Instead of manually training a LoRA for each new fact, the hypernetwork learns to predict what LoRA weights are needed. You give it a fact, and it outputs the adapter weights.

## Files

- `train_lora_adapters.py` - Creates individual LoRA adapters for each fact in the dataset
- `train_hypernetwork.py` - Trains the hypernetwork to predict LoRA weights
- `test_hypernetwork.py` - Tests the trained hypernetwork
- `evaluate_real_loras.py` - Evaluates the individually trained LoRAs
- `check_base_knowledge.py` - Checks if the base model already knows certain facts

## Data

Uses `wikifactdiff_converted.csv` which contains questions and updated answers for fact changes.

## Usage

First, generate the training data (individual LoRAs):
```bash
python train_lora_adapters.py --num_facts 100
```

Then train the hypernetwork:
```bash
python train_hypernetwork.py
```

Test it:
```bash
python test_hypernetwork.py
```

## Model

Base model: Qwen2-7B-Instruct
LoRA config: rank=2, alpha=4, targets=q_proj,v_proj
