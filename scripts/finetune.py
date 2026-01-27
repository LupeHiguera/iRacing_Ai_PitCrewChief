"""
QLoRA fine-tuning script for race engineer model.

Hardware target:
- 96GB DDR5 RAM
- RTX 5080 (16GB VRAM) + RTX 5060 Ti (16GB VRAM)

Uses 4-bit quantization + LoRA adapters to fit in VRAM.
Can run on single GPU or both GPUs via accelerate.

Usage:
    # Single GPU (recommended for small datasets)
    python scripts/finetune.py

    # Multi-GPU with accelerate
    accelerate launch scripts/finetune.py --multi-gpu

    # Custom settings
    python scripts/finetune.py --epochs 5 --lr 1e-4 --rank 32
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",

    # Paths
    "data_path": "data/synthetic/train.json",
    "output_dir": "models/race-engineer-lora",

    # LoRA settings
    "lora_r": 16,              # Rank - higher = more capacity, more VRAM
    "lora_alpha": 32,          # Scaling factor - typically 2x rank
    "lora_dropout": 0.05,      # Dropout for regularization
    "target_modules": [        # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP (optional, more capacity)
    ],

    # Training settings
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "warmup_ratio": 0.03,
    "max_seq_length": 512,

    # Hardware optimization
    "fp16": True,              # Use FP16 mixed precision
    "bf16": False,             # Use BF16 if available (better for newer GPUs)
    "gradient_checkpointing": True,  # Save VRAM at cost of speed

    # Logging
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
}


# =============================================================================
# DATA FORMATTING
# =============================================================================

def format_for_llama(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Format example for Llama 3.1 Instruct chat template.

    Llama 3.1 uses special tokens:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {assistant}<|eot_id|>
    """
    instruction = example["instruction"]
    input_data = example["input"]
    output = example["output"]

    # Build the chat format
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_data}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

    return {"text": text}


def load_and_prepare_data(data_path: str, tokenizer, max_length: int) -> Dataset:
    """Load JSON data and prepare for training."""
    print(f"Loading data from {data_path}")

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} examples")

    # Convert to dataset
    dataset = Dataset.from_list(raw_data)

    # Format for Llama
    dataset = dataset.map(
        lambda x: format_for_llama(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Set labels (for causal LM, labels = input_ids)
    dataset = dataset.map(lambda x: {"labels": x["input_ids"].copy()})

    print(f"Prepared {len(dataset)} training examples")
    return dataset


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Load model with 4-bit quantization and setup LoRA."""

    print(f"Loading model: {config['model_name']}")

    # Quantization config for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Normal Float 4 - better than FP4
        bnb_4bit_compute_dtype=torch.float16, # Compute in FP16
        bnb_4bit_use_double_quant=True,       # Nested quantization for more savings
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPUs
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train(config: Dict[str, Any]):
    """Main training function."""

    print("=" * 60)
    print("Race Engineer Fine-Tuning")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("WARNING: No CUDA GPUs available, training will be slow!")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load and prepare data
    train_dataset = load_and_prepare_data(
        config["data_path"],
        tokenizer,
        config["max_seq_length"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=3,
        fp16=config["fp16"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard for simplicity
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train!
    print("\nStarting training...")
    trainer.train()

    # Save the final adapter
    print(f"\nSaving adapter to {config['output_dir']}")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # Save training config
    config_path = Path(config["output_dir"]) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nTraining complete!")
    print(f"Adapter saved to: {config['output_dir']}")
    print("\nNext steps:")
    print("  1. Test with: python scripts/test_model.py")
    print("  2. Merge and export: python scripts/export_model.py")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune race engineer model")

    # Model settings
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Base model to fine-tune")
    parser.add_argument("--data", type=str, default=DEFAULT_CONFIG["data_path"],
                        help="Path to training data JSON")
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Output directory for adapter")

    # LoRA settings
    parser.add_argument("--rank", type=int, default=DEFAULT_CONFIG["lora_r"],
                        help="LoRA rank (higher = more capacity)")
    parser.add_argument("--alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"],
                        help="LoRA alpha (scaling factor)")

    # Training settings
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["per_device_batch_size"],
                        help="Per-device batch size")
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_seq_length"],
                        help="Maximum sequence length")

    # Hardware settings
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 instead of FP16 (better for RTX 40/50 series)")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing (faster but more VRAM)")

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update({
        "model_name": args.model,
        "data_path": args.data,
        "output_dir": args.output,
        "lora_r": args.rank,
        "lora_alpha": args.alpha,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_batch_size": args.batch_size,
        "max_seq_length": args.max_length,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
    })

    # RTX 50 series supports BF16 well
    if args.bf16:
        config["bf16"] = True
        config["fp16"] = False

    # Ensure output directory exists
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Train!
    train(config)


if __name__ == "__main__":
    main()
