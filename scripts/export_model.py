"""
Export fine-tuned model for LM Studio.

Steps:
1. Merge LoRA adapter with base model
2. Save merged model in HuggingFace format
3. Convert to GGUF using llama.cpp (manual step)

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --adapter models/race-engineer-llama-merged --output models/merged
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def merge_and_export(adapter_path: str, output_path: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Merge LoRA adapter with base model and save."""
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {adapter_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model in full precision for merging
    print("Loading base model in FP16 (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge adapter into base model
    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    print(f"\nMerged model saved to: {output_path}")
    print("\nNext steps to use in LM Studio:")
    print("-" * 40)
    print("""
1. Install llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   cmake -B build
   cmake --build build --config Release

2. Convert to GGUF:
   python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile race-engineer.gguf

3. Quantize (optional, reduces size):
   ./llama.cpp/build/bin/llama-quantize race-engineer.gguf race-engineer-q4_k_m.gguf q4_k_m

4. Load in LM Studio:
   - Open LM Studio
   - Go to My Models > Import
   - Select the .gguf file
   - Load and use!
""".format(output_path=output_path))


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model for LM Studio")
    parser.add_argument("--adapter", type=str, default="models/race-engineer-llama-merged",
                        help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, default="models/race-engineer-merged-full",
                        help="Output path for merged model")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name")

    args = parser.parse_args()

    if not Path(args.adapter).exists():
        print(f"Error: Adapter not found at {args.adapter}")
        return

    merge_and_export(args.adapter, args.output, args.base_model)


if __name__ == "__main__":
    main()