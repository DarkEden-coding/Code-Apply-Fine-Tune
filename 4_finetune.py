"""
Fine-tune a model using QLoRA via Unsloth.
Reads: data/train.jsonl, data/eval.jsonl
Outputs: data/finetuned/ (LoRA adapter + optional GGUF for Ollama)

Supports any Unsloth-compatible base model. Change BASE_MODEL in config.py.
Common options:
  - unsloth/gemma-2-2b-it-bnb-4bit
  - unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit
  - unsloth/Llama-3.2-3B-Instruct-bnb-4bit
  - unsloth/Phi-4-mini-instruct-bnb-4bit
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset

from config import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    EVAL_FILE,
    GRAD_ACCUM,
    LORA_ALPHA,
    LORA_R,
    LR,
    MAX_SEQ_LEN,
    OUTPUT_DIR,
    TRAIN_FILE,
)


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_chat(sample: dict) -> str:
    """
    Format into a single training string.
    Uses ChatML-style tags that work across most instruct models.
    """
    return (
        f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )


def ensure_cuda_ready() -> None:
    """
    Fail fast with a clear message when PyTorch cannot access CUDA.
    Unsloth requires a CUDA-capable torch build + visible GPU.
    """
    if torch.cuda.is_available():
        return

    hints = [
        f"torch.__version__={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
    ]

    if "+cpu" in torch.__version__ or torch.version.cuda is None:
        hints.append(
            "Detected a CPU-only PyTorch build. Install a CUDA build of torch first "
            "(example: uv pip install --upgrade torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128)."
        )
    else:
        hints.append(
            "CUDA build is installed but no GPU is visible to PyTorch. "
            "Check NVIDIA driver, CUDA runtime compatibility, and that this process is not GPU-restricted."
        )

    raise RuntimeError(
        "CUDA is not available, so Unsloth cannot run.\n"
        + "\n".join(f"- {h}" for h in hints)
    )


def ensure_unsloth_cache_on_path() -> None:
    """
    Ensure Windows subprocess workers can import Unsloth's compiled trainer module.
    """
    cache_dir = Path(__file__).resolve().parent / "unsloth_compiled_cache"
    if not cache_dir.exists():
        return

    cache_str = str(cache_dir)
    if cache_str not in sys.path:
        sys.path.insert(0, cache_str)

    existing = os.environ.get("PYTHONPATH", "")
    if cache_str not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            cache_str if not existing else f"{cache_str}{os.pathsep}{existing}"
        )


def main():
    ensure_cuda_ready()
    ensure_unsloth_cache_on_path()

    # Avoid TorchInductor compile instability on some Windows + CUDA stacks.
    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
    except Exception:
        pass

    # Unsloth should be imported before trl/transformers/peft.
    import unsloth  # noqa: F401
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, FastModel

    use_bf16 = torch.cuda.is_bf16_supported()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    effective_max_seq_len = MAX_SEQ_LEN
    effective_batch_size = BATCH_SIZE
    if gpu_mem_gb <= 8:
        # 8GB cards are very tight for Gemma-3 4B QLoRA at 4k context.
        effective_max_seq_len = min(MAX_SEQ_LEN, 2048)
        effective_batch_size = min(BATCH_SIZE, 1)

    print(f"Base model:  {BASE_MODEL}")
    print(f"Train file:  {TRAIN_FILE}")
    print(f"Eval file:   {EVAL_FILE}")
    print(f"Output dir:  {OUTPUT_DIR}\n")
    print(f"GPU memory:  {gpu_mem_gb:.2f} GB")
    print(f"Batch size:  {effective_batch_size}")
    print(f"Max seq len: {effective_max_seq_len}\n")

    # --- load model ---
    if "gemma-3" in BASE_MODEL.lower():
        model, tokenizer = FastModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=effective_max_seq_len,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )

        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            random_state=42,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=effective_max_seq_len,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    # --- load data ---
    train_raw = load_jsonl(TRAIN_FILE)
    eval_raw = load_jsonl(EVAL_FILE)

    train_ds = Dataset.from_list([{"text": format_chat(s)} for s in train_raw])
    eval_ds = Dataset.from_list([{"text": format_chat(s)} for s in eval_raw])

    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples:  {len(eval_ds)}")

    # --- train ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=effective_batch_size,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=effective_max_seq_len,
            packing=True,
            # Windows + Unsloth compiled trainer can fail during multiprocessing
            # tokenization with `ModuleNotFoundError: UnslothSFTTrainer`.
            # Force single-process dataset preprocessing for stability.
            dataset_num_proc=1,
            dataloader_num_workers=0,
        ),
    )

    print("\nStarting training...")
    trainer.train()

    # --- save ---
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # --- export GGUF for Ollama ---
    gguf_dir = OUTPUT_DIR / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    print(f"Exporting GGUF (Q4_K_M) to {gguf_dir}...")

    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print("\nGGUF exported. To use with Ollama:")
        print("  1. Create a Modelfile:")
        print(f'     echo "FROM {gguf_dir}/unsloth.Q4_K_M.gguf" > Modelfile')
        print("  2. ollama create codepatch -f Modelfile")
        print("  3. ollama run codepatch")
    except Exception as e:
        print(f"GGUF export failed (non-fatal): {e}")
        print("You can still use the LoRA adapter directly with transformers.")

    print("\nDone.")


if __name__ == "__main__":
    main()
