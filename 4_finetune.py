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
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

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


def main():
    print(f"Base model:  {BASE_MODEL}")
    print(f"Train file:  {TRAIN_FILE}")
    print(f"Eval file:   {EVAL_FILE}")
    print(f"Output dir:  {OUTPUT_DIR}\n")

    # --- load model ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
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
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=True,
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
            max_seq_length=MAX_SEQ_LEN,
            packing=True,
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
