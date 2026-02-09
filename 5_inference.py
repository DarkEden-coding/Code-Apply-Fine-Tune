"""
Run inference with the fine-tuned model.
Loads the LoRA adapter from a user-specified path and uses the same
ChatML-style format as training.

Example path: data/finetuned/checkpoint-200
"""

import sys
from pathlib import Path

import torch

from config import BASE_MODEL


# System prompt used during training
SYSTEM_PROMPT = """You are a code-edit reconciliation engine. A SEARCH/REPLACE operation failed because the SEARCH block does not match the source code. Identify the correct SEARCH block from the Current Code that the edit was intended to target.

Constraints:
1. Output ONLY the corrected SEARCH block.
2. The output must match the Current Code character-for-character (including whitespace and indentation).
3. Do not include the REPLACE block.
4. Do not include any headers, explanations, or markdown code fences."""


def format_chat(user_input: str) -> str:
    """
    Format into the same ChatML-style string used during training.
    Matches the format_chat function in 4_finetune.py.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
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


def main():
    ensure_unsloth_cache_on_path()

    # Unsloth should be imported before trl/transformers/peft
    import unsloth  # noqa: F401
    from unsloth import FastModel
    from peft import PeftModel

    # Get model path from user
    print("Enter the path to the fine-tuned model/checkpoint.")
    print("Example: data/finetuned/checkpoint-200")
    print()
    model_path = input("Model path: ").strip()

    if not model_path:
        print("No path provided, exiting.")
        return

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Path does not exist: {model_path}")
        return

    print(f"\nLoading base model: {BASE_MODEL}")
    print("This may take a moment...\n")

    # Load the base model first
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    print(f"Loading LoRA adapter from: {model_path}")

    # Load the LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(model, str(model_path))

    print("Model loaded successfully!")
    print("Enter code reconciliation requests (Ctrl+C to exit).\n")

    while True:
        try:
            # Get current code (multi-line input)
            print("=" * 60)
            print("Enter CURRENT CODE (press Enter twice to finish):")
            print("-" * 60)
            current_code_lines = []
            empty_count = 0
            while True:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                else:
                    empty_count = 0
                current_code_lines.append(line)
            current_code = "\n".join(current_code_lines).strip()

            if not current_code:
                print("No current code provided, skipping...\n")
                continue

            # Get search/replace block (multi-line input)
            print("-" * 60)
            print("Enter SEARCH/REPLACE BLOCK (press Enter twice to finish):")
            print("Example format:")
            print("<<<<<<< SEARCH")
            print("    original code")
            print("=======\n    new code")
            print(">>>>>>> REPLACE")
            print("-" * 60)
            search_replace_lines = []
            empty_count = 0
            while True:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                else:
                    empty_count = 0
                search_replace_lines.append(line)
            search_replace = "\n".join(search_replace_lines).strip()

            if not search_replace:
                print("No search/replace block provided, skipping...\n")
                continue

            # Format the input in the same format as training data
            user_input = (
                f"### Current Code:\n```\n{current_code}\n```\n\n"
                f"### Proposed Edit (Failed Match)\n```\n{search_replace}\n```"
            )

            # Format input using the same ChatML format as training
            prompt = format_chat(user_input)

            # Debug: ensure prompt is valid
            if prompt is None:
                print("Error: format_chat returned None")
                continue

            print("-" * 60)
            print("Generating corrected SEARCH block...\n")

            # Tokenize - Gemma3Processor requires explicit text=...
            # Passing positional args can be interpreted as images and leave text=None.
            inputs = tokenizer(text=[prompt], return_tensors="pt").to(model.device)

            if inputs is None or "input_ids" not in inputs:
                print("Error: Tokenization failed")
                continue

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            if outputs is None or len(outputs) == 0:
                print("Error: Model generation failed - no outputs")
                continue

            # Decode and extract only the assistant's response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's part (after the last <|im_start|>assistant)
            if "<|im_start|>assistant" in full_response:
                assistant_response = full_response.split("<|im_start|>assistant")[-1]
                # Remove any trailing <|im_end|> tag
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0]
            else:
                assistant_response = full_response

            print("CORRECTED SEARCH BLOCK:")
            print("-" * 60)
            print(assistant_response.strip())
            print("-" * 60)
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            import traceback

            print(f"Error: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
