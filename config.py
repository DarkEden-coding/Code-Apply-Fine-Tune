from pathlib import Path

# ---------- paths ----------
DATA_DIR = Path("data")
REPOS_DIR = DATA_DIR / "repos"
SNIPPETS_FILE = DATA_DIR / "snippets.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
BENCHMARK_RESULTS = DATA_DIR / "benchmark_results.jsonl"

# ---------- github ----------
# set to None to use unauthenticated (60 req/hr)
# or export GITHUB_TOKEN=ghp_xxx
GITHUB_TOKEN = None
GITHUB_LANGUAGES = ["Python", "Go", "TypeScript", "Rust"]
MIN_STARS = 500
REPOS_PER_LANGUAGE = 5
MAX_REPOS = 20

# ---------- snippet extraction ----------
MIN_SNIPPET_LINES = 6
MAX_SNIPPET_LINES = 60
MAX_SNIPPETS_PER_FILE = 3
MAX_FILES_PER_REPO = 100
FILE_EXTENSIONS = {
    ".py",
    ".go",
    ".ts",
    ".tsx",
    ".rs",
    ".js",
    ".jsx",
}

# ---------- corruption ----------
CORRUPTION_STRATEGIES = [
    "rename_variable",
    "remove_parameter",
    "swap_attribute",
    "drop_trailing_char",
    "indent_shift",
    "omit_line",
    "combine",
]
CORRUPTIONS_PER_SNIPPET = 2
TRAIN_SPLIT = 0.9

# ---------- ollama ----------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "granite4:3b-h"

# ---------- fine-tuning ----------
BASE_MODEL = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
OUTPUT_DIR = DATA_DIR / "finetuned"
LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
MAX_SEQ_LEN = 4096
