"""
Benchmark a local Ollama model on the eval set.
Measures exact-match accuracy and a fuzzy (difflib) similarity score.

Outputs: data/benchmark_results.jsonl + summary to stdout.
"""

import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from config import (
    BENCHMARK_RESULTS,
    EVAL_FILE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


# Number of samples to submit together before waiting for completion.
BATCH_SIZE = 8
# Parallel workers used inside each batch.
MAX_WORKERS = 8


def query_ollama(system: str, prompt: str, model: str = OLLAMA_MODEL) -> dict:
    """Send a chat completion to the local Ollama instance."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
        },
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return {
        "content": data["message"]["content"],
        "total_duration_ns": data.get("total_duration", 0),
        "eval_count": data.get("eval_count", 0),
    }


def normalize(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    t = text.strip()
    # remove common markdown wrappers
    for lang in ("python", "go", "typescript", "rust", "js", "tsx", "jsx", ""):
        fence = f"```{lang}"
        if t.startswith(fence):
            t = t[len(fence) :]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def score_exact(predicted: str, expected: str) -> bool:
    return normalize(predicted) == normalize(expected)


def score_similarity(predicted: str, expected: str) -> float:
    a = normalize(predicted)
    b = normalize(expected)
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_whitespace_exact(predicted: str, expected: str) -> bool:
    """Check if content matches ignoring leading whitespace per line."""
    pred_lines = [l.strip() for l in normalize(predicted).splitlines()]
    exp_lines = [l.strip() for l in normalize(expected).splitlines()]
    return pred_lines == exp_lines


def run_sample(sample: dict) -> dict | None:
    """Run one benchmark sample against Ollama and compute metrics."""
    t0 = time.time()
    try:
        resp = query_ollama(sample["system"], sample["input"])
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None

    latency_ms = (time.time() - t0) * 1000
    predicted = resp["content"]
    expected = sample["output"]

    exact = score_exact(predicted, expected)
    ws_exact = score_whitespace_exact(predicted, expected)
    similarity = score_similarity(predicted, expected)

    meta = sample.get("metadata", {})
    strategy = meta.get("strategy", "unknown")
    language = meta.get("language", "unknown")

    return {
        "exact_match": exact,
        "whitespace_agnostic_match": ws_exact,
        "similarity": similarity,
        "latency_ms": latency_ms,
        "strategy": strategy,
        "language": language,
        "predicted": predicted,
        "expected": expected,
    }


def main():
    if not EVAL_FILE.exists():
        print(f"ERROR: {EVAL_FILE} not found. Run 2_generate_data.py first.")
        return

    samples = []
    with open(EVAL_FILE) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Benchmarking {len(samples)} samples against {OLLAMA_MODEL}...")
    print(f"Ollama endpoint: {OLLAMA_BASE_URL}\n")

    results = []
    stats = {
        "total": 0,
        "exact_match": 0,
        "whitespace_agnostic_match": 0,
        "similarity_sum": 0.0,
        "total_latency_ms": 0,
        "by_strategy": {},
        "by_language": {},
    }

    with tqdm(total=len(samples), desc="Running benchmark (batched)") as pbar:
        for start in range(0, len(samples), BATCH_SIZE):
            batch = samples[start : start + BATCH_SIZE]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(run_sample, sample) for sample in batch]

                for future in as_completed(futures):
                    pbar.update(1)
                    out = future.result()
                    if out is None:
                        continue

                    exact = out["exact_match"]
                    ws_exact = out["whitespace_agnostic_match"]
                    similarity = out["similarity"]
                    latency_ms = out["latency_ms"]
                    strategy = out["strategy"]
                    language = out["language"]

                    result = {
                        "exact_match": exact,
                        "whitespace_agnostic_match": ws_exact,
                        "similarity": round(similarity, 4),
                        "latency_ms": round(latency_ms, 1),
                        "strategy": strategy,
                        "language": language,
                        "predicted": out["predicted"][:500],
                        "expected": out["expected"][:500],
                    }
                    results.append(result)

                    # accumulate stats
                    stats["total"] += 1
                    stats["exact_match"] += int(exact)
                    stats["whitespace_agnostic_match"] += int(ws_exact)
                    stats["similarity_sum"] += similarity
                    stats["total_latency_ms"] += latency_ms

                    for key, val, field in [
                        ("by_strategy", strategy, strategy),
                        ("by_language", language, language),
                    ]:
                        bucket = stats[key].setdefault(
                            field,
                            {"total": 0, "exact": 0, "ws_exact": 0, "sim_sum": 0.0},
                        )
                        bucket["total"] += 1
                        bucket["exact"] += int(exact)
                        bucket["ws_exact"] += int(ws_exact)
                        bucket["sim_sum"] += similarity

    # --- write results ---
    with open(BENCHMARK_RESULTS, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # --- print summary ---
    n = stats["total"]
    if n == 0:
        print("No results.")
        return

    print("\n" + "=" * 60)
    print(f"MODEL: {OLLAMA_MODEL}")
    print(f"SAMPLES: {n}")
    print(
        f"EXACT MATCH:              {stats['exact_match']}/{n} "
        f"({stats['exact_match'] / n * 100:.1f}%)"
    )
    print(
        f"WHITESPACE-AGNOSTIC:      {stats['whitespace_agnostic_match']}/{n} "
        f"({stats['whitespace_agnostic_match'] / n * 100:.1f}%)"
    )
    print(f"AVG SIMILARITY:           {stats['similarity_sum'] / n:.4f}")
    print(f"AVG LATENCY:              {stats['total_latency_ms'] / n:.0f}ms")

    print("\n--- By Corruption Strategy ---")
    for strat, b in sorted(stats["by_strategy"].items()):
        t = b["total"]
        print(
            f"  {strat:25s}  exact={b['exact']:3d}/{t:<3d} "
            f"({b['exact'] / t * 100:5.1f}%)  "
            f"ws={b['ws_exact']:3d}/{t:<3d}  "
            f"sim={b['sim_sum'] / t:.3f}"
        )

    print("\n--- By Language ---")
    for lang, b in sorted(stats["by_language"].items()):
        t = b["total"]
        print(
            f"  {lang:15s}  exact={b['exact']:3d}/{t:<3d} "
            f"({b['exact'] / t * 100:5.1f}%)  "
            f"ws={b['ws_exact']:3d}/{t:<3d}  "
            f"sim={b['sim_sum'] / t:.3f}"
        )

    print("=" * 60)
    print(f"Detailed results -> {BENCHMARK_RESULTS}")


if __name__ == "__main__":
    main()
