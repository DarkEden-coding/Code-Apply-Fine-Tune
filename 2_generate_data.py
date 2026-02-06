"""
Take extracted snippets and generate malformed SEARCH/REPLACE pairs.
Outputs: data/train.jsonl, data/eval.jsonl

Each sample:
  - input:  system prompt + current code + malformed SEARCH/REPLACE block
  - output: the correct SEARCH block (verbatim from source)
"""

import json
import random
import re

from tqdm import tqdm

from config import (
    CORRUPTIONS_PER_SNIPPET,
    EVAL_FILE,
    SNIPPETS_FILE,
    TRAIN_FILE,
    TRAIN_SPLIT,
)


# ------------------------------------------------------------------ #
#                       corruption strategies                         #
# ------------------------------------------------------------------ #


def _random_rename(name: str) -> str:
    """Shorten or abbreviate a variable name."""
    if len(name) <= 2:
        return name + "_x"
    strategies = [
        lambda n: n[:3],
        lambda n: n[0] + n[-1],
        lambda n: n.replace("_", ""),
        lambda n: n[:1].upper() + n[1:],
        lambda n: n + "2",
    ]
    return random.choice(strategies)(name)


def corrupt_rename_variable(lines: list[str]) -> list[str]:
    """Pick a local variable/identifier and rename it."""
    ident_re = re.compile(r"\b([a-z_][a-z0-9_]{2,})\b")
    all_idents = set()
    for line in lines:
        all_idents.update(ident_re.findall(line))

    # filter out keywords
    keywords = {
        "self",
        "def",
        "return",
        "import",
        "from",
        "class",
        "func",
        "const",
        "let",
        "var",
        "for",
        "while",
        "if",
        "else",
        "elif",
        "true",
        "false",
        "none",
        "nil",
        "null",
        "async",
        "await",
        "pub",
        "mut",
        "err",
        "ctx",
    }
    candidates = [i for i in all_idents if i.lower() not in keywords]
    if not candidates:
        return lines

    target = random.choice(candidates)
    replacement = _random_rename(target)
    return [line.replace(target, replacement) for line in lines]


def corrupt_remove_parameter(lines: list[str]) -> list[str]:
    """Remove a random parameter from the first function signature."""
    out = []
    done = False
    for line in lines:
        if not done and ("def " in line or "func " in line or "fn " in line):
            paren_start = line.find("(")
            paren_end = line.rfind(")")
            if paren_start != -1 and paren_end != -1:
                params_str = line[paren_start + 1 : paren_end]
                params = [p.strip() for p in params_str.split(",") if p.strip()]
                if len(params) > 1:
                    # don't remove self/ctx
                    removable = [
                        i
                        for i, p in enumerate(params)
                        if not p.startswith("self")
                        and not p.startswith("ctx")
                        and not p.startswith("&self")
                    ]
                    if removable:
                        idx = random.choice(removable)
                        params.pop(idx)
                        new_params = ", ".join(params)
                        line = line[: paren_start + 1] + new_params + line[paren_end:]
                done = True
        out.append(line)
    return out


def corrupt_swap_attribute(lines: list[str]) -> list[str]:
    """Replace self.X with self.Y (a plausible but wrong name)."""
    attr_re = re.compile(r"self\.([a-z_][a-z0-9_]*)")
    all_attrs = set()
    for line in lines:
        all_attrs.update(attr_re.findall(line))

    if not all_attrs:
        return lines

    target = random.choice(list(all_attrs))
    swaps = {
        "db": "conn",
        "conn": "db",
        "timeout": "ttl",
        "ttl": "timeout",
        "writer": "stream",
        "stream": "writer",
        "config": "cfg",
        "cfg": "config",
        "logger": "log",
        "log": "logger",
        "sessions": "session",
        "session": "sessions",
        "count": "counter",
        "counter": "count",
        "active": "enabled",
        "enabled": "active",
        "lvl": "level",
        "level": "lvl",
        "name": "label",
        "label": "name",
    }
    replacement = swaps.get(target, target + "_ref")
    return [line.replace(f"self.{target}", f"self.{replacement}") for line in lines]


def corrupt_drop_trailing_char(lines: list[str]) -> list[str]:
    """Drop a trailing colon, bracket, or paren from a random line."""
    candidates = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped and stripped[-1] in (":", "{", "(", ")", "}", ";"):
            candidates.append(i)

    if not candidates:
        return lines

    idx = random.choice(candidates)
    out = lines.copy()
    out[idx] = out[idx].rstrip().rstrip(":{}();") + "\n"
    return out


def corrupt_indent_shift(lines: list[str]) -> list[str]:
    """Shift all lines by +/- 1 indent level."""
    direction = random.choice(["add", "remove"])
    out = []
    for line in lines:
        if direction == "add":
            out.append("    " + line)
        else:
            if line.startswith("    "):
                out.append(line[4:])
            elif line.startswith("\t"):
                out.append(line[1:])
            else:
                out.append(line)
    return out


def corrupt_omit_line(lines: list[str]) -> list[str]:
    """Remove a random non-first, non-last line."""
    if len(lines) <= 3:
        return lines
    idx = random.randint(1, len(lines) - 2)
    return lines[:idx] + lines[idx + 1 :]


def corrupt_combine(lines: list[str]) -> list[str]:
    """Apply 2-3 random corruptions in sequence."""
    single_strategies = [
        corrupt_rename_variable,
        corrupt_remove_parameter,
        corrupt_swap_attribute,
        corrupt_drop_trailing_char,
        corrupt_omit_line,
    ]
    n = random.randint(2, 3)
    chosen = random.sample(single_strategies, min(n, len(single_strategies)))
    for fn in chosen:
        lines = fn(lines)
    return lines


STRATEGY_MAP = {
    "rename_variable": corrupt_rename_variable,
    "remove_parameter": corrupt_remove_parameter,
    "swap_attribute": corrupt_swap_attribute,
    "drop_trailing_char": corrupt_drop_trailing_char,
    "indent_shift": corrupt_indent_shift,
    "omit_line": corrupt_omit_line,
    "combine": corrupt_combine,
}


# ------------------------------------------------------------------ #
#                       replacement generation                        #
# ------------------------------------------------------------------ #


def make_replacement(original_code: str) -> str:
    """
    Create a plausible 'replacement' block. This simulates what an
    LLM would *want* to change the code into. We make small additive
    edits: add a log line, add a docstring, rename for clarity, etc.
    """
    lines = original_code.splitlines(keepends=True)

    # randomly pick 1-2 small additive changes
    options = ["add_log", "add_comment", "add_variable"]
    chosen = random.sample(options, min(2, len(options)))

    result = lines.copy()

    if "add_log" in chosen:
        # insert a log/print statement after the first line
        if len(result) > 1:
            indent = len(result[1]) - len(result[1].lstrip())
            space = " " * indent
            log_line = f'{space}print("DEBUG: entered block")\n'
            result.insert(1, log_line)

    if "add_comment" in chosen and len(result) > 0:
        indent = len(result[0]) - len(result[0].lstrip())
        space = " " * indent
        comment = f"{space}# TODO: add error handling\n"
        result.insert(0, comment)

    if "add_variable" in chosen and len(result) > 1:
        indent = len(result[1]) - len(result[1].lstrip())
        space = " " * indent
        var_line = f"{space}_start = __import__('time').time()\n"
        result.insert(1, var_line)

    return "".join(result)


# ------------------------------------------------------------------ #
#                         prompt formatting                           #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = (
    "You are a code-edit reconciliation engine. "
    "A SEARCH/REPLACE operation failed because the SEARCH block does not "
    "match the source code. Identify the correct SEARCH block from the "
    "Current Code that the edit was intended to target.\n\n"
    "Constraints:\n"
    "1. Output ONLY the corrected SEARCH block.\n"
    "2. The output must match the Current Code character-for-character "
    "(including whitespace and indentation).\n"
    "3. Do not include the REPLACE block.\n"
    "4. Do not include any headers, explanations, or markdown code fences."
)


def format_input(
    file_path: str, current_code: str, malformed_search: str, replacement: str
) -> str:
    return (
        f"### Current Code: `{file_path}`\n"
        f"```\n{current_code}```\n\n"
        f"### Proposed Edit (Failed Match)\n"
        f"```\n"
        f"<<<<<<< SEARCH\n{malformed_search}"
        f"=======\n{replacement}"
        f">>>>>>> REPLACE\n"
        f"```"
    )


# ------------------------------------------------------------------ #
#                              main                                   #
# ------------------------------------------------------------------ #


def main():
    if not SNIPPETS_FILE.exists():
        print(f"ERROR: {SNIPPETS_FILE} not found. Run 1_fetch_snippets.py first.")
        return

    snippets = []
    with open(SNIPPETS_FILE) as f:
        for line in f:
            snippets.append(json.loads(line))

    print(f"Loaded {len(snippets)} snippets.")

    samples = []
    strategies = list(STRATEGY_MAP.keys())

    for snippet in tqdm(snippets, desc="Generating corruptions"):
        original_code = snippet["code"]
        original_lines = original_code.splitlines(keepends=True)

        for _ in range(CORRUPTIONS_PER_SNIPPET):
            strategy_name = random.choice(strategies)
            corrupt_fn = STRATEGY_MAP[strategy_name]

            corrupted_lines = corrupt_fn(original_lines.copy())
            malformed_search = "".join(corrupted_lines)

            # skip if corruption had no effect
            if malformed_search == original_code:
                continue

            replacement = make_replacement(original_code)

            input_text = format_input(
                file_path=snippet["file_path"],
                current_code=original_code,
                malformed_search=malformed_search,
                replacement=replacement,
            )

            # the correct answer is always the original code verbatim
            output_text = original_code

            samples.append(
                {
                    "system": SYSTEM_PROMPT,
                    "input": input_text,
                    "output": output_text,
                    "metadata": {
                        "strategy": strategy_name,
                        "repo": snippet["repo"],
                        "language": snippet["language"],
                        "file_path": snippet["file_path"],
                        "line_count": snippet["line_count"],
                    },
                }
            )

    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    for path, data in [(TRAIN_FILE, train_samples), (EVAL_FILE, eval_samples)]:
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")

    print(f"Train: {len(train_samples)} -> {TRAIN_FILE}")
    print(f"Eval:  {len(eval_samples)} -> {EVAL_FILE}")


if __name__ == "__main__":
    main()
