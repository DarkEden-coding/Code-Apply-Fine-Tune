"""
Fetch high-quality GitHub repos and extract code snippets.
Outputs: data/snippets.jsonl
"""

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import requests
from git import Repo as GitRepo
from tqdm import tqdm

from config import (
    DATA_DIR,
    FILE_EXTENSIONS,
    GITHUB_LANGUAGES,
    GITHUB_TOKEN,
    MAX_FILES_PER_REPO,
    MAX_SNIPPET_LINES,
    MAX_SNIPPETS_PER_FILE,
    MIN_SNIPPET_LINES,
    MIN_STARS,
    REPOS_DIR,
    REPOS_PER_LANGUAGE,
    SNIPPETS_FILE,
)


def search_repos(
    language: str, per_page: int = 10, github_token: str | None = None
) -> list[dict]:
    """Search GitHub for top-starred repos in a language."""
    headers = {"Accept": "application/vnd.github+json"}
    token = github_token or GITHUB_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"language:{language} stars:>={MIN_STARS}",
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("items", [])


def clone_repo(clone_url: str, dest: Path) -> Path:
    """Shallow-clone a repo."""
    if dest.exists():
        shutil.rmtree(dest)
    GitRepo.clone_from(clone_url, str(dest), depth=1)
    return dest


def extract_blocks_from_file(filepath: Path) -> list[dict]:
    """
    Extract logical code blocks (functions, classes, methods)
    from a single source file.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = text.splitlines(keepends=True)
    if len(lines) < MIN_SNIPPET_LINES:
        return []

    blocks = []

    # strategy 1: regex-based block detection for common patterns
    # matches function/method/class definitions
    block_start_re = re.compile(
        r"^(\s*)(def |fn |func |class |export |pub fn |pub async fn |"
        r"async fn |async def |const \w+ = |let \w+ = )",
    )

    i = 0
    while i < len(lines):
        m = block_start_re.match(lines[i])
        if m:
            indent = len(m.group(1))
            start = i
            i += 1
            # collect lines that are deeper-indented, blank, or decorators
            while i < len(lines):
                stripped = lines[i].rstrip()
                if stripped == "":
                    i += 1
                    continue
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent > indent:
                    i += 1
                else:
                    break

            block_lines = lines[start:i]
            line_count = len(block_lines)
            if MIN_SNIPPET_LINES <= line_count <= MAX_SNIPPET_LINES:
                blocks.append(
                    {
                        "code": "".join(block_lines),
                        "start_line": start,
                        "end_line": i,
                        "line_count": line_count,
                    }
                )
        else:
            i += 1

    # strategy 2: sliding window for files with no detected blocks
    if not blocks:
        window_sizes = [10, 20, 35]
        for ws in window_sizes:
            if len(lines) >= ws:
                start = random.randint(0, len(lines) - ws)
                snippet = lines[start : start + ws]
                blocks.append(
                    {
                        "code": "".join(snippet),
                        "start_line": start,
                        "end_line": start + ws,
                        "line_count": ws,
                    }
                )
                break

    return blocks[:MAX_SNIPPETS_PER_FILE]


def collect_files(repo_dir: Path) -> list[Path]:
    """Collect source files from a repo, skipping vendored/test dirs."""
    skip_dirs = {
        "vendor",
        "node_modules",
        ".git",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        "venv",
        "env",
    }
    files = []
    for p in repo_dir.rglob("*"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.is_file() and p.suffix in FILE_EXTENSIONS:
            files.append(p)
    random.shuffle(files)
    return files[:MAX_FILES_PER_REPO]


def read_token_from_file(token_path: Path) -> str | None:
    if not token_path.is_file():
        return None
    token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GitHub snippets using an optional token file."
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("key.txt"),
        help="Path to a file containing the GitHub token",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    github_token = read_token_from_file(args.token_file) or GITHUB_TOKEN

    # --- gather repo metadata ---
    all_repos = []
    for lang in GITHUB_LANGUAGES:
        print(f"Searching GitHub for {lang} repos...")
        repos = search_repos(
            lang, per_page=REPOS_PER_LANGUAGE, github_token=github_token
        )
        for r in repos:
            all_repos.append(
                {
                    "name": r["full_name"],
                    "clone_url": r["clone_url"],
                    "language": lang,
                    "stars": r["stargazers_count"],
                }
            )
    print(f"Found {len(all_repos)} repos total.")

    # --- clone and extract ---
    snippets = []
    for repo_meta in tqdm(all_repos, desc="Processing repos"):
        safe_name = repo_meta["name"].replace("/", "__")
        dest = REPOS_DIR / safe_name

        try:
            clone_repo(repo_meta["clone_url"], dest)
        except Exception as e:
            print(f"  Failed to clone {repo_meta['name']}: {e}")
            continue

        files = collect_files(dest)
        for fp in files:
            rel_path = str(fp.relative_to(dest))
            blocks = extract_blocks_from_file(fp)
            for block in blocks:
                snippets.append(
                    {
                        "repo": repo_meta["name"],
                        "language": repo_meta["language"],
                        "file_path": rel_path,
                        "code": block["code"],
                        "start_line": block["start_line"],
                        "end_line": block["end_line"],
                        "line_count": block["line_count"],
                    }
                )

        # cleanup disk space
        shutil.rmtree(dest, ignore_errors=True)

    random.shuffle(snippets)
    with open(SNIPPETS_FILE, "w") as f:
        for s in snippets:
            f.write(json.dumps(s) + "\n")

    print(f"Extracted {len(snippets)} snippets -> {SNIPPETS_FILE}")


if __name__ == "__main__":
    main()
