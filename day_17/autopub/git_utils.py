from __future__ import annotations

import os, subprocess, shlex
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()

def run(cmd: str, cwd: Path | None = None, check=True) -> subprocess.CompletedProcess:
    console.log(f"[dim]$ {cmd}[/]")
    return subprocess.run(shlex.split(cmd), cwd=str(cwd) if cwd else None, check=check, capture_output=True, text=True)

def ensure_user_config():
    name = os.getenv("GIT_USER_NAME")
    email = os.getenv("GIT_USER_EMAIL")
    if name:
        run(f'git config --global user.name "{name}"', check=False)
    if email:
        run(f'git config --global user.email "{email}"', check=False)

def tokenized_url(url: str) -> str:
    token = os.getenv("GIT_HTTPS_TOKEN")
    if token and url.startswith("https://"):
        return url.replace("https://", f"https://{token}:x-oauth-basic@")
    return url

def clone_or_update(repo_url: str, workdir: Path) -> Path:
    ensure_user_config()
    workdir.mkdir(parents=True, exist_ok=True)
    repo_name = Path(repo_url.rstrip("/").split("/")[-1])
    if repo_name.suffix == ".git":
        repo_name = repo_name.with_suffix("")
    dest = workdir / repo_name.name
    if dest.exists() and (dest / ".git").exists():
        console.print(f"[green]Обновляю репозиторий[/]: {dest}")
        run("git fetch --all", cwd=dest)
        run("git pull --ff-only", cwd=dest)
        return dest
    url = tokenized_url(repo_url)
    console.print(f"[green]Клонирую[/]: {url} → {dest}")
    run(f"git clone {url} {shlex.quote(str(dest))}")
    return dest

def new_branch(repo_root: Path, prefix: str, slug: str) -> str:
    from datetime import datetime
    branch = f"{prefix}/{slug}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    run(f"git checkout -b {branch}", cwd=repo_root)
    return branch

def commit_all(repo_root: Path, message: str):
    run("git add -A", cwd=repo_root)
    run(f'git commit -m "{message}"', cwd=repo_root)

def push_branch(repo_root: Path, branch: str):
    run(f"git push -u origin {branch}", cwd=repo_root)
