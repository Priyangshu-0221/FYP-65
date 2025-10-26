"""Utility script to install backend (Python) and frontend (Node) dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
FRONTEND_DIR = ROOT_DIR / "web_frontend"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run a shell command and raise a helpful error on failure."""
    location = cwd or ROOT_DIR
    print(f"\n>>> Running: {' '.join(command)} (cwd={location})")
    try:
        subprocess.run(command, cwd=location, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - command failure handling
        raise SystemExit(
            f"Command {' '.join(command)} failed with exit code {exc.returncode}."
        ) from exc


def ensure_backend_dependencies() -> None:
    """Install Python dependencies listed in requirements.txt using current interpreter."""
    if not REQUIREMENTS_FILE.exists():
        raise SystemExit(f"Missing requirements file: {REQUIREMENTS_FILE}")

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_command([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])



def ensure_frontend_dependencies() -> None:
    """Install Node dependencies inside the web_frontend directory."""
    if not FRONTEND_DIR.exists():
        raise SystemExit(f"Missing frontend directory: {FRONTEND_DIR}")

    package_json = FRONTEND_DIR / "package.json"
    if not package_json.exists():
        raise SystemExit(f"Missing package.json in {FRONTEND_DIR}. Expected frontend project setup.")

    run_command(["npm", "install"], cwd=FRONTEND_DIR)



def main() -> None:
    print("Starting dependency installation for internship recommender project...")
    ensure_backend_dependencies()
    ensure_frontend_dependencies()
    print("\nAll dependencies installed successfully. Ready to run the project!")


if __name__ == "__main__":
    main()
