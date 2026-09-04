"""Install and verify Kaggle credentials.

Kaggle has two schemes and this handles both:

  * **Standalone access token** (current) - a single `KGAT_...` string copied
    from kaggle.com/settings/api. No username needed; it goes in
    `~/.kaggle/access_token`.
  * **Legacy kaggle.json** - a downloaded file with username + key.

    python src\\setup_kaggle.py --token KGAT_xxxxxxxx     # paste a token
    python src\\setup_kaggle.py                           # find kaggle.json in Downloads
    python src\\setup_kaggle.py --from "C:\\path\\kaggle.json"
    python src\\setup_kaggle.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"
ACCESS_TOKEN = KAGGLE_DIR / "access_token"


def _lock_down(path: Path) -> None:
    """Make the credential readable only by this user (best effort)."""
    try:
        os.chmod(path, 0o600)
        if sys.platform == "win32":
            user = os.environ.get("USERNAME", "")
            subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                           capture_output=True, text=True)
    except Exception as e:  # noqa: BLE001 - permissions are best-effort
        print(f"[warn] could not tighten permissions on {path.name}: {e}")


def install_token(token: str) -> None:
    """Install a standalone KGAT_ access token."""
    token = token.strip()
    if not token:
        raise SystemExit("Empty token.")
    if not token.startswith("KGAT_"):
        print("[warn] token does not start with 'KGAT_' - continuing, but check you "
              "copied the whole string from kaggle.com/settings/api")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    ACCESS_TOKEN.write_text(token, encoding="ascii")  # no trailing newline
    _lock_down(ACCESS_TOKEN)
    print(f"[ok] access token installed -> {ACCESS_TOKEN}")


def find_token(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates: list[Path] = []
    for d in (Path.home() / "Downloads", Path.home() / "Desktop", Path.cwd()):
        if d.exists():
            candidates.extend(sorted(d.glob("kaggle*.json")))
    # Newest first - people often download the token more than once.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def install(src: Path) -> None:
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"{src} is not valid JSON ({e}). Re-download the token.")
    if "username" not in data or "key" not in data:
        raise SystemExit(f"{src} has no 'username'/'key'. That is not a Kaggle API token.")

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, KAGGLE_JSON)
    _lock_down(KAGGLE_JSON)
    print(f"[ok] installed token for user '{data['username']}' -> {KAGGLE_JSON}")


def kaggle_cmd() -> list[str]:
    exe = Path(sys.executable).parent / "kaggle.exe"
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "kaggle"]


def whoami() -> tuple[str | None, str | None]:
    """(username, auth_method) as the CLI sees them."""
    try:
        proc = subprocess.run(kaggle_cmd() + ["config", "view"],
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
        user = method = None
        for line in (proc.stdout or "").splitlines():
            if "username:" in line:
                user = line.split("username:", 1)[1].strip()
            elif "auth_method:" in line:
                method = line.split("auth_method:", 1)[1].strip()
        return (user if user and user.lower() != "none" else None), method
    except Exception:  # noqa: BLE001
        return None, None


def verify() -> bool:
    # utf-8/replace: the kaggle CLI emits bytes cp1252 cannot decode on Windows.
    proc = subprocess.run(kaggle_cmd() + ["datasets", "list", "--max-size", "1000"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if "ref" in out.lower() and "lastUpdated" in out:
        user, method = whoami()
        print("[ok] Kaggle API authenticated - `kaggle datasets list` returned results.")
        print(f"[ok] account: {user or '(unknown)'}   auth: {method or '(unknown)'}")
        return True
    print("[FAIL] Kaggle API call did not succeed:")
    print("  " + out.strip().replace("\n", "\n  ")[:900])
    if "401" in out or "authenticat" in out.lower():
        print("\n  The token was rejected. Create a NEW token (the old one may be revoked):")
        print("  kaggle.com -> avatar -> Settings -> API -> Create New Token")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Install and verify Kaggle credentials.")
    p.add_argument("--token", default=None,
                   help="A standalone KGAT_... access token from kaggle.com/settings/api.")
    p.add_argument("--from", dest="src", default=None, help="Path to a downloaded kaggle.json.")
    p.add_argument("--verify-only", action="store_true", help="Skip install, just test auth.")
    args = p.parse_args()

    if args.token:
        install_token(args.token)
    elif not args.verify_only:
        if ACCESS_TOKEN.exists() and not args.src:
            print(f"[skip] {ACCESS_TOKEN} already exists - verifying it instead.")
        elif KAGGLE_JSON.exists() and not args.src:
            print(f"[skip] {KAGGLE_JSON} already exists - verifying it instead.")
        else:
            src = find_token(args.src)
            if src is None:
                raise SystemExit(
                    "\nNo Kaggle credentials found.\n\n"
                    "  Easiest: kaggle.com/settings/api -> Generate New Token, copy the\n"
                    "  KGAT_... string, then:\n"
                    "      python src\\setup_kaggle.py --token KGAT_xxxxxxxx\n\n"
                    "  Or download the legacy kaggle.json and re-run this, or --from <path>.\n\n"
                    "  While you are there: Settings -> Phone Verification. GPU access\n"
                    "  requires it and it can lag, so do it now, not on Saturday.\n"
                )
            print(f"[found] {src}")
            install(src)

    ok = verify()
    if ok:
        print("\nNext: push the training dataset")
        print("  python src\\build_kaggle_dataset.py --labels C:\\dvad\\data\\pseudo_labels.jsonl --push")
        print("\nAlso confirm Kaggle -> Settings -> Phone Verification is done, or the")
        print("notebook cannot select a GPU accelerator.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
