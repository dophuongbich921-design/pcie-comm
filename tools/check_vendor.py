"""Check that the recorded upstream extraction has not drifted."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "UPSTREAM.json").read_text())
for name, entry in manifest["flashinfer"]["files"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != entry["vendored_sha256"]:
        raise SystemExit(f"Unrecorded vendor edit: {name}")
revision = subprocess.check_output(
    ["git", "-C", str(root / "third_party/DeepJIT"), "rev-parse", "HEAD"], text=True
).strip()
assert revision == manifest["deepjit"]["commit"]
dirty = subprocess.check_output(
    ["git", "-C", str(root / "third_party/DeepJIT"), "status", "--porcelain"], text=True
).strip()
assert not dirty, f"DeepJIT modified: {dirty}"
print(f"Verified {len(manifest['flashinfer']['files'])} extracted files and unmodified DeepJIT {revision}")
