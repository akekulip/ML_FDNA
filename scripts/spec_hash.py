import hashlib, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
h = hashlib.sha256()
for f in ("configs/spec.json", "prereg/SPEC.md"):
    h.update((root / f).read_bytes())
print(h.hexdigest()[:16])
