"""Author a checksum inventory for a deliberately edited release COPY."""
import argparse
import hashlib
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def included(path, root):
    return path.is_file() and not path.is_symlink() and not any(
        p in ("__pycache__", "reproduction_runs", ".git") for p in path.relative_to(root).parts)


def seal(root):
    manifests = sorted(root.rglob("SHA256SUMS.txt"), key=lambda p: len(p.parts), reverse=True)
    if root / "SHA256SUMS.txt" not in manifests:
        manifests.append(root / "SHA256SUMS.txt")
    for manifest in manifests:
        base = manifest.parent
        files = sorted(p for p in base.rglob("*") if included(p, root) and p != manifest)
        text = "".join(digest(p) + "  ./" + p.relative_to(base).as_posix() + "\n" for p in files)
        with manifest.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print("Sealed authoring copy:", root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--authoring-copy", action="store_true", required=True,
                   help="Confirm intentional edits to a copy; NEVER use to dismiss a failed integrity check")
    a = p.parse_args()
    seal(a.root.resolve())


if __name__ == "__main__":
    main()
