#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import sys

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_checkpoints(root: Path):
    checkpoints = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = CHECKPOINT_RE.match(p.name)
        if m:
            step = int(m.group(1))
            checkpoints.append((step, p))
    return sorted(checkpoints, key=lambda x: x[0])


def remove_global_steps(root_dir: str, keep_count: int = 2, dry_run: bool = False):
    root = Path(root_dir)

    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {root}")

    checkpoints = find_checkpoints(root)

    if not checkpoints:
        print(f"No checkpoint-N directories found in {root}")
        return

    if keep_count >= len(checkpoints):
        print(
            f"Nothing to delete: keep_count ({keep_count}) >= "
            f"number of checkpoints ({len(checkpoints)})"
        )
        return

    to_delete = checkpoints[:-keep_count]
    to_keep = checkpoints[-keep_count:]

    print("Keeping:")
    for step, path in to_keep:
        print(f"  {path.name}")

    print("Cleaning:")
    for step, ckpt_dir in to_delete:
        matches = sorted(ckpt_dir.glob("global_step*"))
        if not matches:
            print(f"  {ckpt_dir.name}: no global_step* found")
            continue

        for target in matches:
            print(f"  {'[dry-run] ' if dry_run else ''}{target}")
            if not dry_run:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

    print("Done.")


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    keep_count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    dry_run = "--dry-run" in sys.argv
    remove_global_steps(root_dir, keep_count, dry_run)