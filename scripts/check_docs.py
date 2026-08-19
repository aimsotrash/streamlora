"""Verify every relative link and image in the documentation resolves."""
from __future__ import annotations

import pathlib
import re
import sys

LINK = re.compile(r'\[[^\]]*\]\(([^)#\s]+)(#[^)]*)?\)')


def main(root: str = ".") -> int:
    base = pathlib.Path(root)
    files = [base / "README.md", *sorted((base / "docs").glob("*.md"))]
    bad: list[str] = []
    for md in files:
        if not md.exists():
            continue
        for m in LINK.finditer(md.read_text()):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            # Relative to the file that contains the link, always.
            if not (md.parent / target).resolve().exists():
                bad.append(f"{md}: {target}")
    print(f"checked {len(files)} markdown files")
    for b in bad:
        print("  BROKEN:", b)
    print("all links resolve" if not bad else f"{len(bad)} broken links")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
