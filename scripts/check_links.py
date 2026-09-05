"""Check local Markdown/HTML asset targets without contacting external websites."""

from pathlib import Path
import re
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
files = [root / "README.md", root / "README_CN.md", *list((root / "docs").glob("*.md"))]
errors = []
for path in files:
    text = path.read_text("utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'(?:src|href)="([^"]+)"', text)
    for value in targets:
        if value.startswith(("http:", "https:", "mailto:", "#")):
            continue
        target = unquote(value.split("#")[0].strip("<>"))
        if target and not (path.parent / target).exists():
            errors.append(f"{path.relative_to(root)}: {target}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Local links passed in {len(files)} entry documents")
