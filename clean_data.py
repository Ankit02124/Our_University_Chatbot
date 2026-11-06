"""Clean the raw policy text files in data/ and write them to data_clean/.

The .txt files were extracted from the university's PDF manuals, so they contain
leftover tags, page numbers on their own lines and long runs of blank lines.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data"
CLEAN_DIR = ROOT / "data_clean"


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)                          # tags
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # page numbers
    text = re.sub(r"\n{3,}", "\n\n", text)                       # blank lines
    return text.strip()


def main():
    CLEAN_DIR.mkdir(exist_ok=True)
    for path in sorted(SOURCE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        (CLEAN_DIR / path.name).write_text(clean_text(text), encoding="utf-8")
        print(f"cleaned {path.name}")


if __name__ == "__main__":
    main()
