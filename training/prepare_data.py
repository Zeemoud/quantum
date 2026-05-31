"""
Quantum — Data preparation script.
Downloads and cleans training data from Wikipedia and Project Gutenberg.

Usage:
    python -m training.prepare_data
    python -m training.prepare_data --max-articles 5000 --lang both
"""

import argparse
import re
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove wiki markup, extra whitespace, and garbage lines."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove wiki markup
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"==+[^=]+=+=", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove lines that are too short (navigation, categories, etc.)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 40]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"

def fetch_wikipedia_articles(lang: str, max_articles: int) -> list[str]:
    """Fetch random Wikipedia articles via the API."""
    print(f"  Fetching Wikipedia ({lang})...")
    texts = []
    fetched = 0

    while fetched < max_articles:
        batch = min(20, max_articles - fetched)
        url = (
            WIKI_API.format(lang=lang)
            + f"?action=query&generator=random&grnnamespace=0&grnlimit={batch}"
            + "&prop=extracts&explaintext=1&exlimit=max&format=json"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantumAI/0.1"})
            with urllib.request.urlopen(req, timeout=15) as r:
                import json
                data = json.loads(r.read().decode())
            pages = data.get("query", {}).get("pages", {}).values()
            for page in pages:
                extract = page.get("extract", "")
                if len(extract) > 500:
                    texts.append(clean_text(extract))
                    fetched += 1
        except (urllib.error.URLError, Exception) as e:
            print(f"    Warning: {e}")
            break

        print(f"    {fetched}/{max_articles} articles", end="\r")

    print(f"    ✓ {fetched} articles fetched from Wikipedia ({lang})")
    return texts


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------

# A small curated list of classic books (public domain, bilingual)
GUTENBERG_BOOKS = {
    "en": [
        ("1342", "Pride and Prejudice"),
        ("11", "Alice in Wonderland"),
        ("84", "Frankenstein"),
        ("1661", "Sherlock Holmes"),
        ("2701", "Moby Dick"),
        ("98", "A Tale of Two Cities"),
        ("1080", "A Modest Proposal"),
        ("219", "Heart of Darkness"),
    ],
    "fr": [
        ("13951", "Les Misérables"),
        ("4650", "Le Comte de Monte-Cristo"),
        ("17989", "Madame Bovary"),
        ("5097", "Vingt Mille Lieues"),
        ("14287", "Le Tour du Monde"),
        ("799", "Le Père Goriot"),
    ],
}

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

def fetch_gutenberg_books(lang: str) -> list[str]:
    """Download books from Project Gutenberg."""
    print(f"  Fetching Gutenberg ({lang})...")
    texts = []
    books = GUTENBERG_BOOKS.get(lang, [])

    for book_id, title in books:
        url = GUTENBERG_URL.format(id=book_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantumAI/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            # Strip Gutenberg header/footer
            start = raw.find("*** START OF")
            end = raw.find("*** END OF")
            if start != -1:
                raw = raw[start + 50:]
            if end != -1:
                raw = raw[:end]
            cleaned = clean_text(raw)
            if len(cleaned) > 1000:
                texts.append(cleaned)
                print(f"    ✓ {title}")
        except Exception as e:
            print(f"    ✗ {title}: {e}")

    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Quantum training data")
    parser.add_argument("--max-articles", type=int, default=2000,
                        help="Number of Wikipedia articles per language (default: 2000)")
    parser.add_argument("--lang", choices=["fr", "en", "both"], default="both")
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = ["fr", "en"] if args.lang == "both" else [args.lang]

    for lang in langs:
        print(f"\n[{lang.upper()}]")
        texts = []

        # Wikipedia
        wiki_texts = fetch_wikipedia_articles(lang, args.max_articles)
        texts.extend(wiki_texts)

        # Gutenberg
        gutenberg_texts = fetch_gutenberg_books(lang)
        texts.extend(gutenberg_texts)

        # Save
        output_file = output_dir / f"corpus_{lang}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))

        total_chars = sum(len(t) for t in texts)
        print(f"  ✓ Saved {len(texts)} documents → {output_file}")
        print(f"  ✓ Total: {total_chars:,} characters (~{total_chars // 5:,} tokens)")

    print("\nDone! You can now run: python -m training.train")


if __name__ == "__main__":
    main()