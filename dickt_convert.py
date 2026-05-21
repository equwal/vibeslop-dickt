"""
Kenkyusha 新和英大辞典 — English → Portuguese translator
Uses Google Translate (no API key needed).

Run:  python translate_kenkyusha.py <input_zip_or_dir> <output_dir>

Speed improvements over v1:
  • Sentinel-joined batches  — segments joined with a rare delimiter, one HTTP call per batch
  • Thread pool              — WORKERS concurrent requests
  • Exponential backoff      — on 429 / transient errors
  • tqdm progress bar (optional; pip install tqdm)
"""

import json
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import zipfile
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_LANG    = sys.argv[3] if len(sys.argv) > 3 else None
BATCH_SIZE     = 40      # segments joined per HTTP call
WORKERS        = 8       # concurrent threads
RETRY_LIMIT    = 6       # max attempts per batch
BACKOFF_BASE   = 1.5     # seconds; doubles each retry
REQUEST_GAP    = 0.02    # polite pause after each successful request

# Sentinel must survive a round-trip through Google Translate unchanged.
# This Unicode private-use sequence is invisible and never appears in dictionary text.
SEP = "\uE000|\uE001"

# ── Google Translate (gtx endpoint, sentinel-batching) ────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def translate_batch(texts: list[str]) -> list[str]:
    """
    Translate a list of strings in one HTTP call by joining them with SEP.
    Splits the translated result back on the same separator.
    Falls back to originals on permanent failure.
    """
    if not texts:
        return texts

    joined = SEP.join(t[:4000] for t in texts)
    encoded = urllib.parse.quote(joined)
    url = (
        f"https://translate.googleapis.com/translate_a/single"
        f"?client=gtx&sl=en&tl={TARGET_LANG}&dt=t&q={encoded}"
    )
    req = urllib.request.Request(url, headers=_HEADERS)

    delay = BACKOFF_BASE
    for attempt in range(RETRY_LIMIT):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # result[0] is a list of [translated_chunk, original_chunk, ...]
            translated_joined = "".join(
                item[0] for item in result[0] if isinstance(item[0], str)
            )

            # Split on any variant the translator may have introduced
            # (it sometimes adds spaces around the separator)
            parts = re.split(r"\s*\uE000\s*\|\s*\uE001\s*", translated_joined)

            # Pad or trim to match input length
            while len(parts) < len(texts):
                parts.append(texts[len(parts)])

            time.sleep(REQUEST_GAP)
            return parts[:len(texts)]

        except urllib.error.HTTPError as e:
            code = e.code
            print(f"\n  ⚠  HTTP {code} (attempt {attempt+1}); backing off {delay:.1f}s …", flush=True)
            time.sleep(delay)
            delay *= 2

        except Exception as e:
            print(f"\n  ⚠  Error (attempt {attempt+1}): {e}; backing off {delay:.1f}s …", flush=True)
            time.sleep(delay)
            delay *= 2

    print(f"  ✗  Giving up on batch of {len(texts)} — keeping originals", flush=True)
    return texts


# ── Per-line segment extraction ───────────────────────────────────────────────

def extract_english_segments(text: str):
    """
    Return list of (start, end, english_text) for all English portions.

    Two patterns in this dictionary:
      A) Lines with ideographic space U+3000 — English is everything after it.
      B) Standalone English lines (no U+3000, contains Latin letters,
         not a JP/romaji header).
    """
    segments = []
    pos = 0
    for line in text.split("\n"):
        line_len = len(line)
        if "\u3000" in line:
            idx = line.index("\u3000") + 1
            eng = line[idx:]
            if re.search(r"[a-zA-Z]{2,}", eng):
                start = pos + idx
                segments.append((start, start + len(eng), eng))
        else:
            if (re.search(r"[a-zA-Z]{2,}", line)
                    and not line.strip().startswith("[ローマ字]")
                    and not re.match(r'^[\u4e00-\u9fff\u3040-\u30ff].*\[ローマ字\]', line)):
                segments.append((pos, pos + line_len, line))
        pos += line_len + 1
    return segments


def apply_translations(text: str, segments, translations) -> str:
    """Splice translations back into the original string at the correct offsets."""
    if not segments:
        return text
    result = []
    prev = 0
    for (start, end, _), t in zip(segments, translations):
        result.append(text[prev:start])
        result.append(t)
        prev = end
    result.append(text[prev:])
    return "".join(result)


# ── Batched + parallel translation of a flat list ────────────────────────────

def translate_all(flat_eng: list[str]) -> list[str]:
    """
    Translate every string in flat_eng using WORKERS threads and BATCH_SIZE
    grouping.  Returns a list of the same length with translated strings.
    """
    total = len(flat_eng)
    if total == 0:
        return []

    batches = []
    for start in range(0, total, BATCH_SIZE):
        batches.append((start, flat_eng[start:start + BATCH_SIZE]))

    results = [""] * total

    pbar = tqdm(total=total, unit="seg", dynamic_ncols=True, leave=False) if HAS_TQDM else None

    def worker(start_idx, texts):
        translated = translate_batch(texts)
        return start_idx, translated

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, s, t): s for s, t in batches}
        for fut in as_completed(futures):
            start_idx, translated = fut.result()
            for i, t in enumerate(translated):
                results[start_idx + i] = t
            if pbar:
                pbar.update(len(translated))
            else:
                done = sum(1 for r in results if r)
                print(f"    {done}/{total} segments done   \r", end="", flush=True)

    if pbar:
        pbar.close()
    else:
        print()

    return results


# ── File processor ────────────────────────────────────────────────────────────

def process_term_bank(input_path: Path, output_path: Path):
    print(f"\n📖  {input_path.name}", flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_segs = []
    for i, entry in enumerate(data):
        if (isinstance(entry, list) and len(entry) > 5
                and isinstance(entry[5], list) and entry[5]
                and isinstance(entry[5][0], str)):
            segs = extract_english_segments(entry[5][0])
            if segs:
                all_segs.append((i, segs))

    flat_eng   = []
    flat_index = []
    for ai, (_, segs) in enumerate(all_segs):
        for si, (_, _, eng) in enumerate(segs):
            flat_eng.append(eng)
            flat_index.append((ai, si))

    total = len(flat_eng)
    print(f"    {len(data)} entries — {total} English segments to translate", flush=True)

    flat_translated = translate_all(flat_eng)

    trans_map = {idx: t for idx, t in zip(flat_index, flat_translated)}
    for ai, (entry_i, segs) in enumerate(all_segs):
        seg_trans = [trans_map.get((ai, si), segs[si][2]) for si in range(len(segs))]
        data[entry_i][5][0] = apply_translations(data[entry_i][5][0], segs, seg_trans)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"    ✓ saved → {output_path.name}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: python translate_kenkyusha.py <input_zip> <output_dir> <target_lang>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"⚙  Workers={WORKERS}  BatchSize={BATCH_SIZE}  Target={TARGET_LANG}")

    if input_path.suffix.lower() != ".zip":
        print(f"Error: {input_path} is not a .zip file")
        sys.exit(1)
    print(f"📦  Extracting {input_path.name} …", flush=True)
    _tmpdir = tempfile.mkdtemp(prefix="kenkyusha_")
    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(_tmpdir)
    candidates = list(Path(_tmpdir).rglob("term_bank_1.json"))
    input_dir = candidates[0].parent if candidates else Path(_tmpdir)
    print(f"    → unpacked to {input_dir}", flush=True)

    try:
        idx_src = input_dir / "index.json"
        if idx_src.exists():
            with open(idx_src, encoding="utf-8") as f:
                idx = json.load(f)
            idx["title"]    = idx.get("title", "Dictionary") + f" ({TARGET_LANG.upper()})"
            idx["revision"] = idx.get("revision", "1") + f"-{TARGET_LANG}"
            with open(output_dir / "index.json", "w", encoding="utf-8") as f:
                json.dump(idx, f, ensure_ascii=False, indent=2)
            print("📝  index.json updated")

        term_banks = sorted(input_dir.glob("term_bank_*.json"),
                            key=lambda p: int(re.search(r'\d+', p.name).group()))
        print(f"Found {len(term_banks)} term_bank files")

        if not term_banks:
            print("⚠  No term_bank_*.json files found — check the zip structure.")
            sys.exit(1)

        t0 = time.time()
        for tb in term_banks:
            process_term_bank(tb, output_dir / tb.name)

        elapsed = int(time.time() - t0)
        print(f"\n✅  Done in {elapsed//60}m {elapsed%60}s")
        print(f"📁  Output: {output_dir}")
        print("\nNext: zip the output folder contents and import into Yomitan.")

    finally:
        if _tmpdir:
            shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
