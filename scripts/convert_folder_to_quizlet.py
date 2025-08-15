#!/usr/bin/env python3
"""
convert_folder_to_quizlet.py

Usage:
  # Process all .pptx files in a folder
  python convert_folder_to_quizlet.py "C:/path/to/slides"

  # Or process a single file
  python convert_folder_to_quizlet.py "C:/path/to/slides/mydeck.pptx"

Options:
  --model gpt-4o-mini          Model to use (default: gpt-4o-mini)
  --window 3                   Slides per LLM window (default: 3)
  --max-retries 4              API retry attempts (default: 4)
  --sleep 0.3                  Base sleep between API calls in seconds (default: 0.3)
  --min-def-len 12             Drop defs shorter than this (default: 12)
  --dry-run                    Parse slides but do not call the API
  --budget-usd 0               Optional soft budget (USD). If >0, exit when estimate exceeded.
  --verbose                    Extra logs
Env:
  OPENAI_API_KEY must be set.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

try:
    from openai import OpenAI
except Exception as e:
    print("ERROR: openai>=1.0.0 is required. pip install openai", file=sys.stderr)
    raise

try:
    from pptx import Presentation
except Exception as e:
    print("ERROR: python-pptx is required. pip install python-pptx", file=sys.stderr)
    raise


# ---------------------------
# Configuration defaults
# ---------------------------
DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You extract high-quality, testable term–definition pairs or big-idea summaries from military training slides.
Return ONLY valid JSON (list of objects). No commentary.

Guidelines:
- Prioritize the main concepts, big ideas, and key takeaways from each slide over isolated words.
- Include acronyms ONLY if they are explicitly defined or expanded in the provided text; otherwise, assume acronyms are already understood and do not output separate cards for them.
- Focus on military doctrine, procedures, and tactics that would be tested in Army training.
- Emphasize key concepts relevant to Army Air Defense Artillery (ADA) Basic Officer Leader Course (BOLC).
- Include step-by-step procedures, operational concepts, and critical safety information.
- Prioritize information that supports mission command and tactical decision-making.

Output schema:
[{"term": str, "definition": str, "source_slide": int, "confidence": float}]
"""

SKIP_TITLES = {"objectives", "agenda", "summary", "references", "q&a", "questions"}
MIN_DEF_LEN = 12


# ---------------------------
# Slide parsing
# ---------------------------
def _clean_ws(s: str) -> str:
    s = re.sub(r"\r\n?", "\n", s or "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def extract_slide_text(slide) -> Dict[str, str]:
    # Title
    title = ""
    try:
        if slide.shapes.title and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text or ""
    except Exception:
        pass

    # Bodies & tables
    bodies = []
    for shape in slide.shapes:
        try:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                t = shape.text or ""
                if t and t != title:
                    bodies.append(t)
            if getattr(shape, "has_table", False) and shape.has_table:
                tbl_text = []
                for r in shape.table.rows:
                    cells = []
                    for c in r.cells:
                        cells.append((c.text or "").strip())
                    tbl_text.append(" | ".join(cells))
                if tbl_text:
                    bodies.append("\n".join(tbl_text))
        except Exception:
            # skip shapes we can't read
            continue

    # Speaker notes
    notes = ""
    try:
        if slide.has_notes_slide and slide.notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text or ""
    except Exception:
        pass

    return {
        "title": _clean_ws(title),
        "body": _clean_ws("\n".join(bodies)),
        "notes": _clean_ws(notes),
    }


def slide_windows(slides_ctx: List[Dict[str, Any]], window: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Overlapping windows to let the model see neighboring slides where defs often live.
    Example with window=3: [0..2], [2..4], [4..6] (stride = window-1)
    """
    i = 0
    chunks = []
    stride = max(1, window - 1)
    while i < len(slides_ctx):
        chunk = slides_ctx[i:i + window]
        if chunk:
            chunks.append(chunk)
        i += stride
    return chunks


def build_prompt(chunk: List[Dict[str, Any]]) -> str:
    parts = []
    for s in chunk:
        title_line = f"# Slide {s['index']}: {s['title']}".strip()
        parts.append(title_line)
        if s["body"]:
            parts.append(s["body"])
        if s["notes"]:
            parts.append(f"(Notes) {s['notes']}")
        parts.append("")  # spacer
    ctx = "\n".join(parts).strip()
    return (
        "Extract term–definition pairs from the following slides. "
        "Return ONLY JSON (no markdown).\n\nSlides:\n" + ctx
    )


# ---------------------------
# LLM calling & post-processing
# ---------------------------
def call_llm(client: OpenAI, model: str, prompt: str, max_retries: int = 4, base_sleep: float = 0.3, verbose: bool = False):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.15,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            txt = (resp.choices[0].message.content or "").strip()
            # Strip codefences if any
            txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.MULTILINE).strip()
            data = json.loads(txt)
            if isinstance(data, list):
                return data
            # If not a list, try to coerce
            return []
        except json.JSONDecodeError as e:
            last_err = e
            if verbose:
                print(f"[warn] JSON decode failed (attempt {attempt+1}/{max_retries}).", file=sys.stderr)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(x in msg for x in ("rate", "quota", "429", "timeout", "temporarily unavailable", "overloaded", "500", "502", "503", "504")):
                # exponential backoff
                sleep_s = base_sleep * (2 ** attempt)
                if verbose:
                    print(f"[info] transient error: {e}. sleeping {sleep_s:.2f}s", file=sys.stderr)
                time.sleep(sleep_s)
                continue
            if verbose:
                print(f"[error] non-retryable error: {e}", file=sys.stderr)
            break
    if verbose and last_err:
        print(f"[error] giving up after {max_retries} tries: {last_err}", file=sys.stderr)
    return []


def is_skippable_title(title: str) -> bool:
    if not title:
        return False
    t = re.sub(r"[^a-z]", "", title.lower())
    return t in SKIP_TITLES


def good_definition(defi: str, min_len: int) -> bool:
    d = (defi or "").strip()
    if len(d) < min_len:
        return False
    if len(d) > 600:
        return False
    return True


def canonical_term(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def compact_definition(d: str) -> str:
    # keep 1–2 sentences
    parts = re.split(r"(?<=[.!?])\s+", d.strip())
    return " ".join(parts[:2]).strip()


def dedupe_and_filter(cards: List[Dict[str, Any]], min_def_len: int) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for c in cards:
        term = (c.get("term") or "").strip()
        defi = (c.get("definition") or "").strip()
        conf = float(c.get("confidence", 0.0) or 0.0)
        src = int(c.get("source_slide", 0) or 0)

        if not term or not good_definition(defi, min_def_len):
            continue

        key = canonical_term(term)
        old = best.get(key)
        if (old is None) or (conf > old.get("confidence", 0.0)):
            best[key] = {
                "term": term,
                "definition": compact_definition(defi),
                "confidence": conf,
                "source_slide": src,
            }

    # Sort: source_slide then term
    return sorted(best.values(), key=lambda x: (x["source_slide"], x["term"].lower()))


def extract_cards_from_presentation(pptx_path: Path, client: OpenAI, model: str, window: int, max_retries: int, base_sleep: float, min_def_len: int, dry_run: bool, verbose: bool) -> List[Dict[str, Any]]:
    prs = Presentation(str(pptx_path))
    slides_ctx = []
    for idx, slide in enumerate(prs.slides, start=1):
        info = extract_slide_text(slide)
        info["index"] = idx
        slides_ctx.append(info)

    # Optionally filter out housekeeping-only slides
    filtered = []
    for s in slides_ctx:
        if is_skippable_title(s["title"]) and not s["body"] and not s["notes"]:
            continue
        filtered.append(s)

    chunks = slide_windows(filtered, window=window)
    all_cards: List[Dict[str, Any]] = []

    for chunk in chunks:
        prompt = build_prompt(chunk)
        if dry_run:
            if verbose:
                print(f"[dry-run] would call LLM on slides {[c['index'] for c in chunk]}")
            continue

        data = call_llm(client, model, prompt, max_retries=max_retries, base_sleep=base_sleep, verbose=verbose)
        if not data:
            continue

        # Fill missing source_slide with the chunk's middle slide index
        mid_idx = chunk[len(chunk)//2]["index"]
        for c in data:
            if not c.get("source_slide"):
                c["source_slide"] = mid_idx
        all_cards.extend(data)

        # gentle pacing to play nice with rate limits
        time.sleep(base_sleep)

    return dedupe_and_filter(all_cards, min_def_len=min_def_len)


# ---------------------------
# Output
# ---------------------------
def write_quizlet_tsv(cards: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for c in cards:
            w.writerow([c["term"], c["definition"]])


def write_json_log(cards: List[Dict[str, Any]], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)


# ---------------------------
# Budget estimation (very rough)
# ---------------------------
def estimate_usd_for_chunked_calls(num_chunks: int, model: str, approx_input_toks=1200, approx_output_toks=350):
    """
    Very rough estimate based on per-window context size.
    Defaults assume window=3 with titles/bullets/notes.
    """
    pricing = {
        # input_per_1k, output_per_1k
        "gpt-4o-mini": (0.000150, 0.000600),
        "gpt-4o": (0.0025, 0.0100),
        "gpt-4.1-mini": (0.00030, 0.0012),
    }
    inp, outp = pricing.get(model, pricing["gpt-4o-mini"])
    per_call = (approx_input_toks / 1000.0) * inp + (approx_output_toks / 1000.0) * outp
    return per_call * num_chunks


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract term–definition pairs from PPTX and output Quizlet-ready TSVs.")
    ap.add_argument("path", help="Folder containing .pptx files or a single .pptx file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.3, help="Base sleep between API calls")
    ap.add_argument("--min-def-len", type=int, default=MIN_DEF_LEN)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=0.0, help="If >0, stop when rough estimate exceeds this.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"Path not found: {p}", file=sys.stderr)
        sys.exit(1)

    # Collect files
    files: List[Path] = []
    if p.is_dir():
        files = sorted([x for x in p.iterdir() if x.suffix.lower() == ".pptx"])
    else:
        if p.suffix.lower() != ".pptx":
            print("Provide a .pptx file or a folder containing .pptx files.", file=sys.stderr)
            sys.exit(1)
        files = [p]

    if not args.dry_run:
        # Init client (needs OPENAI_API_KEY)
        client = OpenAI()
    else:
        client = None  # type: ignore

    total_estimate = 0.0
    for fp in files:
        if args.verbose:
            print(f"\n=== Processing: {fp.name} ===")

        # Parse once to count chunks for budget estimation
        prs = Presentation(str(fp))
        slides_ctx = []
        for idx, slide in enumerate(prs.slides, start=1):
            info = extract_slide_text(slide)
            info["index"] = idx
            slides_ctx.append(info)
        filtered = []
        for s in slides_ctx:
            if is_skippable_title(s["title"]) and not s["body"] and not s["notes"]:
                continue
            filtered.append(s)
        chunks = slide_windows(filtered, window=args.window)
        this_estimate = estimate_usd_for_chunked_calls(len(chunks), args.model)
        total_estimate += this_estimate

        if args.budget_usd > 0 and total_estimate > args.budget_usd:
            print(f"[budget] Estimated cost {total_estimate:.4f} exceeds budget {args.budget_usd:.4f}. Stopping.", file=sys.stderr)
            break

        if args.dry_run:
            if args.verbose:
                print(f"[dry-run] {len(chunks)} chunks; est ${this_estimate:.4f}")
            continue

        # Real extraction
        cards = extract_cards_from_presentation(
            fp,
            client=client,
            model=args.model,
            window=args.window,
            max_retries=args.max_retries,
            base_sleep=args.sleep,
            min_def_len=args.min_def_len,
            dry_run=False,
            verbose=args.verbose,
        )

        out_tsv = fp.with_suffix(".quizlet.tsv")
        out_json = fp.with_suffix(".quizlet.json")
        write_quizlet_tsv(cards, out_tsv)
        write_json_log(cards, out_json)

        # Also produce a cleaned copy (trims whitespace; stays TSV)
        # (Quizlet likes simple [term<TAB>definition] lines)
        # Already clean by construction, but we can rewrite to be safe:
        cleaned = []
        for c in cards:
            term = re.sub(r"\s+", " ", c["term"]).strip()
            defi = re.sub(r"\s+", " ", c["definition"]).strip()
            cleaned.append((term, defi))
        out_clean = fp.with_suffix(".quizlet.clean.tsv")
        with out_clean.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for term, defi in cleaned:
                w.writerow([term, defi])

        if args.verbose:
            print(f"[done] {fp.name}: {len(cards)} cards →")
            print(f"       {out_tsv.name}, {out_clean.name}, {out_json.name}")

    if args.verbose:
        print(f"\nAll done. Estimated total spend: ${total_estimate:.4f}")


if __name__ == "__main__":
    main()
