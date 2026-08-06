"""Prepare leakage-controlled prompt manifests for VLM-to-CLAP projection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = text.replace("_", " ").lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def load_esc50_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        labels = {row["category"] for row in csv.DictReader(handle)}
    if len(labels) != 50:
        raise ValueError(f"Expected 50 ESC-50 labels, found {len(labels)} in {path}")
    return sorted(label.replace("_", " ") for label in labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prompts", type=Path, required=True)
    parser.add_argument("--esc50-meta", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("projection/prompts"),
    )
    parser.add_argument("--template", default="this is the sound of {text}")
    args = parser.parse_args()

    base_rows = load_jsonl(args.base_prompts)
    esc50_labels = load_esc50_labels(args.esc50_meta)
    blocked_tokens = {
        token
        for label in esc50_labels
        for token in normalize_text(label).split()
    }

    fit_rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    skipped_for_leakage = 0
    for row in base_rows:
        source_split = str(row.get("split_closed", ""))
        if source_split not in {"train", "val"}:
            continue

        raw_text = str(row["text"]).strip()
        row_tokens = set(normalize_text(raw_text).split())
        row_tokens.update(normalize_text(str(row.get("concept_group", ""))).split())
        if row_tokens & blocked_tokens:
            skipped_for_leakage += 1
            continue

        text = args.template.format(text=raw_text)
        normalized = normalize_text(text)
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        fit_rows.append(
            {
                "id": f"fit_{row['id']}",
                "text": text,
                "split": source_split,
                "source_id": row["id"],
                "category": row.get("category"),
                "concept_group": row.get("concept_group"),
            }
        )

    eval_rows = []
    for label in esc50_labels:
        slug = normalize_text(label).replace(" ", "_")
        eval_rows.append(
            {
                "id": f"esc50_{slug}",
                "text": args.template.format(text=label),
                "split": "esc50_test",
                "class_name": label,
                "category": "esc50",
                "concept_group": slug,
            }
        )

    if not any(row["split"] == "train" for row in fit_rows):
        raise ValueError("No training prompts remained after leakage filtering.")
    if not any(row["split"] == "val" for row in fit_rows):
        raise ValueError("No validation prompts remained after leakage filtering.")

    output_dir = args.output_dir
    write_jsonl(output_dir / "projection_fit_prompts.jsonl", fit_rows)
    write_jsonl(output_dir / "esc50_eval_prompts.jsonl", eval_rows)
    write_jsonl(output_dir / "vlm_all_prompts.jsonl", fit_rows + eval_rows)

    train_count = sum(row["split"] == "train" for row in fit_rows)
    val_count = sum(row["split"] == "val" for row in fit_rows)
    print(f"fit prompts: {len(fit_rows)} (train={train_count}, val={val_count})")
    print(f"ESC-50 evaluation prompts: {len(eval_rows)}")
    print(f"removed for lexical ESC-50 leakage: {skipped_for_leakage}")
    print(f"wrote manifests to {output_dir}")


if __name__ == "__main__":
    main()
