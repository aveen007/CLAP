"""Extract CLAP GPT-2 end-of-text features before CLAP's projection layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from msclap import CLAP  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="2023", choices=["2022", "2023"])
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.prompts)
    texts = [str(row["text"]) for row in rows]
    clap_model = CLAP(
        model_fp=args.model_file,
        version=args.version,
        use_cuda=args.use_cuda,
    )
    text_encoder = clap_model.clap.caption_encoder
    text_encoder.base.eval()
    device = next(text_encoder.base.parameters()).device

    features = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="CLAP encoder"):
            batch_texts = texts[start : start + args.batch_size]
            tokenized = clap_model.preprocess_text(batch_texts)
            hidden_states = text_encoder.base(**tokenized)[0]

            # This exactly mirrors TextEncoder.forward, stopping before projection.
            sequence_lengths = torch.ne(tokenized["input_ids"], 0).sum(-1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=device)
            encoder_features = hidden_states[batch_indices, sequence_lengths]
            features.append(encoder_features.float().cpu())

    encoder_features = torch.cat(features, dim=0)
    if encoder_features.ndim != 2 or encoder_features.shape[1] != 768:
        raise ValueError(
            "Expected pre-projection CLAP features shaped [N, 768], got "
            f"{tuple(encoder_features.shape)}"
        )

    payload = {
        "kind": "clap_text_encoder_features",
        "model_id": f"microsoft/msclap-{args.version}",
        "representation": "gpt2_endoftext_pre_projection",
        "ids": [row["id"] for row in rows],
        "texts": texts,
        "rows": rows,
        "encoder_features": encoder_features,
        "hidden_dim": 768,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {len(rows)} prompts to {args.output}")
    print("encoder_features:", tuple(encoder_features.shape))
    print("CLAP projection layer was not applied")


if __name__ == "__main__":
    main()
