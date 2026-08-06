"""Evaluate projected 768D VLM features while bypassing CLAP's text encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from esc50_dataset import ESC50  # noqa: E402
from msclap import CLAP  # noqa: E402


FEATURE_KEYS = (
    "projected_encoder_features",
    "encoder_features",
    "projected",
    "embeddings",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_projected_features(path: Path, ordered_ids: list[str]) -> torch.Tensor:
    payload = torch_load(path)
    if isinstance(payload, torch.Tensor):
        features = payload
        ids = None
    elif isinstance(payload, dict):
        features = next(
            (payload[key] for key in FEATURE_KEYS if key in payload),
            None,
        )
        if features is None:
            raise ValueError(
                f"{path} has no feature tensor. Expected one of {FEATURE_KEYS}."
            )
        ids = payload.get("ids")
    else:
        raise TypeError(f"Unsupported projected feature payload: {type(payload)!r}")

    features = torch.as_tensor(features).float()
    if features.ndim != 2 or features.shape[1] != 768:
        raise ValueError(
            "Projected encoder features must have shape [N, 768], got "
            f"{tuple(features.shape)}"
        )

    if ids is None:
        if features.shape[0] != len(ordered_ids):
            raise ValueError(
                f"Tensor has {features.shape[0]} rows but {len(ordered_ids)} prompts."
            )
        return features

    id_to_index = {str(row_id): i for i, row_id in enumerate(ids)}
    missing = [row_id for row_id in ordered_ids if row_id not in id_to_index]
    if missing:
        raise ValueError(f"Projected bundle is missing prompt ids: {missing}")
    return features[[id_to_index[row_id] for row_id in ordered_ids]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--projected-features", type=Path, required=True)
    parser.add_argument("--version", default="2023", choices=["2022", "2023"])
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    dataset = ESC50(root=args.dataset_root.expanduser().resolve(), download=False)
    rows = load_rows(args.prompts)
    rows_by_class = {str(row["class_name"]): row for row in rows}
    missing_classes = [label for label in dataset.classes if label not in rows_by_class]
    if missing_classes:
        raise ValueError(f"Prompt manifest is missing ESC-50 classes: {missing_classes}")

    ordered_rows = [rows_by_class[label] for label in dataset.classes]
    ordered_ids = [str(row["id"]) for row in ordered_rows]
    encoder_features = load_projected_features(args.projected_features, ordered_ids)

    clap_model = CLAP(
        model_fp=args.model_file,
        version=args.version,
        use_cuda=args.use_cuda,
    )
    text_projection = clap_model.clap.caption_encoder.projection

    # Native CLAP text encoding is intentionally disabled for this evaluation.
    # native_text_embeddings = clap_model.get_text_embeddings(prompt_texts)
    clap_model.clap.caption_encoder.base = None

    projection_param = next(text_projection.parameters())
    encoder_features = encoder_features.to(
        device=projection_param.device,
        dtype=projection_param.dtype,
    )
    with torch.inference_mode():
        text_embeddings = text_projection(encoder_features)

    predictions, targets = [], []
    for i in tqdm(range(len(dataset)), desc="Projected ESC-50"):
        audio_path, _, one_hot_target = dataset[i]
        audio_embeddings = clap_model.get_audio_embeddings([audio_path], resample=True)
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        prediction = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        predictions.append(prediction)
        targets.append(one_hot_target.detach().cpu().numpy())

    targets_array = np.concatenate(targets, axis=0)
    predictions_array = np.concatenate(predictions, axis=0)
    accuracy = accuracy_score(
        np.argmax(targets_array, axis=1),
        np.argmax(predictions_array, axis=1),
    )
    print("CLAP text encoder used: no")
    print("projected encoder features:", tuple(encoder_features.shape))
    print(f"Projected ESC50 Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
