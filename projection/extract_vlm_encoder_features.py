"""Extract one pre-projection encoder feature per prompt from a VLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel


VLM_REGISTRY = {
    "qwen3vl": ("Qwen/Qwen3-VL-2B-Instruct", "qwen3vl"),
    "llava": ("llava-hf/llava-1.5-7b-hf", "llava"),
    "clip_vit_b32": ("openai/clip-vit-base-patch32", "clip"),
    "clip_vit_b16": ("openai/clip-vit-base-patch16", "clip"),
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def first_attr(obj: Any, paths: list[str]) -> Any | None:
    for path in paths:
        value = obj
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        return value
    return None


def load_model(
    choice: str | None,
    model_id: str | None,
    loader: str,
    device: torch.device,
    device_map_auto: bool,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, Any, str, str]:
    if choice:
        default_model_id, loader = VLM_REGISTRY[choice]
        model_id = model_id or default_model_id
    if model_id is None:
        raise ValueError("Provide --choice or --model-id")

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if device_map_auto and device.type == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = "auto"

    if loader == "clip":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
    elif loader == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        tokenizer = processor.tokenizer
    elif loader == "llava":
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        tokenizer = processor.tokenizer
    else:
        model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

    if not device_map_auto or device.type != "cuda":
        model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return model, tokenizer, model_id, loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--choice", choices=sorted(VLM_REGISTRY), default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--loader",
        choices=["auto", "clip", "qwen3vl", "llava"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.prompts)
    texts = [str(row["text"]) for row in rows]
    device = torch.device(args.device)
    model, tokenizer, model_id, loader = load_model(
        args.choice,
        args.model_id,
        args.loader,
        device,
        args.device_map_auto,
        args.trust_remote_code,
    )

    if loader == "clip":
        forward_model = model.text_model
    else:
        forward_model = first_attr(
            model,
            [
                "model.language_model",
                "language_model.model",
                "language_model",
                "model",
            ],
        )
        if forward_model is None:
            forward_model = model
    feature_device = next(forward_model.parameters()).device

    features = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="VLM encoder"):
            batch_texts = texts[start : start + args.batch_size]
            tokenized = tokenizer(
                batch_texts,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(feature_device)
            attention_mask = tokenized["attention_mask"].to(feature_device)

            if loader == "clip":
                outputs = forward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                encoder_features = outputs.pooler_output
                representation = "clip_pooler_pre_text_projection"
            else:
                outputs = forward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
                sequence_lengths = attention_mask.sum(-1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=feature_device)
                encoder_features = hidden_states[batch_indices, sequence_lengths]
                representation = "last_valid_token_pre_projection"

            features.append(encoder_features.float().cpu())

    encoder_features = torch.cat(features, dim=0)
    if encoder_features.ndim != 2:
        raise ValueError(
            "Expected one VLM encoder vector per prompt shaped [N, D], got "
            f"{tuple(encoder_features.shape)}"
        )

    payload = {
        "kind": "vlm_text_encoder_features",
        "model_id": model_id,
        "loader": loader,
        "representation": representation,
        "ids": [row["id"] for row in rows],
        "texts": texts,
        "rows": rows,
        "encoder_features": encoder_features,
        "hidden_dim": int(encoder_features.shape[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {len(rows)} prompts to {args.output}")
    print("encoder_features:", tuple(encoder_features.shape))
    print("no VLM projection layer or token averaging was applied")


if __name__ == "__main__":
    main()
