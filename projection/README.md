# VLM-to-CLAP Encoder-Space Projection

This experiment maps one VLM text-encoder vector per prompt into the raw
768-dimensional GPT-2 feature space used by Microsoft CLAP 2023. Projection
fitting does not use CLAP's 1024-dimensional joint-space text embeddings.

At evaluation time, the native CLAP text encoder is disabled. A projected
`[50, 768]` tensor replaces its output and is passed through only CLAP's frozen
text projection module so it can be compared with CLAP audio embeddings.

## Files

- `prompts/projection_fit_prompts.jsonl`: train and validation prompts encoded
  by both CLAP and the VLM.
- `prompts/esc50_eval_prompts.jsonl`: 50 held-out ESC-50 prompts encoded only by
  the VLM and projected for evaluation.
- `prompts/vlm_all_prompts.jsonl`: combined Kaggle input manifest.
- `extract_clap_encoder_features.py`: extracts GPT-2 end-of-text features before
  CLAP's projection layer.
- `extract_vlm_encoder_features.py`: extracts one final encoder feature per
  prompt without averaging tokens or applying a VLM projection layer.
- `evaluate_projected_esc50.py`: evaluates projected 768D features without
  invoking CLAP's text encoder.

## CLAP Target Extraction

Run on the server:

```bash
cd /home/tnn/shared/CLAP
python projection/extract_clap_encoder_features.py \
  --prompts projection/prompts/projection_fit_prompts.jsonl \
  --output projection/outputs/clap_encoder_fit.pt \
  --use-cuda
```

The output bundle contains `encoder_features` shaped `[N, 768]`. It contains no
CLAP targets for the 50 ESC-50 evaluation prompts.

## VLM Extraction

Upload `prompts/vlm_all_prompts.jsonl` and
`extract_vlm_encoder_features.py` to Kaggle. Example for LLaVA:

```bash
python extract_vlm_encoder_features.py \
  --prompts vlm_all_prompts.jsonl \
  --choice llava \
  --device-map-auto \
  --output vlm_encoder_all.pt
```

Example for Qwen3-VL:

```bash
python extract_vlm_encoder_features.py \
  --prompts vlm_all_prompts.jsonl \
  --choice qwen3vl \
  --device-map-auto \
  --trust-remote-code \
  --output vlm_encoder_all.pt
```

The VLM bundle contains `encoder_features` shaped `[N, D_vlm]`. Fit the mapping
with rows whose manifest split is `train`, use `val` for model selection, and
apply it to rows whose split is `esc50_test`.

Save the 50 projected rows in prompt order as either a raw `[50, 768]` tensor or
as a bundle that preserves ids:

```python
torch.save(
    {
        "kind": "projected_clap_encoder_features",
        "ids": esc50_ids,
        "projected_encoder_features": projected_esc50.float().cpu(),
    },
    "projected_esc50_encoder_features.pt",
)
```

## Projected Evaluation

```bash
cd /home/tnn/shared/CLAP
python projection/evaluate_projected_esc50.py \
  --dataset-root /home/tnn/shared/ESC-50 \
  --prompts projection/prompts/esc50_eval_prompts.jsonl \
  --projected-features projection/outputs/projected_esc50_encoder_features.pt \
  --use-cuda
```

The evaluator prints `CLAP text encoder used: no` before reporting projected
ESC-50 accuracy. The native zero-shot reference from this environment is 94.35%.
