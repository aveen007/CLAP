# ESC-50 Evaluation Results

## Local Run

The Microsoft CLAP 2023 model was evaluated zero-shot on all 2,000 clips in the
downloaded ESC-50 repository. No ESC-50 training or fine-tuning was performed.

```bash
python -u zero_shot_classification.py \
  --dataset-root /home/tnn/shared/ESC-50 \
  --use-cuda
```

- Accuracy: **94.35%**
- Correct predictions: **1,887 / 2,000**
- Incorrect predictions: **113 / 2,000**
- GPU inference time: approximately **31 seconds**
- Throughput: approximately **63.58 clips/second**

## Comparison

| Evaluation | Accuracy | Difference from local run |
|---|---:|---:|
| Local zero-shot run | **94.35%** | - |
| Paper/repository zero-shot CLAP | 93.90% | **+0.45 percentage points** |
| Paper's best zero-shot benchmark | 94.80% | -0.45 percentage points |
| Paper's fine-tuned ESC-50 model | 98.25% | -3.90 percentage points |

The direct comparison is the local zero-shot result against the paper's 93.90%
zero-shot result. The local run is nine correct predictions higher on the
2,000-clip dataset. The 98.25% result is not directly comparable because the
paper obtained it by fine-tuning the audio encoder on ESC-50.

## Sources

- [Natural Language Supervision for General-Purpose Audio Representations](https://arxiv.org/abs/2309.05767)
- [Microsoft CLAP repository](https://github.com/microsoft/CLAP)
