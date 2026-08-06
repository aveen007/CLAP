"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

import argparse
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from msclap import CLAP

from esc50_dataset import ESC50


def parse_args():
    default_dataset_root = Path(__file__).resolve().parents[2] / 'ESC-50'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=default_dataset_root,
        help='Path to an existing ESC-50 checkout (dataset downloading is disabled).',
    )
    parser.add_argument('--use-cuda', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    print(f'Using ESC-50 dataset: {dataset_root}')
    dataset = ESC50(root=dataset_root, download=False)
    prompt = 'this is the sound of '
    labels = [prompt + label for label in dataset.classes]

    clap_model = CLAP(version='2023', use_cuda=args.use_cuda)
    text_embeddings = clap_model.get_text_embeddings(labels)

    predictions, targets = [], []
    for i in tqdm(range(len(dataset))):
        audio_path, _, one_hot_target = dataset[i]
        audio_embeddings = clap_model.get_audio_embeddings([audio_path], resample=True)
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        prediction = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        predictions.append(prediction)
        targets.append(one_hot_target.detach().cpu().numpy())

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    accuracy = accuracy_score(np.argmax(targets, axis=1), np.argmax(predictions, axis=1))
    print(f'ESC50 Accuracy: {accuracy}')


if __name__ == '__main__':
    main()

"""
The output:

ESC50 Accuracy: 93.9%

"""
