# In the name of Allah
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech Separation using DPTNet
Author: Kaituo XU (original)
Refactored:2024 by Hadi Alizadeh
"""

import argparse
import os
from collections import OrderedDict
import numpy as np
import soundfile as sf
import torch

from data_2spk import EvalDataLoader, EvalDataset
from models import DPTNet_base as DPTNet
from utils import remove_pad


def parse_arguments():
    parser = argparse.ArgumentParser(description='Separate speech using DPTNet')

    # Model architecture
    parser.add_argument('--N', type=int, default=64, help='Number of filters in autoencoder')
    parser.add_argument('--L', type=int, default=2, help='Length of window in autoencoder')
    parser.add_argument('--H', type=int, default=4, help='Number of heads in multi-head attention')
    parser.add_argument('--K', type=int, default=250, help='Segment size')
    parser.add_argument('--B', type=int, default=6, help='Number of repeats')

    # File paths
    parser.add_argument('--model_path', type=str, default='exp/temp/temp_best.pth.tar',
                        help='Path to trained model file')
    parser.add_argument('--mix_dir', type=str, default=None,
                        help='Directory containing mixture WAV files')
    parser.add_argument('--mix_json', type=str, default='data/tt/mix.json',
                        help='JSON file listing mixture WAV files')
    parser.add_argument('--out_dir', type=str, default='exp/result',
                        help='Output directory for separated WAV files')

    # Execution settings
    parser.add_argument('--use_cuda', type=int, default=0, help='Use GPU (1) or CPU (0)')
    parser.add_argument('--sample_rate', type=int, default=8000, help='Audio sample rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    return parser.parse_args()


def load_model(args):
    """Load DPTNet model from checkpoint."""
    model = DPTNet(args.N, 2, args.L, args.H, args.K, args.B)

    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.model_path, map_location='cuda' if args.use_cuda else 'cpu')
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)

    model.eval()
    return model


def write_audio(file_path, audio, sample_rate):
    """Write audio to file."""
    sf.write(file_path, audio, sample_rate)


def compute_correlation(input_audio, estimate_source):
    """
    Compute Pearson correlation coefficient between input audio
    and estimated sources.
    """
    input_np = input_audio.squeeze().cpu().numpy()
    estimate_np = estimate_source.squeeze().cpu().numpy()

    normalized_input = (input_np - np.mean(input_np)) / (np.std(input_np) + 1e-8)
    correlations = []

    if estimate_np.ndim == 1:
        estimate_np = estimate_np[np.newaxis, :]

    for channel_data in estimate_np:
        normalized_channel = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
        corr = np.corrcoef(normalized_input, normalized_channel)[0, 1]
        correlations.append(corr)

    return correlations


@torch.no_grad()
def separate(args):
    """Run source separation on given mixtures."""
    if not args.mix_dir and not args.mix_json:
        raise ValueError("Must provide either --mix_dir or --mix_json.")

    # Prepare model and data
    model = load_model(args)
    dataset = EvalDataset(args.mix_dir, args.mix_json,
                          batch_size=args.batch_size,
                          sample_rate=args.sample_rate)
    loader = EvalDataLoader(dataset, batch_size=1)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, (mixture, mix_lengths, filenames) in enumerate(loader):
        if args.use_cuda:
            mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()

        # Forward pass
        estimate_source = model(mixture)
        corrs = compute_correlation(mixture, estimate_source)
        print(f"[{i + 1}] Correlations: " + " | ".join(f"S{j + 1}: {corr:.4f}" for j, corr in enumerate(corrs)))

        # Remove padding
        flat_estimate = remove_pad(estimate_source, mix_lengths)
        mixture = remove_pad(mixture, mix_lengths)

        # Write outputs
        for j, filename in enumerate(filenames):
            base_name = os.path.splitext(os.path.basename(filename))[0]
            out_base = os.path.join(args.out_dir, base_name)

            write_audio(out_base + '_mix.wav', mixture[j], args.sample_rate)
            for c, source in enumerate(flat_estimate[j], 1):
                write_audio(out_base + f'_s{c}.wav', source, args.sample_rate)


def main():
    args = parse_arguments()
    print(f"Running DPTNet separation with args:\n{args}\n")
    separate(args)


if __name__ == '__main__':
    main()
