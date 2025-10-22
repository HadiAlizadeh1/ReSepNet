# In the name of Allah
# Evaluate separation performance using DPTNet
# Original Author: yoonsanghyu
# Reference: Kaituo Xu
# Revised by Hadi Alizadeh (2024)


import argparse
from collections import OrderedDict
import numpy as np
import torch
from mir_eval.separation import bss_eval_sources

from data_2spk import AudioDataLoader, AudioDataset
from pit_criterion2 import cal_loss
from models import DPTNet_base as DPTNet
from utils import remove_pad


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate separation performance using DPTNet')
    parser.add_argument('--N', type=int, default=64, help='Number of filters in autoencoder')
    parser.add_argument('--C', type=int, default=2, help='Number of speakers')
    parser.add_argument('--L', type=int, default=2, help='Length of window in autoencoder')
    parser.add_argument('--H', type=int, default=4, help='Number of heads in Multi-head attention')
    parser.add_argument('--K', type=int, default=250, help='Segment size')
    parser.add_argument('--B', type=int, default=6, help='Number of repeats')
    parser.add_argument('--model_path', type=str, default='../Epoch501.pth.tar', help='Path to trained model file')
    parser.add_argument('--data_dir', type=str, default='data/tt', help='Directory with mix.json, s1.json, s2.json')
    parser.add_argument('--cal_sdr', type=int, default=1, help='Calculate SDR (slow operation)')
    parser.add_argument('--use_cuda', type=int, default=1, help='Use GPU if available')
    parser.add_argument('--sample_rate', type=int, default=8000, help='Sample rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    return parser.parse_args()


def load_model(model_path, use_cuda=True):
    """Load DPTNet model from checkpoint."""
    model = DPTNet(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=6,
        segment_size=250,
        nspk=2,
        win_len=2
    )

    if use_cuda:
        model = model.cuda()

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = OrderedDict(
        (k.replace("module.", ""), v)
        for k, v in checkpoint['model_state_dict'].items()
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def simple_correlation(input_audio, estimate_source):
    """Compute correlation coefficients between input audio and estimated sources."""
    input_np = input_audio.squeeze().cpu().numpy()
    estimate_np = estimate_source.squeeze().cpu().numpy()

    normalized_input = (input_np - np.mean(input_np)) / (np.std(input_np) + 1e-8)

    if estimate_np.ndim == 1:
        estimate_np = estimate_np[None, :]

    correlations = []
    for channel in estimate_np:
        norm_ch = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
        corr = np.corrcoef(normalized_input, norm_ch)[0, 1]
        correlations.append(corr)

    return correlations


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi)."""
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, _, _, _ = bss_eval_sources(src_ref, src_est)
    sdr0, _, _, _ = bss_eval_sources(src_ref, src_anchor)
    return np.mean((sdr - sdr0))


def cal_SISNR(ref, est, eps=1e-8):
    """Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR)."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    proj = np.sum(ref * est) * ref / (np.sum(ref ** 2) + eps)
    noise = est - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate SI-SNR improvement (SI-SNRi)."""
    sisnr_mix = [cal_SISNR(src_ref[i], mix) for i in range(2)]
    sisnr_est = [cal_SISNR(src_ref[i], src_est[i]) for i in range(2)]
    return np.mean(np.array(sisnr_est) - np.array(sisnr_mix))


def evaluate(args):
    """Evaluate model performance on given dataset."""
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, use_cuda=(device.type == 'cuda'))
    dataset = AudioDataset(args.data_dir, args.batch_size, sample_rate=args.sample_rate, segment=-1, cv_maxlen=1000)
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    total_SISNRi, total_SDRi, total_cnt, total_cor = 0.0, 0.0, 0, 0

    with torch.no_grad():
        for i, (padded_mix, mix_len, padded_src, _) in enumerate(data_loader):
            padded_mix, mix_len, padded_src = [x.to(device) for x in (padded_mix, mix_len, padded_src)]

            est_src = model(padded_mix)
            est_src2 = model(est_src[0, 1, :].unsqueeze(0))

            cor1 = simple_correlation(padded_mix, est_src)
            cor2 = simple_correlation(est_src[0, 1, :].unsqueeze(0), est_src2)

            core1 = int(any(abs(c) > 0.93 for c in cor1))
            core2 = int(any(abs(c) > 0.93 for c in cor2))

            max_snr = cal_loss(padded_src, est_src, mix_len)
            reordered = torch.cat(
                (est_src[:, max_snr[0], :].unsqueeze(0), est_src[:, max_snr[1], :].unsqueeze(0)), dim=1
            )

            mix = remove_pad(padded_mix, mix_len)
            src = remove_pad(padded_src, mix_len)
            est_src = remove_pad(reordered, mix_len)

            for mix_i, ref_i, est_i in zip(mix, src, est_src):
                total_cnt += 1
                print(f"Utt {total_cnt}")

                ref_i = ref_i.cpu().numpy()
                est_i = est_i.cpu().numpy()
                mix_i = mix_i.cpu().numpy()

                avg_SISNRi = cal_SISNRi(ref_i, est_i, mix_i)
                total_SISNRi += avg_SISNRi
                print(f"\tSI-SNRi = {avg_SISNRi:.2f}")

                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(ref_i, est_i, mix_i)
                    total_SDRi += avg_SDRi
                    print(f"\tSDRi = {avg_SDRi:.2f}")

                if core1 == 0 and core2 == 1:
                    total_cor += 1

                print(f"Total Correct Correlations: {total_cor}")

    # Summary
    if args.cal_sdr and total_cnt > 0:
        print(f"\nAverage SDR improvement: {total_SDRi / total_cnt:.2f}")
        print(f"Average Correlation Count: {total_cor / total_cnt:.2f}")
    print(f"Average SI-SNR improvement: {total_SISNRi / total_cnt:.2f}")


if __name__ == '__main__':
    args = parse_args()
    print(args)
    evaluate(args)
