# In the name of Allah
# Train
# Created: 2018/12 by Kaituo XU
# Edited: 2024 by Hadi Alizadeh

import os
import argparse

import torch

from data_2spk import AudioDataLoader, AudioDataset
from data_3spk import AudioDataset3

from solver import Solver
from models import DPTNet_base as DPTNet
from torch.utils.data import ConcatDataset, Subset
import random

parser = argparse.ArgumentParser(
    "Dual-path transformer"
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir2', type=str, default='data/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir2', type=str, default='data/cv',
                    help='directory including mix.json, s1.json and s2.json')

parser.add_argument('--train_dir3', type=str, default='data/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir3', type=str, default='data/cv',
                    help='directory including mix.json, s1.json and s2.json')

parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=64, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--C', default=2, type=int,
                    help='Maximum number of speakers')
parser.add_argument('--L', default=4, type=int,
                    help='Length of window in autoencoder') # L=2 in paper
parser.add_argument('--H', default=4, type=int,
                    help='Number of head in Multi-head attention')
parser.add_argument('--K', default=250, type=int,
                    help='segment size')
parser.add_argument('--B', default=6, type=int,
                    help='Number of repeats')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=110, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=3, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=0.125, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='../exp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='../exp/temp_best.pth.tar',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='../exp/final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=1000, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')


def main(args):
    # Construct Solver
    # data2
    tr_dataset2 = AudioDataset(args.train_dir2, args.batch_size,
                              sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset2 = AudioDataset(args.valid_dir2, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
   
    #### data3
    tr_dataset3 = AudioDataset3(args.train_dir3, args.batch_size,
                              sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset3 = AudioDataset3(args.valid_dir3, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio

    combined_dataset = ConcatDataset([tr_dataset3, tr_dataset2])
    indices = list(range(len(combined_dataset)))
    random.shuffle(indices)
    shuffled_subset = Subset(combined_dataset, indices)
    shuffled_dataset = ConcatDataset([shuffled_subset])
    tr_loader3 = AudioDataLoader(shuffled_dataset, batch_size=1,
                                shuffle=True,
                                num_workers=args.num_workers)

    combined_dataset = ConcatDataset([cv_dataset2, cv_dataset3])
    indices = list(range(len(combined_dataset)))
    random.shuffle(indices)
    shuffled_subset = Subset(combined_dataset, indices)
    shuffled_dataset = ConcatDataset([shuffled_subset])
    cv_loader3 = AudioDataLoader(shuffled_dataset, batch_size=1,
                                shuffle=True,
                                num_workers=args.num_workers)                        


    data = {'tr_loader': tr_loader3, 'cv_loader': cv_loader3}

    # model
    model = DPTNet(enc_dim=64, feature_dim=64, hidden_dim=128, layer=6, segment_size=250, nspk=2, win_len=2)
    print(model)
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"]='0'
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9,lr=args.lr,
                                      weight_decay=args.l2)

    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

