# In the name of Allah
# Loss function for 3 speaker
# Created: 2018/12 by Kaituo XU
# Edited: 2024 by Hadi Alizadeh

from itertools import permutations

import torch
import torch.nn.functional as F

EPS = 1e-8


def cal_loss3(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    loss,remind_source = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - torch.mean(loss)
    #reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss,remind_source#, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    B, C, T = source.size()

    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = T         
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate

    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
 
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C_source]


    max_snr_idx = torch.argmax(pair_wise_si_snr[:,0,:], dim=1) 
 
    remind_source=torch.zeros(B,C-1,T).cuda()
    loss1 = torch.zeros(B,1).cuda()

    for i in range(B):
      c=0
      loss1[i,0] = pair_wise_si_snr[i,0,max_snr_idx[i]]

      for j in range(C):
         if j != max_snr_idx[i]:
            remind_source[i,c,:] = source[i,j,:]  # [B, C, T]
            c=c+1

    ################ loss Remaind  ############################
    s_estimate =  estimate_source[:,1,:]  # [B, T]
    s_target = torch.sum(remind_source,dim=1) # [B, T] # s1+s2+...

    # zero_min
    m_target = torch.sum(s_target,dim=1, keepdim=True)/T
    m_estimate= torch.sum(s_estimate,dim=1, keepdim=True)/T

    s_target = s_target - m_target #[B,T]
    s_estimate = s_estimate - m_estimate #[B,T]

    s_target = s_target.unsqueeze(1).unsqueeze(2)    # [B, 1, C, T]
    s_estimate = s_estimate.unsqueeze(1).unsqueeze(2)         # [B, C, 1, T]

    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C_source]

    loss_remaind = pair_wise_si_snr[:,0,0].unsqueeze(1)


    #loss = loss1+loss2/2

   
    return loss1,loss_remaind,remind_source


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 2, 3, 12
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, 2, T))
    #source[1: :, -3:] = 0
    #estimate_source[:, :, -3:] = 0
    source_lengths = torch.LongTensor([T,T,T+10])

    loss,_ = cal_loss3(source, estimate_source, source_lengths)
    print('loss', loss)
   