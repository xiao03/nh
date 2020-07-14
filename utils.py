"""Utility functions for CTLSTM model."""

import torch


def generate_sim_time_seqs(time_seqs, seqs_length):
    """Generate a simulated time interval sequences from original time interval sequences based on uniform distribution
    
    Args:
        time_seqs: list of torch float tensors
    Results:
        sim_time_seqs: list of torch float tensors
        sim_index_seqs: list of torch long tensors
    """
    sim_time_seqs = torch.zeros((time_seqs.size()[0], time_seqs.size()[1]-1)).float()
    sim_index_seqs = torch.zeros((time_seqs.size()[0], time_seqs.size()[1]-1)).long()
    restore_time_seqs, restore_sim_time_seqs = [], []
    for idx, time_seq in enumerate(time_seqs):
        restore_time_seq = torch.stack([torch.sum(time_seq[0:i]) for i in range(1,seqs_length[idx]+1)])
        restore_sim_time_seq, _ = torch.sort(torch.empty(seqs_length[idx]-1).uniform_(0, restore_time_seq[-1]))
        
        sim_time_seq = torch.zeros(seqs_length[idx]-1)
        sim_index_seq = torch.zeros(seqs_length[idx]-1).long()

        for idx_t, t in enumerate(restore_time_seq):
            indices_to_update = restore_sim_time_seq > t

            sim_time_seq[indices_to_update] = restore_sim_time_seq[indices_to_update] - t
            sim_index_seq[indices_to_update] = idx_t

        restore_time_seqs.append(restore_time_seq)
        restore_sim_time_seqs.append(restore_sim_time_seq)
        sim_time_seqs[idx, :seqs_length[idx]-1] = sim_time_seq
        sim_index_seqs[idx, :seqs_length[idx]-1] = sim_index_seq

    return sim_time_seqs, sim_index_seqs


def pad_bos(batch_data, type_size):
    event_seqs, time_seqs, total_time_seqs, seqs_length = batch_data
    pad_event_seqs = torch.zeros((event_seqs.size()[0], event_seqs.size()[1]+1)).long() * type_size
    pad_time_seqs = torch.zeros((time_seqs.size()[0], event_seqs.size()[1]+1)).float()

    pad_event_seqs[:, 1:] = event_seqs.clone()
    pad_event_seqs[:, 0] = type_size
    pad_time_seqs[:, 1:] = time_seqs.clone()

    return pad_event_seqs, pad_time_seqs, total_time_seqs, seqs_length


if __name__ == '__main__':
    a = torch.tensor([0., 1., 2., 3., 4., 5.])
    b = torch.tensor([0., 2., 4., 6., 0., 0.])

    sim_time_seqs, sim_index_seqs, restore_time_seqs, restore_sim_time_seqs =\
        generate_sim_time_seqs(torch.stack([a,b]), torch.LongTensor([6,4]))


