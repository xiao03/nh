'''
  Utility functions for CTLSTM model
'''

import torch

def generate_sim_time_seqs(time_seqs):
    ''' Generate a simulated time interval sequences from original time interval sequences based on uniform distribution
    
    Args:
        time_seqs: list of torch float tensors
    Results:
        sim_time_seqs: list of torch float tensors
        sim_index_seqs: list of torch long tensors
    '''
    sim_time_seqs, sim_index_seqs = [], []
    # restore_time_seqs, restore_sim_time_seqs = [], []
    for time_seq in time_seqs:
        restore_time_seq = torch.stack([torch.sum(time_seq[0:i]) for i in range(1, len(time_seq)+1)])
        # Generate N-1 time points. Here N includes <BOS>
        restore_sim_time_seq, _ = torch.sort(torch.empty(len(time_seq)-1).uniform_(0, restore_time_seq[-1]))
        
        sim_time_seq = torch.zeros(len(restore_sim_time_seq))
        sim_index_seq = torch.zeros(len(sim_time_seq)).long()

        for idx, t in enumerate(restore_time_seq):
            indices_to_update = restore_sim_time_seq > t

            sim_time_seq[indices_to_update] = restore_sim_time_seq[indices_to_update] - t
            sim_index_seq[indices_to_update] = idx

        # restore_time_seqs.append(restore_time_seq)
        # restore_sim_time_seqs.append(restore_sim_time_seq)
        sim_time_seqs.append(sim_time_seq)
        sim_index_seqs.append(sim_index_seq)

    return sim_time_seqs, sim_index_seqs


if __name__ == '__main__':
    a = torch.tensor([0., 1., 2., 3., 4., 5.])
    b = torch.tensor([0., 2., 4., 6.])
    sim_time_seqs, sim_index_seqs = generate_sim_time_seqs([a,b])
    # print(restore_time_seqs)
    # print(restore_sim_time_seqs)
    print(sim_time_seqs)
    print(sim_index_seqs)

