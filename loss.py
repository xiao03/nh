'''
Calculate log likelihood from LSTM output and simulated time sequence. Batch-wise

Args:
    lstm_outputs: list of state tensors from lstm network, i.e. [[h, c, c_bar, o, delta]...]
    sim_time_seqs: list of simulated time interval sequences
    sim_index_seqs: list of index sequences

Result:
    loss: minus log likelihood of this batch
'''

def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
    c_d_t = c_bar_t + (c_t - c_bar_t) * torch.exp(-delta_t * duration_t)
    h_d_t = o_t * torch.tanh(c_d_t)

    return h_d_t


def log_likelihood(lstm_output, sim_time_seq, sim_index_seq):
    # Calculate term 1 from known hidden_states, discard <BOS>
    h, c, c_bar, o, delta = torch.chunk(lstm_output, 5, 0)
    
    origin_likelihood = 