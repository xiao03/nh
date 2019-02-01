# -*- coding: utf-8 -*-
'''
CTLSTM.py
Created by Xiao Liu on Jan. 31, 2019.

Description:
    A continuous time LSTM network

TODO:
    
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import dataloader
from torch.utils.data import DataLoader


class CTLSTM(nn.Module):
    '''Continuous time LSTM network with decay function.
    '''
    def __init__(self, hidden_size, type_size, batch_first=True):
        ''' We assume input event size is always equal to hidden size.
        '''
        super(CTLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.type_size = type_size
        self.batch_first = batch_first
        self.num_layers = 1

        # Parameters
        self.w_r = nn.Parameter(torch.empty((2*self.hidden_size, 7*self.hidden_size)).uniform_(-1, 1))
        self.b_r = nn.Parameter(torch.zeros(7*self.hidden_size))
        self.w_a = nn.Parameter(torch.empty((self.hidden_size, self.type_size)).uniform_(-1, 1))
        self.emb_event = nn.Parameter(torch.empty((self.type_size + 1, self.hidden_size)).uniform_(-1, 1))

        # State tensors
        self.h_d = torch.zeros(self.hidden_size, dtype=torch.float)
        self.c_bar = torch.zeros(self.hidden_size, dtype=torch.float)
        self.c = torch.zeros(self.hidden_size, dtype=torch.float)

    
    def embedding_event(self, event):
        return self.emb_event[int(event)]

    
    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1))

        (gate_i,
        gate_f,
        gate_z,
        gate_o,
        gate_i_bar,
        gate_f_bar,
        gate_delta) = torch.chunk((torch.matmul(feed, self.w_r) + self.b_r), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z
        h_t = gate_o * torch.tanh(c_t)

        return c_t, c_bar_t, h_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * torch.exp(-delta_t * duration_t)
        h_d_t = o_t * torch.tanh(c_d_t)

        return h_d_t
    
    def forward(self, event, duration):
        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []
        seq_length = len(event)

        for t in range(seq_length):
            self.c, self.c_bar, h_t, o_t, delta_t = self.recurrence(self.embedding_event(event[t]), self.h_d, self.c, self.c_bar)
            self.h_d = self.decay(self.c, self.c_bar, o_t, delta_t, duration[t])
            h_list.append(h_t)
            c_list.append(self.c)
            c_bar_list.append(self.c_bar)
            o_list.append(o_t)
            delta_list.append(delta_t)
        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        c_bar_seq = torch.stack(c_bar_list)
        o_seq = torch.stack(o_list)
        delta_seq = torch.stack(delta_list)

        self.output = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))
        return self.output
    
    def log_likelihood(self, event_seq, sim_time_seq, sim_index_seq, total_time):
        ''' Calculate log likelihood per event
        '''
        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        out_shape = (h.size()[1], h.size()[2])

        h = h.view(out_shape)
        c = c.view(out_shape)
        c_bar = c_bar.view(out_shape)
        o = o.view(out_shape)
        delta = delta.view(out_shape)

        # Calculate term 1 from original state tensors
        # Ignore <BOS> event.
        lambda_k = F.softplus(torch.matmul(h, self.w_a))
        row_select = torch.arange(len(event_seq)-1)
        column_select = event_seq[1:].long()

        original_loglikelihood = torch.sum(torch.log(1e-9 + 
                                                     lambda_k[row_select, column_select]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        for idx, sim_duration in enumerate(sim_time_seq):
            h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        sim_lambda_k = F.softplus(torch.matmul(h_d, self.w_a))
        mc_coefficient = total_time / len(event_seq)
        simulated_loglikelihood = mc_coefficient * torch.sum(torch.sum(sim_lambda_k))

        loglikelihood = original_loglikelihood - simulated_loglikelihood

        return loglikelihood


if __name__ == '__main__':
    # event_seq = [5,1,2,4,0,1,2,0,3,1,0,3,2,4,1]
    # time_seq = [0, 0.1, 0.3, 0.4, 1.2, 1.9, 3.2, 4.8, 4.9, 5.4, 6.3, 6.8, 7.2, 8.2, 9]
    model = CTLSTM(32, 5)
    # output = model.forward(event_seq, time_seq)
    train_dataset = dataloader.CTLSTMDataset('data/train.pkl')
    dev_dataset = dataloader.CTLSTMDataset('data/dev.pkl')

    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=dataloader.pad_batch_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset)

    # for i, sample in enumerate(train_dataset):
    #     print(len(sample['event_seq']))
    for i, sample in enumerate(dev_dataloader):
        if i == 1:
            event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs)
    
    
    # for i_batch, sample_batched in enumerate(train_dataloader):
        # if i_batch == 1:
        #     event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample_batched, model.type_size)
        #     sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs)
        #     batch_output = []
        #     for idx, (event_seq, time_seq, sim_time_seq, sim_index_seq, total_time) in enumerate(zip(event_seqs, time_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs)):
        #         output = model.forward(event_seq, time_seq)
        #         likelihood = model.log_likelihood(event_seq, sim_time_seq, sim_index_seq, total_time)
        #         print(likelihood)
        #         # batch_output.append(output)
        #         print(output.size())

