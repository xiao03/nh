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
        self.rec = nn.Linear(2*self.hidden_size, 7*self.hidden_size)
        # self.w_r = nn.Parameter(torch.empty((2*self.hidden_size, 7*self.hidden_size)).uniform_(-1, 1))
        # self.b_r = nn.Parameter(torch.zeros(7*self.hidden_size))
        # self.w_a = nn.Parameter(torch.empty((self.hidden_size, self.type_size)).uniform_(-1, 1))
        self.wa = nn.Linear(self.hidden_size, self.type_size)
        self.emb = nn.Embedding(self.type_size+1, self.hidden_size)
        # self.emb_event = nn.Parameter(torch.empty((self.type_size + 1, self.hidden_size)).uniform_(-1, 1))

    def init_states(self, batch_size):
        self.h_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c_bar = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        # B * 2H
        (gate_i,
        gate_f,
        gate_z,
        gate_o,
        gate_i_bar,
        gate_f_bar,
        gate_delta) = torch.chunk(self.rec(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        # print(delta_t.size(), duration_t.size())
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
            torch.exp(-delta_t * duration_t.view(-1,1))

        h_d_t = o_t * torch.tanh(c_d_t)

        return c_d_t, h_d_t
    
    def forward(self, event_seqs, duration_seqs, batch_first = True):
        if batch_first:
            event_seqs = event_seqs.transpose(0,1)
            duration_seqs = duration_seqs.transpose(0,1)
        
        batch_size = event_seqs.size()[1]
        batch_length = event_seqs.size()[0]

        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []

        for t in range(batch_length):
            self.init_states(batch_size)
            c, self.c_bar, o_t, delta_t = self.recurrence(self.emb(event_seqs[t]), self.h_d, self.c_d, self.c_bar)
            _, h_p = self.decay(c, self.c_bar, o_t, delta_t, torch.zeros(1))
            self.c_d, self.h_d = self.decay(c, self.c_bar, o_t, delta_t, duration_seqs[t])
            h_list.append(self.h_d)
            c_list.append(c)
            c_bar_list.append(self.c_bar)
            o_list.append(o_t)
            delta_list.append(delta_t)
        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        c_bar_seq = torch.stack(c_bar_list)
        o_seq = torch.stack(o_list)
        delta_seq = torch.stack(delta_list)
        
        # 5 x L x B x H
        self.output = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))
        # if batch_first:
        #     self.output = self.transpose(1,2)
        return self.output
    
    def _forward(self, event, duration):
        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []
        seq_length = len(event)

        for t in range(seq_length):
            self.c, self.c_bar, o_t, delta_t = self.recurrence(self.embedding_event(event[t]), self.h_d, self.c_d, self.c_bar)
            _, h_p = self.decay(self.c, self.c_bar, o_t, delta_t, 0)
            self.c_d, self.h_d = self.decay(self.c, self.c_bar, o_t, delta_t, duration[t])
            h_list.append(h_p)
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

    def log_likelihood(self, event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, batch_first=True):
        ''' Calculate log likelihood per sequence
        '''

        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        # out_shape = (h.size()[1], h.size()[2])
        # L * B * H
        h = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate term 1 from original state tensors
        # Ignore <BOS> event.
        # lambda_k = F.softplus(self.wa(h))

        # seq_length_select = torch.arange(1, seqs_length)
        # seq_event_select = event_seq[1:].long()
        original_loglikelihood = 0.0
        # lambda_k = lambda_k.transpose(0, 1)
        # print(lambda_k.size())
        for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
            # seq_event_select = event_seqs[idx][1:].long()
            lambda_k = F.softplus(self.wa(h[1:seq_len+1, idx, :]))
            
            # print('event', event_seq[1:seq_len+1].shape)
            # print('lambda_k shape', lambda_k[ torch.arange(seq_len).long(), event_seq[1:seq_len+1]].shape)
            original_loglikelihood += torch.sum(torch.log( 
                                                     lambda_k[ torch.arange(seq_len).long() , event_seq[1:seq_len+1]]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        if batch_first:
            sim_time_seqs = sim_time_seqs.transpose(0,1)
        for idx, sim_duration in enumerate(sim_time_seqs):
            _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        # sim_lambda_k = F.softplus(self.wa(h_d)).transpose(0,1)
        simulated_likelihood = 0.0
        for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, seqs_length)):
            mc_coefficient = total_time / (seq_len)
            sim_lambda_k = F.softplus(self.wa(h_d[:seq_len,idx,:]))
            simulated_likelihood += mc_coefficient * torch.sum(torch.sum(sim_lambda_k))

        loglikelihood = original_loglikelihood - simulated_likelihood
        print('Term 1:\t{}\nTerm 3:\t{}'.format(original_loglikelihood, simulated_likelihood))
        return loglikelihood
    
    def _log_likelihood(self, event_seq, sim_time_seq, sim_index_seq, total_time):
        ''' Calculate log likelihood per sequence
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
        lambda_k = F.softplus(self.wa(h))
        row_select = torch.arange(1, len(event_seq))
        column_select = event_seq[1:].long()

        original_loglikelihood = torch.sum(torch.log(1e-9 + 
                                                     lambda_k[row_select, column_select]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        for idx, sim_duration in enumerate(sim_time_seq):
            _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        sim_lambda_k = F.softplus(self.wa(h_d))
        mc_coefficient = total_time / (len(event_seq) - 1)
        simulated_likelihood = mc_coefficient * torch.sum(torch.sum(sim_lambda_k))

        loglikelihood = original_loglikelihood - simulated_likelihood

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
    # for i, sample in enumerate(train_dataloader):
    #     if i == 1:
    #         event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample, model.type_size)
    #         sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs)
    
    
    for i_batch, sample_batched in enumerate(train_dataloader):
        if i_batch == 1:
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            # event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            print(event_seqs, time_seqs, total_time_seqs, seqs_length)
            print(event_seqs.size(), time_seqs.size(), total_time_seqs.size())
            print(sim_time_seqs, sim_index_seqs)
            print(sim_time_seqs.size(), sim_index_seqs.size())
            output = model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)
            print(likelihood)
                # batch_output.append(output)
            print(output.size())

