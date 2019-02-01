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

        # 2 * hidden_size x 7 * hidden_size
        self.w_r = torch.empty((2*self.hidden_size, 7*self.hidden_size)).uniform_(-1, 1)
        # 1 x 7 * hidden_size
        self.b_r = torch.zeros(7*self.hidden_size)
        # hidden_size x type_size
        self.w_a = torch.empty((self.hidden_size, self.type_size)).uniform_(-1, 1)
        self.emb_event = torch.empty((self.type_size + 1, self.hidden_size)).uniform_(-1, 1)

        # self.c_d_t = torch.zeros(self.hidden_size, dtype=torch.float)
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

        output = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))

        return output


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

    
    
    for i_batch, sample_batched in enumerate(train_dataloader):
        if i_batch == 1:
            event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs)
            batch_output = []
            for idx, (event_seq, time_seq) in enumerate(zip(event_seqs, time_seqs)):
                output = model.forward(event_seq, time_seq)
                # batch_output.append(output)
                print(output.size())

