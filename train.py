# -*- coding: utf-8 -*-
"""Training code for neural hawkes model."""
import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

import dataloader
import CTLSTM
import utils


def train(settings):
    """Training process."""
    hidden_size = settings['hidden_size']
    type_size = settings['type_size']
    train_path = settings['train_path']
    dev_path = settings['dev_path']
    batch_size = settings['batch_size']
    epoch_num = settings['epoch_num']
    current_date = settings['current_date']

    model = CTLSTM.CTLSTM(hidden_size, type_size)
    optim = opt.Adam(model.parameters())
    train_dataset = dataloader.CTLSTMDataset(train_path)
    dev_dataset = dataloader.CTLSTMDataset(dev_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=dataloader.pad_batch_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=dataloader.pad_batch_fn, shuffle=True)

    last_dev_loss = 0.0
    for epoch in range(epoch_num):
        tic_epoch = time.time()
        epoch_train_loss = 0.0
        epoch_dev_loss = 0.0
        train_event_num = 0
        dev_event_num = 0
        print('Epoch.{} starts.'.format(epoch))
        tic_train = time.time()
        for i_batch, sample_batched in enumerate(train_dataloader):
            tic_batch = time.time()
            
            optim.zero_grad()
            
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)
            batch_event_num = torch.sum(seqs_length)
            batch_loss = -likelihood

            batch_loss.backward()
            optim.step()
            
            toc_batch = time.time()
            if i_batch % 100 == 0:
                print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'.format(epoch, i_batch, likelihood/batch_event_num, toc_batch-tic_batch))
            epoch_train_loss += batch_loss
            train_event_num += batch_event_num

        toc_train = time.time()
        print('---\nEpoch.{} Training set\nTrain Likelihood per event: {:5f} nats\nTrainig Time:{:2f} s'.format(epoch, -epoch_train_loss/train_event_num, toc_train-tic_train))

        tic_eval = time.time()
        for i_batch, sample_batched in enumerate(dev_dataloader):
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs,seqs_length)
            
            dev_event_num += torch.sum(seqs_length)
            epoch_dev_loss -= likelihood

        toc_eval = time.time()
        toc_epoch = time.time()
        print('Epoch.{} Devlopment set\nDev Likelihood per event: {:5f} nats\nEval Time:{:2f}s.\n'.format(epoch, -epoch_dev_loss/dev_event_num, toc_eval-tic_eval))
        
        with open("loss_{}.txt".format(current_date), 'a') as l:
            l.write("Epoch {}:\n".format(epoch))
            l.write("Train Event Number:\t\t{}\n".format(train_event_num))
            l.write("Train Likelihood per event:\t{:.5f}\n".format(-epoch_train_loss/train_event_num))
            l.write("Training time:\t\t\t{:.2f} s\n".format(toc_train-tic_train))
            l.write("Dev Event Number:\t\t{}\n".format(dev_event_num))
            l.write("Dev Likelihood per event:\t{:.5f}\n".format(-epoch_dev_loss/dev_event_num))
            l.write("Dev evaluating time:\t\t{:.2f} s\n".format(toc_eval-tic_eval))
            l.write("Epoch time:\t\t\t{:.2f} s\n".format(toc_epoch-tic_epoch))
            l.write("\n")
        
        gap = epoch_dev_loss/dev_event_num - last_dev_loss
        if abs(gap) < 1e-4:
            print('Final log likelihood: {} nats'.format(-epoch_dev_loss/dev_event_num))
            break
        
        last_dev_loss = epoch_dev_loss/dev_event_num
    
    return


if __name__ == "__main__":
    settings = {
        'hidden_size': 32,
        'type_size': 8,
        'train_path': 'data/train.pkl',
        'dev_path': 'data/dev.pkl',
        'batch_size': 32,
        'epoch_num': 100,
        'current_date': datetime.date.today()
    }

    train(settings)
