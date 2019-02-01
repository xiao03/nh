''' Training code for CTLSTM model
'''
import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

import dataloader
import CTLSTM
import utils

def train(settings):
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
            
            batch_loss = torch.tensor(0.0)
            batch_event_num = 0
            
            optim.zero_grad()
            event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs)
            for idx, (event_seq, time_seq, sim_time_seq, sim_index_seq, total_time) in enumerate(zip(event_seqs, time_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs)):
                # optim.zero_grad()
                model.forward(event_seq, time_seq)
                likelihood = model.log_likelihood(event_seq, sim_time_seq, sim_index_seq, total_time)
                batch_event_num += len(event_seq) - 1
                seq_loss = -likelihood
                # seq_loss.backward()
                batch_loss += seq_loss
            batch_loss.backward(retain_graph=True)
            optim.step()
            
            toc_batch = time.time()
            print('Epoch.{} Batch.{}:\nBatch Loss per event: {:5f} nats\nTrain Time: {:2f} s'.format(epoch, i_batch, batch_loss/batch_event_num, toc_batch-tic_batch))
            epoch_train_loss += batch_loss
            train_event_num += batch_event_num
            if i_batch > 0:
                break
        toc_train = time.time()
        print('---\nEpoch.{} Training set\nTrain Loss per event: {:5f} nats\nTrainig Time:{:2f} s'.format(epoch, epoch_train_loss/train_event_num, toc_train-tic_train))

        tic_eval = time.time()
        for i_batch, sample_batched in enumerate(dev_dataloader):
            event_seqs, time_seqs, total_time_seqs = dataloader.restore_batch(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs)
            for idx, (event_seq, time_seq, sim_time_seq, sim_index_seq, total_time) in enumerate(zip(event_seqs, time_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs)):
                model.forward(event_seq, time_seq)
                likelihood = model.log_likelihood(event_seq, sim_time_seq, sim_index_seq, total_time)
                dev_event_num += len(event_seq) - 1
            epoch_dev_loss -= likelihood
            if i_batch > 0:
                break
        toc_eval = time.time()
        toc_epoch = time.time()
        print('Epoch.{} Devlopment set\nDev Loss per event: {:5f} nats\nEval Time:{:2f}\n'.format(epoch, epoch_dev_loss/dev_event_num, toc_eval-tic_eval))
        
        with open("loss_{}.txt".format(current_date), 'a') as l:
            l.write("Epoch {}:\n".format(epoch))
            l.write("Train Event Number:\t\t{}\n".format(train_event_num))
            l.write("Train loss per event:\t{:.4f}\n".format(epoch_train_loss/train_event_num))
            l.write("Training time:\t\t\t{:.2f} s\n".format(toc_train-tic_train))
            l.write("Dev Event Number:\t\t{}\n".format(dev_event_num))
            l.write("Dev loss per event:\t\t{:.4f}\n".format(epoch_dev_loss/dev_event_num))
            l.write("Dev evaluating time:\t{:.2f} s\n".format(toc_eval-tic_eval))
            l.write("Epoch time:\t\t\t\t{:.2f} s\n".format(toc_epoch-tic_epoch))
            l.write("\n")
        
        gap = epoch_dev_loss/dev_event_num - last_dev_loss
        if epoch > 5 and gap < 1e-4:
            print('Final log likelihood: {} nats'.format(-epoch_dev_loss/dev_event_num))
            break
        
        last_dev_loss = epoch_dev_loss/dev_event_num


        

if __name__ == "__main__":
    settings = {
        'hidden_size': 32,
        'type_size': 5,
        'train_path': 'data/train.pkl',
        'dev_path': 'data/dev.pkl',
        'batch_size': 32,
        'epoch_num': 10,
        'current_date': datetime.date.today()
    }

    train(settings)
