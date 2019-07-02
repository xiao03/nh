import pickle

for name in ['train', 'dev', 'test']:
    with open('data/{}.pkl'.format(name), 'wb') as w:
        with open('data/snhpm2d32k4_format/{}.pkl'.format(name), 'rb') as r:
            seqs = pickle.load(r, encoding='latin1')
            data={'train': [], 'dev': [], 'test': []}
            for seq in seqs:
                new_seq = []
                event_seq = seq['seq']
                last_time = 0
                for event in event_seq:
                    new_event = {'type_event': event['event_type'], 'time_since_last_event': event['time'] - last_time}
                    last_time = event['time']
                    new_seq.append(new_event)
                data[name].append(new_seq)
            pickle.dump(data, w)

for name in ['train', 'dev', 'test']:
    with open('data/{}.pkl'.format(name), 'rb') as r:
        seqs = pickle.load(r, encoding='latin1')[name]
        for seq in seqs:
            print(seq)
            import ipdb; ipdb.set_trace()


                

