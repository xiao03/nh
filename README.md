# A simple Pytorch implementation of neural Hawkes process

This repository is a simple PyTorch implementation of Neural Hawkes Process from paper *[The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)*. 

## Data format
The data can obtain from the author's [GitHub page](https://github.com/HMEIatJHU/neurawkes). Download the pickle files under data folder.

Here is an example of the training set.
```python
{
	'train':[
		[{'type_event': 'A',
		 'time_since_last_event': 1.0},	# First event
		{'type_event': 'B',
		 'time_since_last_event': 1.2},
		{'type_event': 'C',
		 'time_since_last_event': 2.0},], # First sequence

		[{'type_event': 'B',
		 'time_since_last_event': 1.1},
		{'type_event': 'A',
		 'time_since_last_event': 0.6},
		{'type_event': 'C',
		 'time_since_last_event': 2.3},], # Second sequence
	],

	'dev': [], # Only not empty in the development dataset
	'test': [], # Only not empty in the test dataset
}
```


## How to Run
A quick look of the code:
```python
python train.py
```

Please use Pytorch version 1.0 and later.



## What's more
This repository serves as an understanding of the neural Hawkes process, so not all experiments in the paper are implemented and tested. Feel free to take it and implement your own.
