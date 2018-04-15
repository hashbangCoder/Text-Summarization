# Text-Summarization

Text Summarization using Residual Logarithmic LSTMS. The model also uses Pointer Generator Network, Coverage, Reinforcement based learning and Attention Mechanism.

This is based on Pytorch implementation of [Get To The Point: Summarization with Pointer-Generator Networks by See et. al.](https://arxiv.org/abs/1704.04368), modified from [here](https://github.com/hashbangCoder/Text-Summarization)


## Dependencies
You require :

1. [Pytorch](pytorch.org/) v0.2 with CUDA support
2. [Visdom](https://github.com/facebookresearch/visdom/) visualization package for easy monitoring of training progress in web browser
3. tqdm for terminal-level progress updates

## How to get the data

Download the fully-preprocessed data splits [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail) and save yourself the trouble of downloading CoreNLP and subsequent tokenization. 
This implementation only uses the `Finised_Files` directory containing `train.bin`, `val.bin`, `test.bin` splits. Move these files to the `finished_files` directory.

The only other file `vocabulary.bin` has already been provided in the `finished_files` in this repo. 

Alternatively, follow the instructions [here](https://github.com/abisee/cnn-dailymail) to download and preprocess the dataset.
Use the given `preprocess.py` instead script to generate the data splits as this version does not have Tensorflow dependencies. 


## How to run

The original hyperparameters as described in the paper are memory intensive, so this implementation uses a smaller RNN `hidden_size` as the default setting. All other hyperparameters are kept the same.
You can trade-off `vocabulary_size`, `batch_size`, `hidden_size`, `max_abstract_size` and `max_article_size` to achieve your memory budget.

1. Fire up a visdom server :
`python -m visdom.server`

2. To train using default settings, from the repo's root directory :
`CUDA_VISIBLE_DEVICES=0 python main.py`

3. Monitor the training progress by going to `127.0.0.1:8097` in your web browser or the remote URL is you're executing your code on SSH

Configurations can be changed using command line options.

`python main.py --help` to get a list of all options.


The model is evaluated periodically during training on a sample from `test.bin`and decoding is done using beam search. The model is also saved after every epoch in `Saved-Models`


To bootstrap with pre-trained embeddings, you will need to obtain pre-trained Glove/Word2Vec embeddings for words in your vocabulary. OOV words can be assigned a random value. Save this as a Pytorch Tensor `embeds.pkl` and make sure the size of vocabulary matches size of tensor. 
The default setting is initialize with random word embeddings since that has been reported to perform better. 

