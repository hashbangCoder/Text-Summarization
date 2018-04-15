import torch
from torch.autograd import Variable
import cPickle as pickle
import argparse
import pdb, os
import numpy as np
import models
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm
import dataloader
from visdom import Visdom
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()

parser.add_argument("--train-file", dest="train_file", help="Path to train datafile", default='finished_files/train.bin', type=str)
parser.add_argument("--test-file", dest="test_file", help="Path to test/eval datafile", default='finished_files/test.bin', type=str)
parser.add_argument("--vocab-file", dest="vocab_file", help="Path to vocabulary datafile", default='finished_files/vocabulary.bin', type=str)

parser.add_argument("--max-abstract-size", dest="max_abstract_size", help="Maximum size of abstract for decoder input", default=110, type=int)
parser.add_argument("--max-article-size", dest="max_article_size", help="Maximum size of article for encoder input", default=300, type=int)
parser.add_argument("--batch-size", dest="batchSize", help="Mini-batch size", default=32, type=int)
parser.add_argument("--embed-size", dest="embedSize", help="Size of word embedding", default=300, type=int)
parser.add_argument("--hidden-size", dest="hiddenSize", help="Size of hidden to model", default=128, type=int)

parser.add_argument("--lambda", dest="lmbda", help="Hyperparameter for auxillary cost", default=1, type=float)
parser.add_argument("--beam-size", dest="beam_size", help="beam size for beam search decoding", default=4, type=int)
parser.add_argument("--max-decode", dest="max_decode", help="Maximum length of decoded output", default=120, type=int)
parser.add_argument("--truncate-vocab", dest="trunc_vocab", help="size of truncated Vocabulary <= 50000 [to save memory]", default=50000, type=int)
parser.add_argument("--bootstrap", dest="bootstrap", help="Bootstrap word embeds with GloVe?", default=0, type=int)
parser.add_argument("--print-ground-truth", dest="print_ground_truth", help="Print the article and abstract", default=1, type=int)

parser.add_argument("--load-model", dest="load_model", help="Directory from which to load trained models", default=None, type=str)
parser.add_argument("--article", dest="article_path", help="Path to article text file", default=None, type=str)
parser.add_argument("--num-eval", dest="num_eval", help="num of times to evaluate", default = 10, type=int)
# Note that the parameters of the saved model should match the ones passed.
opt = parser.parse_args()
vis = Visdom()

assert opt.load_model is not None and os.path.isfile(opt.vocab_file), 'Invalid Path to trained model file'


def displayOutput(all_summaries, article, abstract, article_oov, show_ground_truth=False):
    """
    Utility code for displaying generated abstract/multiple abstracts from beam search
    """
    print '*' * 80
    print '\n'
    if show_ground_truth:
        print 'ARTICLE TEXT : \n', article
        print 'ACTUAL ABSTRACT : \n', abstract
    for i, summary in enumerate(all_summaries):
        # generated_summary = ' '.join([dl.id2word[ind] if ind<=dl.vocabSize else article_oov[ind % dl.vocabSize] for ind in summary])
        try:
            generated_summary = ' '.join([dl.id2word[ind] if ind<=dl.vocabSize else article_oov[ind % dl.vocabSize - 1] for ind in summary])
            print 'GENERATED ABSTRACT #%d : \n' %(i+1), generated_summary
        except:
            print '^^^^^^error in index^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', '\n vocab len :', dl.vocabSize
            print '\n OOV length', len(article_oov), '\n', [ind for ind in summary]
            pass
    print '*' * 80
    return

# Utility code to save model to disk # Not needed here.
# def save_model(net, optimizer,all_summaries, article_string, abs_string):
#     save_dict = dict({'model': net.state_dict(), 'optim': optimizer.state_dict(), 'epoch': dl.epoch, 'iter':dl.iterInd, 'summaries':all_summaries, 'article':article_string, 'abstract_gold':abs_string})
#     print '\n','-' * 60
#     print 'Saving Model to : ', opt.save_dir
#     save_name = opt.save_dir + 'savedModel_E%d_%d.pth' % (dl.epoch, dl.iterInd)
#     torch.save(save_dict, save_name)
#     print '-' * 60  
#     return

assert opt.trunc_vocab <= 50000, 'Invalid value for --truncate-vocab'
assert os.path.isfile(opt.vocab_file), 'Invalid Path to vocabulary file'
with open(opt.vocab_file) as f:
    vocab = pickle.load(f)                                                          #list of tuples of word,count. Convert to list of words
    vocab = [item[0] for item in vocab[:-(5+ 50000 - opt.trunc_vocab)]]             # Truncate vocabulary to conserve memory
vocab += ['<unk>', '<go>', '<end>', '<s>', '</s>']                                  # add special token to vocab to bring total count to 50k

#Create an object of the Dataloader class.
dl = dataloader.dataloader(opt.batchSize, None, vocab, opt.train_file, opt.test_file, 
                          opt.max_article_size, opt.max_abstract_size, test_mode=True)


wordEmbed = torch.nn.Embedding(len(vocab) + 1, opt.embedSize, 0)
print 'Building SummaryNet...'
net = models.SummaryNet(opt.embedSize, opt.hiddenSize, dl.vocabSize, wordEmbed,
                       start_id=dl.word2id['<go>'], stop_id=dl.word2id['<end>'], unk_id=dl.word2id['<unk>'],
                       max_decode=opt.max_decode, beam_size=opt.beam_size, lmbda=opt.lmbda)
net = net.cuda()

print 'Loading weights from file...might take a minute...'
saved_file = torch.load(opt.load_model)
net.load_state_dict(saved_file['model'])
print '\n','*'*30, 'LOADED WEIGHTS FROM MODEL FILE : %s' %opt.load_model,'*'*30
    
############################################################################################
# Set model to eval mode
############################################################################################
net.eval()
print '\nSetting Model to Evaluation Mode\n'

# Run num_eval times to get num_eval random test data samples for output
for _ in range(opt.num_eval):
    # If article file provided
    if opt.article_path is not None and os.path.isfile(opt.article_path):
        with open(opt.article_path,'r') as f:
            article_string = f.read().strip()
            article_tokenized = word_tokenize(article_string)
            print article_tokenized
        _article, _revArticle,  _extArticle, max_article_oov, article_oov = dl.getInputTextSample(article_tokenized)
        abs_string = 'Fresh Article : **No abstract available**'
    else:
    # pull random test sample
        data_batch = dl.getEvalSample()
        _article, _revArticle,  _extArticle, max_article_oov, article_oov, article_string, abs_string = dl.getEvalSample()

    _article = Variable(_article.cuda(), volatile=True)
    _extArticle = Variable(_extArticle.cuda(), volatile=True)
    _revArticle = Variable(_revArticle.cuda(), volatile=True)    
    all_summaries = net((_article, _revArticle, _extArticle), max_article_oov, decode_flag=True)

    displayOutput(all_summaries, article_string, abs_string, article_oov, show_ground_truth=opt.print_ground_truth)

# TODO: Evalute for ROUGE Scores
