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

parser = argparse.ArgumentParser()

parser.add_argument("--train-file", dest="train_file", help="Path to train datafile", default='finished_files/train.bin', type=str)
parser.add_argument("--test-file", dest="test_file", help="Path to test/eval datafile", default='finished_files/test.bin', type=str)
parser.add_argument("--vocab-file", dest="vocab_file", help="Path to vocabulary datafile", default='finished_files/vocabulary.bin', type=str)

parser.add_argument("--max-abstract-size", dest="max_abstract_size", help="Maximum size of abstract for decoder input", default=110, type=int)
parser.add_argument("--max-article-size", dest="max_article_size", help="Maximum size of article for encoder input", default=300, type=int)
parser.add_argument("--num-epochs", dest="epochs", help="Number of epochs", default=6, type=int)
parser.add_argument("--batch-size", dest="batchSize", help="Mini-batch size", default=16, type=int)
parser.add_argument("--embed-size", dest="embedSize", help="Size of word embedding", default=256, type=int)
parser.add_argument("--hidden-size", dest="hiddenSize", help="Size of hidden to model", default=128, type=int)

parser.add_argument("--adam", dest="adam", help="adam solver", default=True, type=bool)
parser.add_argument("--lr", dest="lr", help="Learning Rate", default=0.001, type=float)
parser.add_argument("--lambda", dest="lmbda", help="Hyperparameter for auxillary cost", default=1, type=float)
parser.add_argument("--beam-size", dest="beam_size", help="beam size for beam search decoding", default=4, type=int)
parser.add_argument("--max-decode", dest="max_decode", help="Maximum length of decoded output", default=80, type=int)
parser.add_argument("--grad-clip", dest="grad_clip", help="Clip gradients of RNN model", default=2, type=float)
parser.add_argument("--truncate-vocab", dest="trunc_vocab", help="size of truncated Vocabulary <= 50000 [to save memory]", default=50000, type=int)
parser.add_argument("--bootstrap", dest="bootstrap", help="Bootstrap word embeds with GloVe?", default=0, type=int)
parser.add_argument("--print-ground-truth", dest="print_ground_truth", help="Print the article and abstract", default=1, type=int)

parser.add_argument("--eval-freq", dest="eval_freq", help="How frequently (every mini-batch) to evaluate model", default=20000, type=int)
parser.add_argument("--save-dir", dest="save_dir", help="Directory to save trained models", default='Saved-Models/', type=str)
parser.add_argument("--load-model", dest="load_model", help="Directory from which to load trained models", default=None, type=str)

opt = parser.parse_args()
vis = Visdom()

def evalModel(model):
    """
    Code for running model in eval mode
    """
    # Set the model to Evaluation mode as needed by Pytorch (Helpful in Batchnorm and Dropout etc)
    model.eval()
    print '\n\n'
    print '*'*30, ' MODEL EVALUATION ', '*'*30
    # Get one eval sample for testing
    _article, _revArticle,  _extArticle, max_article_oov, article_oov, article_string, abs_string = dl.getEvalBatch()
    _article = Variable(_article.cuda(), volatile=True)
    _extArticle = Variable(_extArticle.cuda(), volatile=True)
    _revArticle = Variable(_revArticle.cuda(), volatile=True)
    all_summaries = model((_article, _revArticle, _extArticle), max_article_oov, decode_flag=True)
    # Set the model back to training mode.
    model.train()
    return all_summaries, article_string, abs_string, article_oov

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

def save_model(net, optimizer,all_summaries, article_string, abs_string):
    """
    Utility code to save model and state to disk
    """
    save_dict = dict({'model': net.state_dict(), 'optim': optimizer.state_dict(), 'epoch': dl.epoch, 'iter':dl.iterInd, 'summaries':all_summaries, 'article':article_string, 'abstract_gold':abs_string})
    print '\n','-' * 60
    print 'Saving Model to : ', opt.save_dir
    save_name = opt.save_dir + 'savedModel_E%d_%d.pth' % (dl.epoch, dl.iterInd)
    torch.save(save_dict, save_name)
    print '-' * 60
    return

assert opt.trunc_vocab <= 50000, 'Invalid value for --truncate-vocab'
assert os.path.isfile(opt.vocab_file), 'Invalid Path to vocabulary file'
with open(opt.vocab_file) as f:
    vocab = pickle.load(f)                                                          #list of tuples of word,count. Convert to list of words
    vocab = [item[0] for item in vocab[:-(5+ 50000 - opt.trunc_vocab)]]             # Truncate vocabulary to conserve memory
vocab += ['<unk>', '<go>', '<end>', '<s>', '</s>']                                  # add special token to vocab to bring total count to 50k

#Create an object of the Dataloader class.
dl = dataloader.dataloader(opt.batchSize, opt.epochs, vocab, opt.train_file, opt.test_file,
                          opt.max_article_size, opt.max_abstract_size)

# TODO: See how nn.Embeddings work and try to use Glove Vectors instead of learning new. Trainable?
if opt.bootstrap:
    # bootstrap with pretrained embeddings
    wordEmbed = torch.nn.Embedding(len(vocab) + 1, 300, 0)
    print 'Bootstrapping with pretrained GloVe word vectors...'
    assert os.path.isfile('embeds.pkl'), 'Cannot find pretrained Word embeddings to bootstrap'
    with open('embeds.pkl', 'rb') as f:
        embeds = pickle.load(f)
    assert wordEmbed.weight.size() == embeds.size()
    wordEmbed.weight.data[1:,:] = embeds
else:
    # learn embeddings from scratch (default)
    wordEmbed = torch.nn.Embedding(len(vocab) + 1, opt.embedSize, 0)
# **************************************************************************************************

print 'Building and initializing SummaryNet...'
net = models.SummaryNet(opt.embedSize, opt.hiddenSize, dl.vocabSize, wordEmbed,
                       start_id=dl.word2id['<go>'], stop_id=dl.word2id['<end>'], unk_id=dl.word2id['<unk>'],
                       max_decode=opt.max_decode, beam_size=opt.beam_size, lmbda=opt.lmbda)
net = net.cuda()
if opt.adam:
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
else:
    optimizer = torch.optim.Adagrad(net.parameters(), lr=opt.lr)

if opt.load_model is not None and os.path.isfile(opt.load_model):
    saved_file = torch.load(opt.load_model)
    net.load_state_dict(saved_file['model'])
    optimizer.load_state_dict(saved_file['optim'])
    dl.epoch = saved_file['epoch']
    dl.iterInd = saved_file['iter']
    dl.pbar.update(dl.iterInd)
    print '\n','*'*30, 'RESUME FROM CHECKPOINT : %s' %opt.load_model,'*'*30

else:
    print '\n','*'*30, 'START TRAINING','*'*30

#dl.iterInd = 287226
#dl.pbar.update(dl.iterInd)
all_loss = []
win = None
### Training loop'
while dl.epoch <= opt.epochs:
    data_batch = dl.getBatch(opt.batchSize)
    batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets, _, _, max_article_oov, article_oov  = data_batch
    # end of training/max epoch reached
    if data_batch is None:
        print '-'*50, 'END OF TRAINING', '-'*50
        break

    batchArticles = Variable(batchArticles.cuda())
    batchExtArticles = Variable(batchExtArticles.cuda())
    batchRevArticles = Variable(batchRevArticles.cuda())
    batchTargets = Variable(batchTargets.cuda())
    batchAbstracts = Variable(batchAbstracts.cuda())

    # Losses for the whole batch. Then do SGD.
    losses = net((batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets), max_article_oov)
    batch_loss = losses.mean()

    # Backpropagation Step
    batch_loss.backward()
    # gradient clipping by norm
    clip_grad_norm(net.parameters(), opt.grad_clip)
    optimizer.step()
    # Flush the gradients.
    optimizer.zero_grad()

    # update loss ticker
    dl.pbar.set_postfix(loss=batch_loss.cpu().data[0])
    dl.pbar.update(opt.batchSize)

    # save losses periodically
    if dl.iterInd % 50:
        all_loss.append(batch_loss.cpu().data.tolist()[0])
        title = 'Residual Logirithmic LSTM'
        if win is None:
            win = vis.line(Y=np.array(all_loss), X=np.arange(1, len(all_loss)+1), opts=dict(title=title, xlabel='#Mini-Batches (x%d)' %(opt.batchSize),
                           ylabel='Train-Loss'))
        vis.line(Y=np.array(all_loss), X=np.arange(1, len(all_loss)+1), win=win, update='replace', opts=dict(title=title, xlabel='#Mini-Batches (x%d)' %(opt.batchSize),
                           ylabel='Train-Loss'))

    # evaluate model periodically
    if dl.iterInd % opt.eval_freq < opt.batchSize and dl.iterInd > opt.batchSize:
        all_summaries, article_string, abs_string, article_oov = evalModel(net)
        displayOutput(all_summaries, article_string, abs_string, article_oov, show_ground_truth=opt.print_ground_truth)

    # Saving the Model : Frequency is 5 times that of Evaluating
    if dl.iterInd % (5*opt.eval_freq) < opt.batchSize and dl.iterInd > opt.batchSize:
        save_model(net, optimizer, all_summaries, article_string, abs_string)

    del batch_loss, batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets
