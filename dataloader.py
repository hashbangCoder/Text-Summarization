import cPickle as pickle
import os, random,pdb, torch
from itertools import groupby
from tqdm import tqdm
import pdb

class dataloader():
    def __init__(self, batchSize, epochs, vocab, train_path, test_path, max_article_size=400, max_abstract_size=140):
        self.maxEpochs = epochs
        self.epoch = 1
        self.batchSize = batchSize
        self.iterInd = 0
        self.globalInd = 1
        #self.vocab = vocab      #list of all vocabulary words
        self.word2id, self.id2word = self.getVocabMap(vocab)
        self.vocabSize = len(vocab)
        self.max_article_size = max_article_size
        self.max_abstract_size = max_abstract_size

        assert os.path.isfile(train_path) and os.path.isfile(test_path), 'Invalid paths to train/test datafiles'
        self.train_path = train_path
        self.test_path = test_path
        
        print 'Loading training data from disk...will take a minute...'
        with open(self.train_path,'rb') as f:
            self.train_data = pickle.load(f)                
        self.trainSamples = len(self.train_data)

        #Load eval set
        print 'Loading eval data from disk...'
        with open(self.test_path,'rb') as f:
            self.test_data = pickle.load(f)                
        self.testSamples = len(self.test_data)

        # self.loadEvalBatch()                
        self.stopFlag = False
        self.pbar = tqdm(total=self.trainSamples * self.maxEpochs)
        self.pbar.set_description('Epoch : %d/%d' % (self.epoch, self.maxEpochs))

    def getVocabMap(self, vocab):
        word2id, id2word = {}, {}
        for i, word in enumerate(vocab):            
            word2id[word] = i+1 # reserve 0 for pad
            id2word[i+1] = word
        id2word[0] = ''
        # word2id['<go>'] = len(word2id) + 1
        # id2word[len(word2id)] = '<go>'
        # word2id['<end>'] = len(word2id) + 1
        # id2word[len(word2id)] = '<end>'        
        return word2id, id2word

    def makeEncoderInput(self, article):
        # tokenize article
        # list of OOV words in article
        self.encUnkCount = 1
        _intArticle, extIntArticle = [], []
        article_oov = []
        art_len = min(self.max_article_size, len(article))
        for word_ind, word in enumerate(article[:art_len]):
            try:
                _intArticle.append(self.word2id[word.lower().strip()])                
                extIntArticle.append(self.word2id[word.lower().strip()])                
            except KeyError:
                _intArticle.append(self.word2id['<unk>'])        
                extIntArticle.append(self.vocabSize + self.encUnkCount)        
                article_oov.append(word)
                #article_oov_ind.append(word_ind)
                self.encUnkCount += 1            
        
        return _intArticle, extIntArticle, article_oov, art_len

    def makeDecoderInput(self, abstract, article_oov):
        _intAbstract, extIntAbstract = [], []
        abs_len = min(self.max_abstract_size, len(abstract)) 
        # tokenize abstract
        self.decUnkCount = 0
        for word in abstract[:abs_len]:
            try:
                _intAbstract.append(self.word2id[word.lower().strip()])
                extIntAbstract.append(self.word2id[word.lower().strip()])                    
            except KeyError:
                _intAbstract.append(self.word2id['<unk>'])                
                #check if OOV word present in article
                if word in article_oov:
                    extIntAbstract.append(self.vocabSize + article_oov.index(word) + 1)                    
                else:
                    extIntAbstract.append(self.word2id['<unk>'])
                self.decUnkCount += 1
        return _intAbstract, extIntAbstract, abs_len

    def preproc(self, samples):   
        # batchArticles --> tensor batch of articles with ids
        # batchRevArticles --> tensor batch of reversed articles with ids
        # batchExtArticles --> tensor batch of articles with ids and <unk> replaced by temp OOV ids
        # batchAbstracts --> tensor batch of abstract (input for decoder) with ids
        # batchTargets --> tensor batch of target abstracts
        # max_article_oov --> max number of OOV tokens in article batch


        # limit max article size to 400 tokens     
        extIntArticles, intRevArticles, intAbstract, intTargets, extIntAbstracts = [], [], [], [], []
        art_lens, abs_lens= [], []
        maxLen = 0
        max_article_oov = 0
        for sampl in samples:            
            article = sampl['article'].split(' ')
            abstract = sampl['abstract'].split(' ')
            # get article and abstract int-tokenized
            _intArticle, _extIntArticle, article_oov, art_len = self.makeEncoderInput(article)
            if max_article_oov < len(article_oov):
                max_article_oov = len(article_oov)
            _intRevArticle = list(reversed(_intArticle))
            _intAbstract, _extIntAbstract, abs_len = self.makeDecoderInput(abstract, article_oov)

            # append stopping/start tokens and increment length by 1
            intAbstract.append([self.word2id['<go>']] + _intAbstract)            
            # append end token        
            intTargets.append(_extIntAbstract + [self.word2id['<end>']])                
            abs_len += 1            
            extIntArticles.append(_extIntArticle)            
            intRevArticles.append(_intRevArticle)
            art_lens.append(art_len)
            abs_lens.append(abs_len)            
        
        padExtArticles = [torch.LongTensor(item + [0] * (max(art_lens) - len(item))) for item in extIntArticles]        
        padRevArticles = [torch.LongTensor(item + [0] * (max(art_lens) - len(item))) for item in intRevArticles]        
        padAbstracts = [torch.LongTensor(item + [0] * (max(abs_lens) - len(item))) for item in intAbstract]
        padTargets = [torch.LongTensor(item + [0] * (max(abs_lens) - len(item))) for item in intTargets]  

        batchExtArticles = torch.stack(padExtArticles, 0)
        # replace temp ids with unk token id for enc input
        batchArticles = batchExtArticles.clone().masked_fill_((batchExtArticles > self.vocabSize), self.word2id['<unk>'])
        batchRevArticles = torch.stack(padRevArticles, 0)
        batchAbstracts = torch.stack(padAbstracts, 0)
        batchTargets = torch.stack(padTargets, 0)        
        art_lens = torch.LongTensor(art_lens)
        abs_lens = torch.LongTensor(abs_lens)        
        return batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets, art_lens, abs_lens, max_article_oov, article_oov       

    def getBatch(self, num_samples=None):
        if num_samples is None:
            num_samples = self.batchSize

        if self.epoch > self.maxEpochs:
            print 'Maximum Epoch Limit reached'
            self.stopFlag = True
            return None
        
        if self.iterInd + num_samples - 1 > self.trainSamples:
            data = [self.train_data[i] for i in xrange(self.iterInd, self.trainSamples)]
        else:
            data = [self.train_data[i] for i in xrange(self.iterInd, self.iterInd + num_samples)]
        
        batchData = self.preproc(data)
        
        self.globalInd += 1
        self.iterInd += num_samples
        if self.iterInd > self.trainSamples:
            self.iterInd = 0            
            self.epoch += 1
            self.globalInd = 1
            self.pbar.set_description('Epoch : %d/%d' % (self.epoch, self.maxEpochs))

        return batchData

    def getEvalBatch(self, num_samples=1):
        # select first sample for eval        
        data = [self.test_data[i] for i in range(num_samples)]
        batchData = self.evalPreproc(data[0])
        return batchData   

    def evalPreproc(self, sample):   
        # sample length = 1
        # limit max article size to 400 tokens     
        extIntArticles, intRevArticles = [], []
        max_article_oov = 0        
        article = sample['article'].split(' ')                  
        # get article  int-tokenized
        _intArticle, _extIntArticle, article_oov, _ = self.makeEncoderInput(article)
        if max_article_oov < len(article_oov):
            max_article_oov = len(article_oov)
        _intRevArticle = list(reversed(_intArticle))
        # _intAbstract, _extIntAbstract, abs_len = self.makeDecoderInput(abstract, article_oov)
        
        extIntArticles.append(_extIntArticle)            
        intRevArticles.append(_intRevArticle)
        
        padExtArticles = [torch.LongTensor(item) for item in extIntArticles]        
        padRevArticles = [torch.LongTensor(item) for item in intRevArticles]                

        batchExtArticles = torch.stack(padExtArticles, 0)
        # replace temp ids with unk token id for enc input
        batchArticles = batchExtArticles.clone().masked_fill_((batchExtArticles > self.vocabSize), self.word2id['<unk>'])
        batchRevArticles = torch.stack(padRevArticles, 0)
                
        return batchArticles, batchRevArticles, batchExtArticles, max_article_oov, article_oov, sample['article'], sample['abstract']