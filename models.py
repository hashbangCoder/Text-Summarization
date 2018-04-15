import torch
from torch.autograd import Variable
from torch.nn import LSTM, GRU, Linear, LSTMCell, Module, DataParallel
import torch.nn.functional as F
import pdb
from numpy import inf
import numpy as np
# use_gpu = True


# idea similar to https://github.com/abisee/pointer-generator/blob/master/beam_search.py
class Hypothesis(object):
    def __init__(self, token_id, hidden_state, cell_state, log_prob):
        self._h = hidden_state
        self._c = cell_state
        self.log_prob = log_prob
        self.full_prediction = token_id # list
        self.survivability = self.log_prob/ float(len(self.full_prediction))

    def extend(self, token_id, hidden_state, cell_state, log_prob):
        return Hypothesis(token_id= self.full_prediction + [token_id],
                          hidden_state=hidden_state,
                          cell_state=cell_state,
                          log_prob= self.log_prob + log_prob)

# encoder net for the article
class Encoder(Module):
    def __init__(self, input_size, hidden_size, wordEmbed):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.word_embed = wordEmbed
        self.fwd_rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        # self.fwd_rnn = DataParallel(self.fwd_rnn)
        self.bkwd_rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        # self.bkwd_rnn = DataParallel(self.bkwd_rnn)
        # Since we have a bi-directional lstm we need to map from hidden*2 to hidden for decoder inputs
        self.output_cproj = Linear(self.hidden_size * 2, self.hidden_size)
        self.output_cproj = DataParallel(self.output_cproj)
        self.output_hproj = Linear(self.hidden_size * 2, self.hidden_size)
        self.output_hproj = DataParallel(self.output_hproj)

    def init_hidden(self, batch_size):
        """
         Before we've done anything, we dont have any hidden state.
         Refer to the Pytorch documentation to see exactly
         why they have this dimensionality.
         The axes semantics are (num_layers, minibatch_size, hidden_size)
         return (ht,ct)
        """
        var = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
        # if use_gpu:
        #     var = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()),
        #         Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()))
        # else:
        #     var = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
        #         Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))
        return var


    def forward(self, _input, rev_input):
        """ _input/rev_input is batch_size x len """
        batch_size, max_len = _input.size(0), _input.size(1)
        #_input = _input.view(max_len,batch_size)
        #rev_input = rev_input.view(max_len, batch_size)
        embed_fwd = self.word_embed(_input)
        embed_rev = self.word_embed(rev_input)
        #print('####################################################')
        #print(embed_fwd.size())
        #print(_input.size())
        self.hidden = self.init_hidden(batch_size)
        self.hidden_rev = self.init_hidden(batch_size)
        # get mask for location of PAD
        mask = _input.eq(0).detach()

        # Reuse them for both forward and backward
        lstm_hidden = [self.init_hidden(batch_size) for i in range(max_len) ]
        context, _ =  self.init_hidden(batch_size)

        # lstm_hidden_rev = [self.init_hidden(batch_size) for i in range(max_len) ]

        lstm_out = Variable(torch.zeros(batch_size, max_len, self.hidden_size).cuda())
        lstm_out_rev = Variable(torch.zeros(batch_size, max_len, self.hidden_size).cuda())
        # if use_gpu:
        #     lstm_out = Variable(torch.zeros(batch_size, max_len, self.hidden_size).cuda())
        #     #input_embeds = input_embeds.cuda()
        #     #decoder_out = Variable(torch.zeros(len_summary, batch_size, self.hidden_size).cuda())
        # else:
        #     lstm_out = Variable(torch.zeros(batch_size, max_len, self.hidden_size))
        #     #decoder_out = Variable(torch.zeros(len_summary, batch_size, self.hidden_size))

        #FORWARD
        for j in range(max_len):
            #calculate the context
            context = context*0
            if j>0:
                for k in range(int(np.log2(j)) + 1):
                    context = context + lstm_hidden[j-2**k][0]
            #forward pass into the encoder
#             print("input dim: {}, hidden_length: {}, context_len: {}".format(input_embeds[j].dim(), self.hidden[0].dim(), context.dim()))
            #print("##############", embed_fwd[:,j,:])
            out, self.hidden = self.fwd_rnn(embed_fwd[:,j,:].contiguous().view(batch_size,1, -1), (context, self.hidden[1]) )
            lstm_out[:,j,:] = out
            lstm_hidden[j] = self.hidden
        # encoder_hidden = self.hidden
        # decoder_hidden = self.init_hidden()
        # decoder_hidden = decoder_hidden + encoder_hidden

        fwd_out = lstm_out
        fwd_state = self.hidden
        # fwd_out, fwd_state = self.fwd_rnn(embed_fwd)

        #BACKWARD
        for j in range(max_len):
            #calculate the context
            context = context*0
            if j>0:
                for k in range(int(np.log2(j)) + 1):
                    context = context + lstm_hidden[j-2**k][0]
            #forward pass into the encoder
            #print("input dim: {}, hidden_length: {}, context_len: {}".format(input_embeds[j].dim(), self.hidden[0].dim(), context.dim()))
            #print("##############", embed_fwd[:,j,:])
            out, self.hidden_rev = self.bkwd_rnn(embed_rev[:,j,:].contiguous().view(batch_size,1, -1), (context, self.hidden_rev[1]) )
            lstm_out_rev[:,j,:] = out
            lstm_hidden[j] = self.hidden_rev
        
        bkwd_out = lstm_out_rev
        bkwd_state = self.hidden_rev
        # bkwd_out, bkwd_state = self.bkwd_rnn(embed_rev)
        #print("$$$$$$$$",fwd_out.size(), bkwd_out.size())
        hidden_cat = torch.cat((fwd_out, bkwd_out), 2)

        # inverse of mask
        inv_mask = mask.eq(0).unsqueeze(2).expand(batch_size, max_len, self.hidden_size * 2).float().detach()
        hidden_out = hidden_cat * inv_mask
        final_hidd_proj = self.output_hproj(torch.cat((fwd_state[0].squeeze(0), bkwd_state[0].squeeze(0)), 1))
        final_cell_proj = self.output_cproj(torch.cat((fwd_state[1].squeeze(0), bkwd_state[1].squeeze(0)), 1))

        return hidden_out, final_hidd_proj, final_cell_proj, mask

# TODO Enhancement: Project input embedding with previous context vector for current input
class PointerAttentionDecoder(Module):
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed):
        super(PointerAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embed = wordEmbed

        #self.decoderRNN = LSTMCell(self.input_size, self.hidden_size)
        self.decoderRNN = LSTM(self.input_size, self.hidden_size, batch_first=True)
        # self.decoderRNN = DataParallel(self.decoderRNN)
        #params for attention
        self.Wh = Linear(2 * self.hidden_size, 2*self. hidden_size)
        self.Wh = DataParallel(self.Wh)
        self.Ws = Linear(self.hidden_size, 2*self.hidden_size)
        self.Ws = DataParallel(self.Ws)
        self.w_c = Linear(1, 2*self.hidden_size)
        self.w_c = DataParallel(self.w_c)
        self.v = Linear(2*self.hidden_size, 1)
        self.v = DataParallel(self.v)

        # parameters for p_gen
        self.w_h = Linear(2 * self.hidden_size, 1)    # double due to concat of BiDi encoder states
        self.w_s = Linear(self.hidden_size, 1)
        self.w_x = Linear(self.input_size, 1)
        self.w_h = DataParallel(self.w_h)
        self.w_s = DataParallel(self.w_s)
        self.w_x = DataParallel(self.w_x)

        #params for output proj
        self.V = Linear(self.hidden_size * 3, self.vocab_size)
        self.V = DataParallel(self.V)
        self.min_length = 40

    def setValues(self, start_id, stop_id, unk_id, beam_size, max_decode=40, lmbda=1):
        # start/stop tokens
        self.start_id = start_id
        self.stop_id = stop_id
        self.unk_id = unk_id
        self.max_decode_steps = max_decode
        # max_article_oov -> max number of OOV in articles i.e. enc inputs. Will be set for each batch individually
        self.max_article_oov = None
        self.beam_size = beam_size
        self.lmbda = lmbda

    def forward(self, enc_states, enc_final_state, enc_mask, _input, article_inds, targets, decode=False):
        """
        enc_states -> output states of encoder
        enc_final_state -> final output of encoder
        enc_mask -> mask indicating location of PAD in encoder input
        _input -> decoder inputs
        article_inds -> modified encoder input with temporary OOV ids for each OOV token
        targets -> decoder targets
        decode -> Boolean flag for train/eval mode
        """

        if decode is True:
            return self.decode(enc_states, enc_final_state, enc_mask, article_inds)

        batch_size, max_enc_len, enc_size = enc_states.size()
        max_dec_len = _input.size(1)
        # coverage initially zero
        coverage =  Variable(torch.zeros(batch_size, max_enc_len).cuda())
        dec_lens = (_input > 0).float().sum(1)
        state = enc_final_state[0].unsqueeze(0),enc_final_state[1].unsqueeze(0)

        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)
        embed_input = self.word_embed(_input)

        lm_loss, cov_loss = [], []
        hidden, _ = self.decoderRNN(embed_input, state)

        # step through decoder hidden states
        for _step in range(max_dec_len):
            _h = hidden[:, _step, :]
            target = targets[:, _step].unsqueeze(1)

            dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
            cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
            e_t = self.v(F.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))

            # mask to -INF before applying softmax
            attn_scores = e_t.view(batch_size, max_enc_len)
            del e_t
                        #print(attn_scores,enc_mask.data)
            #attn_scores.masked_fill_(enc_mask, float('-inf'))
            attn_scores.masked_fill_(enc_mask, -1e30)
            attn_scores = F.softmax(attn_scores)

            context = attn_scores.unsqueeze(1).bmm(enc_states).squeeze(1)
            p_vocab =     F.softmax(self.V(torch.cat((_h, context), 1)))                                    #output proj calculation
            p_gen = F.sigmoid(self.w_h(context) + self.w_s(_h) + self.w_x(embed_input[:, _step, :]))    # p_gen calculation
            p_gen = p_gen.view(-1, 1)
            weighted_Pvocab = p_gen * p_vocab
            weighted_attn = (1-p_gen)* attn_scores

            if self.max_article_oov > 0:
                ext_vocab = Variable(torch.zeros(batch_size, self.max_article_oov).cuda())                #create OOV (but in-article) zero vectors
                combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
                del ext_vocab
            else:
                combined_vocab = weighted_Pvocab

            del weighted_Pvocab
            assert article_inds.data.min() >=0 and article_inds.data.max() <= (self.vocab_size+ self.max_article_oov), 'Recheck OOV indexes!'

            # scatter article word probs to combined vocab prob.
            # subtract one to account for 0-index
            article_inds_masked = article_inds.add(-1).masked_fill_(enc_mask, 0)
            combined_vocab = combined_vocab.scatter_add(1, article_inds_masked, weighted_attn)

            # mask the output to account for PAD
            # subtract one from target for 0-index
            target_mask_0 = target.ne(0).detach()
            target_mask_p = target.eq(0).detach()
            target = target - 1
            output = combined_vocab.gather(1, target.masked_fill_(target_mask_p, 0))
            lm_loss.append(output.log().mul(-1) * target_mask_0.float())

            coverage = coverage + attn_scores

            # Coverage Loss
            # take minimum across both attn_scores and coverage
            _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
            cov_loss.append(_cov_loss.sum(1))

        # add individual losses
        total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda*torch.stack(cov_loss, 1).sum(1).div(dec_lens)
        return total_masked_loss

    def decode_step(self, enc_states, state, _input, enc_mask, article_inds):
        # decode for one step with beam search
        # for first step, batch_size =1
        # successive steps batch_size = beam_size
        batch_size, max_enc_len, enc_size = enc_states.size()

        # coverage initially zero
        coverage =  Variable(torch.zeros(batch_size, max_enc_len).cuda())

        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)
        embed_input = self.word_embed(_input)

        _h, _c = self.decoderRNN(embed_input, state)[1]
        _h = _h.squeeze(0)
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        e_t = self.v(F.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
                #attn_scores.masked_fill_(enc_mask, -float('Inf'))
        attn_scores.masked_fill_(enc_mask, -1e30)
        attn_scores = F.softmax(attn_scores)

        context = attn_scores.unsqueeze(1).bmm(enc_states)
        p_vocab =     F.softmax(self.V(torch.cat((_h, context.squeeze(1)), 1)))                            # output proj calculation
        p_gen = F.sigmoid(self.w_h(context.squeeze(1)) + self.w_s(_h) + self.w_x(embed_input[:, 0, :]))    # p_gen calculation
        p_gen = p_gen.view(-1, 1)
        weighted_Pvocab = p_gen * p_vocab
        weighted_attn = (1-p_gen)* attn_scores

        if self.max_article_oov > 0:
            ext_vocab = Variable(torch.zeros(batch_size, self.max_article_oov).cuda())                    # create OOV (but in-article) zero vectors
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        assert article_inds.data.min() >=0 and article_inds.data.max() <= (self.vocab_size+ self.max_article_oov), 'Recheck OOV indexes!'

        # scatter article word probs to combined vocab prob.
        # subtract one to account for 0-index
        combined_vocab = combined_vocab.scatter_add(1, article_inds.add(-1), weighted_attn)
        return combined_vocab, _h, _c.squeeze(0)

    def getOverallTopk(self, vocab_probs, _h, _c, all_hyps, results):
        # return top-k values i.e. top-k over all beams i.e. next step input ids
        # return hidden, cell states corresponding to topk
        probs, inds = vocab_probs.topk(k=self.beam_size, dim=1)
        probs = probs.log().data
        inds = inds.data
        inds.add_(1)
        candidates = []
        assert len(all_hyps) == probs.size(0), '# Hypothesis and log-prob size dont match'
        # cycle through all hypothesis in full beam
        for i, hypo in enumerate(probs.tolist()):
            for j, _ in enumerate(hypo):
                new_cand = all_hyps[i].extend(token_id=inds[i,j],
                                              hidden_state=_h[i].unsqueeze(0),
                                              cell_state=_c[i].unsqueeze(0),
                                              log_prob= probs[i,j])
                candidates.append(new_cand)
        # sort in descending order
        candidates = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        new_beam, next_inp = [], []
        next_h, next_c = [], []
        #prune hypotheses and generate new beam
        for h in candidates:
            if h.full_prediction[-1] == self.stop_id:
                # weed out small sentences that likely have no meaning
                if len(h.full_prediction)>=self.min_length:
                    results.append(h.full_prediction)
            else:
                new_beam.append(h)
                next_inp.append(h.full_prediction[-1])
                next_h.append(h._h.data)
                next_c.append(h._c.data)
            if len(new_beam) >= self.beam_size:
                break
        assert len(new_beam) >= 1, 'Non-existent beam'
        return new_beam, torch.LongTensor([next_inp]), results, torch.cat(next_h, 0), torch.cat(next_c, 0)

    # Beam Search Decoding
    def decode(self, enc_states, enc_final_state, enc_mask, article_inds):
        _input = Variable(torch.LongTensor([[self.start_id]]).cuda(), volatile=True)
        init_state = enc_final_state[0].unsqueeze(0),enc_final_state[1].unsqueeze(0)
        decoded_outputs = []
        # all_hyps --> list of current beam hypothesis. start with base initial hypothesis
        all_hyps = [Hypothesis([self.start_id], None, None, 0)]
        # start decoding
        for _step in range(self.max_decode_steps):
            # ater first step, input is of batch_size=curr_beam_size
            # curr_beam_size <= self.beam_size due to pruning of beams that have terminated
            # adjust enc_states and init_state accordingly
            curr_beam_size = _input.size(0)
            beam_enc_states = enc_states.expand(curr_beam_size, enc_states.size(1), enc_states.size(2)).contiguous().detach()
            beam_article_inds = article_inds.expand(curr_beam_size, article_inds.size(1)).detach()

            vocab_probs, next_h, next_c = self.decode_step(beam_enc_states, init_state, _input, enc_mask, beam_article_inds)

            # does bulk of the beam search
            # decoded_outputs --> list of all ouputs terminated with stop tokens and of minimal length
            all_hyps, decode_inds, decoded_outputs, init_h, init_c = self.getOverallTopk(vocab_probs, next_h, next_c, all_hyps, decoded_outputs)

            # convert OOV words to unk tokens for lookup
            decode_inds.masked_fill_((decode_inds > self.vocab_size), self.unk_id)
            decode_inds = decode_inds.t()
            _input = Variable(decode_inds.cuda(), volatile=True)
            init_state = (Variable(init_h.unsqueeze(0), volatile=True), Variable(init_c.unsqueeze(0), volatile=True))

        non_terminal_output = [item.full_prediction for item in all_hyps]
        all_outputs = decoded_outputs + non_terminal_output
        return all_outputs

class SummaryNet(Module):
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed, start_id, stop_id, unk_id, beam_size=4, max_decode=40, lmbda=1):
        super(SummaryNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(self.input_size, self.hidden_size, wordEmbed)
        self.pointerDecoder = PointerAttentionDecoder(self.input_size, self.hidden_size, vocab_size, wordEmbed)
        self.pointerDecoder.setValues(start_id, stop_id, unk_id, beam_size, max_decode, lmbda)

    def forward(self, _input, max_article_oov, decode_flag=False):
        """input -> (batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets), max_article_oov"""
        # set num article OOVs in decoder
        self.pointerDecoder.max_article_oov = max_article_oov
        # decode/eval code
        if decode_flag:
            enc_input, rev_enc_input, article_inds = _input
            enc_states, enc_hn, enc_cn, enc_mask = self.encoder(enc_input, rev_enc_input)
            model_summary = self.pointerDecoder(enc_states, (enc_hn, enc_cn), enc_mask, None, article_inds, targets=None, decode=True)
            return model_summary
        else:
        # train code
            enc_input, article_inds, rev_enc_input, dec_input, dec_target = _input
            enc_states, enc_hn, enc_cn, enc_mask = self.encoder(enc_input, rev_enc_input)
            total_loss = self.pointerDecoder(enc_states, (enc_hn, enc_cn), enc_mask, dec_input, article_inds, targets=dec_target)
            return total_loss
