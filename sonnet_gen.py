# -*- coding: utf-8 -*-

"""
Author:         Jey Han Lau
Date:           Aug 17
"""


import argparse
import os
import cPickle
import sys
import random
import codecs
import numpy as np
import multiprocessing
from collections import namedtuple
from sonnet_model import SonnetModel
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import cmudict
from util import *

#parser arguments
desc = "Loads a trained model to do generation"
parser = argparse.ArgumentParser(description=desc)

#arguments
parser.add_argument("-m", "--model-dir", required=True, help="directory of the saved model")
parser.add_argument("-n", "--num-samples", default=1, type=int, help="number of quatrains to generate (default=1)")
parser.add_argument("-r", "--rm-threshold", default=0.9, type=float,
    help="rhyme cosine similarity threshold (0=off; default=0.9)")
parser.add_argument("-s", "--sent-sample", default=10, type=int,
    help="number of sentences to sample from using pentameter loss as sample probability (1=turn off sampling; default=10)")
parser.add_argument("-a", "--temp-min", default=0.6, type=float, help="minimum temperature for word sampling (default=0.6)")
parser.add_argument("-b", "--temp-max", default=0.8, type=float, help="maximum temperature for word sampling (default=0.8)")
parser.add_argument("-d", "--seed", default=1, type=int, help="seed for generation (default=1)")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-p", "--save-pickle", help="save samples in a pickle (list of quatrains)")
args = parser.parse_args()

#constants
pad_symbol = "<pad>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, end_symbol, unk_symbol]
custom_stopwords = [ "thee", "thou", "thy", "'d", "'s", "'ll", "must", "shall" ]

###########
#functions#
###########

# -- quatrain generation
def generate(input_queue, output_queue, idxword, idxchar, charxid, wordxchar, pad_symbol_id,
    end_symbol_id, unk_symbol_id, space_id, avoid_symbols, stopwords, temp_min, temp_max, max_lines, max_words,
    sent_sample, rm_threshold, verbose=False):


    def reset():
        rhyme_aabb  = ([None, 0, None, 2], [None, None, 1, None])
        rhyme_abab  = ([None, None, 0, 1], [None, 0, None, None])
        rhyme_abba  = ([None, None, 1, 0], [None, 0, None, None])
        rhyme_pttn  = random.choice([rhyme_aabb, rhyme_abab, rhyme_abba])
        x           = [[end_symbol_id]]
        xchar       = [wordxchar[end_symbol_id]]
        xchar_len   = [1]
        sonnet      = []
        sent_probs  = []
        last_words  = []
        total_words = 0
        total_lines = 0
        
        #get zero state for decoder
        input_queue.put(1)
        state = output_queue.get()
        prev_state  = state
        
        return state, prev_state, x, xchar, xchar_len, sonnet, sent_probs, last_words, total_words, total_lines, \
            rhyme_pttn[0], rhyme_pttn[1]

    end_symbol = idxword[end_symbol_id]
    sent_temp  = 0.1 #sentence sampling temperature

    state, prev_state, x, xchar, xchar_len, sonnet, sent_probs, last_words, total_words, total_lines, \
        rhyme_pttn_pos, rhyme_pttn_neg = reset()

    #verbose prints during generation
    if verbose:
        sys.stdout.write("  Number of generated lines = 0/4\r")
        sys.stdout.flush()

    while total_words < max_words and total_lines < max_lines:

        #add history context
        if len(sonnet) == 0 or sonnet.count(end_symbol_id) < 1:
            hist = [[unk_symbol_id] + [pad_symbol_id]*5]
        else:
            hist = [sonnet + [pad_symbol_id]*5]
        hlen = [len(hist[0])]

        """
        print "\n\n", "="*80
        print "state =", state[0][1][0][:10]
        print "prev state =", prev_state[0][1][0][:10]
        print "x =", x, idxword[x[0][0]]
        print "hist =", hist, " ".join(idxword[item] for item in hist[0])
        print "sonnet =", sonnet, " ".join(idxword[item] for item in sonnet)
        """

        #get rhyme targets for the 'first' word
        rm_target_pos, rm_target_neg = None, None
        if rhyme_pttn_pos[total_lines] != None:
            rm_target_pos = last_words[rhyme_pttn_pos[total_lines]]
        if rhyme_pttn_neg[total_lines] != None:
            rm_target_neg = last_words[rhyme_pttn_neg[total_lines]]

        #genereate N sentences and sample from them (using softmax(-one_pl) as probability)
        all_sent, all_state, all_pl = [], [], []

        for _ in range(sent_sample):
            input_queue.put((state, x, hist, hlen, xchar, xchar_len,
                avoid_symbols, stopwords, temp_min, temp_max, unk_symbol_id, pad_symbol_id, end_symbol_id, space_id,
                idxchar, charxid, idxword, wordxchar, rm_target_pos, rm_target_neg, rm_threshold, last_words, max_words))

        for _ in range(sent_sample):
            one_sent, one_state, one_pl = output_queue.get()
            if one_sent != None:
                all_sent.append(one_sent)
                all_state.append(one_state)
                all_pl.append(-one_pl)

        #unable to generate half of the required sentences; reset whole quatrain
        if len(all_sent) < sent_sample/2:

            state, prev_state, x, xchar, xchar_len, sonnet, sent_probs, last_words, total_words, total_lines, \
                rhyme_pttn_pos, rhyme_pttn_neg = reset()

        else:

            ix = np.argsort(all_pl)
            all_sent = [all_sent[i] for i in ix]
            all_state = [all_state[i] for i in ix]
            all_pl = [all_pl[i] for i in ix]

            #convert pm_loss to probability using softmax
            probs = np.exp(np.array(all_pl)/sent_temp)
            probs = probs.astype(np.float64) #convert to float64 for higher precision
            probs = probs / math.fsum(probs)

            #sample a sentence
            sent_id = np.argmax(np.random.multinomial(1, probs, 1))
            sent    = all_sent[sent_id]
            state   = all_state[sent_id]
            pl      = all_pl[sent_id]

            total_words += len(sent)
            total_lines += 1
            prev_state   = state

            sonnet.extend(sent)
            sent_probs.append(-pl)
            last_words.append(sent[0])

        if verbose:
            sys.stdout.write("  Number of generated lines = %d/4\r" % (total_lines))
            sys.stdout.flush()

    #postprocessing
    sonnet = sonnet[:-1] if sonnet[-1] == end_symbol_id else sonnet
    sonnet = [ postprocess_sentence(item) for item in \
        " ".join(list(reversed([ idxword[item] for item in sonnet ]))).strip().split(end_symbol) ]

    return sonnet, list(reversed(sent_probs))

def reverse_dic(idxvocab):
    vocabxid = {}
    for vi, v in enumerate(idxvocab):
        vocabxid[v] = vi

    return vocabxid

#########
#classes#
#########
#we parallelise the sentence generation here; each SentenceGenerator will load the trained model individually
class SentenceGenerator(multiprocessing.Process):

    def __init__(self, input_queue, output_queue, proc_id, seed, idxword, idxchar, charxid, pad_symbol, model_dir):

        multiprocessing.Process.__init__(self)
        self.input_queue  = input_queue
        self.output_queue = output_queue
        self.proc_id      = proc_id
        self.seed         = seed
        self.idxword      = idxword
        self.idxchar      = idxchar
        self.charxid      = charxid
        self.pad_symbol   = pad_symbol
        self.model_dir    = model_dir


    def run(self):

        import tensorflow as tf

        #initialise and load model parameters
        with tf.Graph().as_default(), tf.Session() as sess:
            tf.set_random_seed(self.seed + self.proc_id + 1)
            np.random.seed(self.seed + self.proc_id + 1)

            with tf.variable_scope("model", reuse=None):
                mgen = SonnetModel(False, 1, len(self.idxword), len(self.idxchar),
                    self.charxid[" "], self.charxid[self.pad_symbol], cf)

            #load tensorflow model
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.model_dir, "model.ckpt"))

            while True:
                input = self.input_queue.get()
                if input is None:
                    self.input_queue.task_done()
                    break
                elif input == 1:
                    zero_state = sess.run(mgen.lm_dec_cell.zero_state(1, tf.float32))
                    self.output_queue.put(zero_state)
                    self.input_queue.task_done()
                else:
                    sent, state, pl = mgen.sample_sent(sess, input)
                    self.output_queue.put([sent, state, pl])
                    self.input_queue.task_done()
            sess.close()
            return

######
#main#
######

def main():

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    #set the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    #load the vocabulary
    idxword, idxchar, wordxchar = cPickle.load(open(os.path.join(args.model_dir, "vocabs.pickle")))
    wordxid = reverse_dic(idxword)
    charxid = reverse_dic(idxchar)

    #symbols to avoid for generation
    avoid_symbols = ["(", ")", "“", "‘", "”", "’", "[", "]"]
    avoid_symbols = [ wordxid[item.decode("utf-8")] for item in avoid_symbols ]
    stopwords = set([ wordxid[item] for item in (nltk_stopwords.words("english") + custom_stopwords) if item in wordxid ])


    #initialise the sentence generation threads
    sg_list      = []
    input_queue  = multiprocessing.JoinableQueue()
    output_queue = multiprocessing.JoinableQueue()

    for i in range(args.sent_sample):
        sg = SentenceGenerator(input_queue, output_queue, i, args.seed, idxword, idxchar, charxid, pad_symbol,
            args.model_dir)
        sg_list.append(sg)
    for sg in sg_list:
        sg.start()

    #quatrain generation
    quatrains = []
    for _ in range(args.num_samples):

        #generate some random sentences
        print "\nTemperature =", args.temp_min, "-", args.temp_max
        q, probs = generate(input_queue, output_queue, idxword, idxchar, charxid, wordxchar, wordxid[pad_symbol],
            wordxid[end_symbol], wordxid[unk_symbol], charxid[" "], avoid_symbols, stopwords,
            args.temp_min, args.temp_max, 4, 400, args.sent_sample, args.rm_threshold, args.verbose)
        for line_id, line in enumerate(q):
            print "  %02d  [%.2f]  %s" % (line_id+1, probs[line_id], line)
        sys.stdout.flush()
        quatrains.append(q)

    #all done, closing the sentence generation threads
    for i in range(args.sent_sample):
        input_queue.put(None)
    input_queue.join()
    for sg in sg_list:
        sg.join()

    #save generated samples in a pickle file
    if args.save_pickle and len(quatrains) > 0:
        cPickle.dump(quatrains, open(args.save_pickle, "w"))


if __name__ == "__main__":

    #load config
    cf_dict = cPickle.load(open(os.path.join(args.model_dir, "config.pickle")))
    ModelConfig = namedtuple("ModelConfig", " ".join(cf_dict.keys()))
    cf = ModelConfig(**cf_dict)

    main()
