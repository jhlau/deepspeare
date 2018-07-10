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
import tensorflow as tf
from collections import namedtuple
from sonnet_model import SonnetModel
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import cmudict
from util import *

#parser arguments
desc = "Loads a trained model to do some test inferences and generation"
parser = argparse.ArgumentParser(description=desc)

#arguments
parser.add_argument("-m", "--model-dir", required=True, help="directory of the saved model")
parser.add_argument("-n", "--num-samples", default=1, type=int, help="number of quatrains to generate (default=1)")
parser.add_argument("-r", "--rm-threshold", default=0.9, type=float,
    help="rhyme cosine similarity threshold (0=off; default=0.9)")
parser.add_argument("-s", "--sent-sample", default=10, type=int,
    help="number of sentences to sample from using pentameter loss as sample probability (1=turn off sampling; default=10)")
parser.add_argument("-a", "--temp-min", default=0.5, type=float, help="minimum temperature for word sampling (default=0.5)")
parser.add_argument("-b", "--temp-max", default=0.8, type=float, help="maximum temperature for word sampling (default=0.8)")
parser.add_argument("-d", "--seed", default=1, type=int, help="seed for generation (default=1)")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-p", "--save-pickle", help="save samples in a pickle (list of quatrains)")
parser.add_argument("--evaluate_stress", help="evaluate pentameter and print the attention weights given test data")
parser.add_argument("--evaluate_rhyme", help="evaluate rhyme given test data")
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

def reverse_dic(idxvocab):
    vocabxid = {}
    for vi, v in enumerate(idxvocab):
        vocabxid[v] = vi

    return vocabxid

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

    quatrains = []
    #initialise and load model parameters
    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(args.seed)

        with tf.variable_scope("model", reuse=None):
            mtest = SonnetModel(False, cf.batch_size, len(idxword), len(idxchar), charxid[" "], charxid[pad_symbol], cf)

        with tf.variable_scope("model", reuse=True):
            mgen = SonnetModel(False, 1, len(idxword), len(idxchar), charxid[" "], charxid[pad_symbol], cf)

        #load tensorflow model
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.model_dir, "model.ckpt"))

        #print stress attention as given by the pentameter model given test data
        if args.evaluate_stress:

            _, char_data, _, _, _ = load_data(args.evaluate_stress, wordxid, idxword, charxid, idxchar, dummy_symbols)
            char_batch = create_char_batch(char_data, 1, charxid[pad_symbol], mgen.pentameter, idxchar, False)
            attention_data = []

            for bi, b in enumerate(char_batch):
                
                feed_dict = {mgen.pm_enc_x: b[0], mgen.pm_enc_xlen: b[1], mgen.pm_cov_mask: b[2]}
                attentions, costs, logits, mius = sess.run([mgen.pm_attentions, mgen.pm_costs, mgen.pm_logits,
                    mgen.mius], feed_dict)

                attn = np.squeeze(np.array(attentions))
                char = "".join(idxchar[item] for item in b[0][0])
                attention_data.append((char, attn))

                print "\n", "="*100
                print_pm_attention(b, 1, costs, logits, attentions, mius, idxchar)

            cPickle.dump(attention_data, open("summer_attention.pickle", "w"))

        #evaluate rhyme given test data
        if args.evaluate_rhyme:

            _, _, rhyme_data, _, _= load_data(args.evaluate_rhyme, wordxid, idxword, charxid, idxchar, dummy_symbols)

            rhyme_batch = create_rhyme_batch(rhyme_data, cf.batch_size, charxid[pad_symbol], wordxchar, cf.rm_neg, False)

            #variables
            rhyme_thresholds = [0.9, 0.8, 0.7, 0.6]
            cmu              = cmudict.dict()
            cmu_rhyme        = {}
            cmu_norhyme      = {}

            #rhyme predition
            rhyme_pr = {} #precision/recall for each rhyme threshold
            for rt in rhyme_thresholds:
                rhyme_pr[rt] = [[], []]

            for bi, b in enumerate(rhyme_batch):

                num_c       = 3 + cf.rm_neg
                feed_dict   = {mtest.pm_enc_x: b[0], mtest.pm_enc_xlen: b[1], mtest.rm_num_context: num_c}
                cost, attns = sess.run([mtest.rm_cost, mtest.rm_attentions], feed_dict)
                eval_rhyme(rhyme_pr, rhyme_thresholds, cmu, attns, b, idxchar, charxid, pad_symbol, cf,
                    cmu_rhyme, cmu_norhyme)

            #print performance
            for t in rhyme_thresholds:
                p = np.mean(rhyme_pr[t][0])
                r = np.mean(rhyme_pr[t][1])
                f = 2*p*r / (p+r) if (p != 0.0 and r != 0.0) else 0.0
                sys.stdout.write("Test Data Rhyme P/R/F@%.1f  = %.3f / %.3f / %.3f\n" % (t, p, r, f))


            #print top-N mistmatch
            print "\n\nTop-N CMU rhyme mismatch pairs:"
            for k,v in sorted(cmu_rhyme.items(), key=operator.itemgetter(1))[:50]:
                print k, v
            print "\n\nTop-N CMU non-rhyme mismatch pairs:"
            for k,v in sorted(cmu_norhyme.items(), key=operator.itemgetter(1), reverse=True)[:50]:
                print k, v


        #quatrain generation
        for _ in range(args.num_samples):

            #generate some random sentences
            print "\nTemperature =", args.temp_min, "-", args.temp_max
            q, probs = mgen.generate(sess, idxword, idxchar, charxid, wordxchar, wordxid[pad_symbol],
                wordxid[end_symbol], wordxid[unk_symbol], charxid[" "], avoid_symbols, stopwords,
                args.temp_min, args.temp_max, 4, 400, args.sent_sample, args.rm_threshold, args.verbose)
            for line_id, line in enumerate(q):
                print "  %02d  [%.2f]  %s" % (line_id+1, probs[line_id], line)
            sys.stdout.flush()
            quatrains.append(q)

    #save generated samples in a pickle file
    if args.save_pickle and len(quatrains) > 0:
        cPickle.dump(quatrains, open(args.save_pickle, "w"))


if __name__ == "__main__":

    #load config
    cf_dict = cPickle.load(open(os.path.join(args.model_dir, "config.pickle")))
    ModelConfig = namedtuple("ModelConfig", " ".join(cf_dict.keys()))
    cf = ModelConfig(**cf_dict)

    main()
