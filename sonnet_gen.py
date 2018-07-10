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
