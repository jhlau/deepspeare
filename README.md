# Requirements
- python2.7
- tensorflow 0.12
- nltk, cmudict and stopwords
   - `import nltk; nltk.download("cmudict"); nltk.download("stopwords")`
- gensim
   - `pip install gensim`
- sklearn
   - `pip install sklearn`
- numpy
   - `pip install numpy`

# Data / Models

- datasets/gutenberg/data.tgz: sonnet data, with train/valid/test splits
- pretrain_word2vec/dim100/*: pre-trained word2vec model
- trained_model/model.tgz: trained sonnet model

# Pre-training Word Embeddings

- The pre-trained word2vec model has already been supplied: pretrain_word2vec/dim100/*
- The pre-trained word2vec model is trained on 34M Gutenberg poetry data: [download link](https://ibm.box.com/s/yj38zwrk21q584y1y9qkjt1huf5nepuu)
- If you want to train your own word embeddings, you can use the python script (uses gensim's word2vec)
   * `python pretrain_word2vec.py`

# Training the Sonnet Model

1. Extract the data; it should produce the train/valid/test splits
   * `cd datasets/gutenberg; tar -xvzf data.tgz`
1. Unzip the pre-trained word2vec model
   * `gunzip pretrain_word2vec/dim100/*`
1. Set up model hyper-parameters and other settings, which are all defined in **config.py**
   * the default configuration is the optimal configuration used in the paper
1. Run `python sonnet_train.py`

# Generating Sonnet Quatrain

1. Extract the trained model
   * `cd trained_model; tar -xvzf model.tgz`
1. Run `python sonnet_gen.py -m trained_model`
   * the default configuration is the generation configuration used in the paper

```
usage: sonnet_gen.py [-h] -m MODEL_DIR [-n NUM_SAMPLES] [-r RM_THRESHOLD]
                     [-s SENT_SAMPLE] [-a TEMP_MIN] [-b TEMP_MAX] [-d SEED]
                     [-v] [-p SAVE_PICKLE]

Loads a trained model to do generation

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model-dir MODEL_DIR
                        directory of the saved model
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        number of quatrains to generate (default=1)
  -r RM_THRESHOLD, --rm-threshold RM_THRESHOLD
                        rhyme cosine similarity threshold (0=off; default=0.9)
  -s SENT_SAMPLE, --sent-sample SENT_SAMPLE
                        number of sentences to sample from using pentameter
                        loss as sample probability (1=turn off sampling;
                        default=10)
  -a TEMP_MIN, --temp-min TEMP_MIN
                        minimum temperature for word sampling (default=0.6)
  -b TEMP_MAX, --temp-max TEMP_MAX
                        maximum temperature for word sampling (default=0.8)
  -d SEED, --seed SEED  seed for generation (default=1)
  -v, --verbose         increase output verbosity
  -p SAVE_PICKLE, --save-pickle SAVE_PICKLE
                        save samples in a pickle (list of quatrains)
```      

# Generated Quatrains:

```
python sonnet_gen.py -m trained_model/ -d 1

Temperature = 0.6 - 0.8
  01  [0.43]  with joyous gambols gay and still array
  02  [0.44]  no longer when he twas, while in his day
  03  [0.00]  at first to pass in all delightful ways
  04  [0.40]  around him, charming and of all his days
  
  
python sonnet_gen.py -m trained_model/ -d 2
  
Temperature = 0.6 - 0.8
  01  [0.44]  shall i behold him in his cloudy state
  02  [0.00]  for just but tempteth me to stop and pray
  03  [0.00]  a cry: if it will drag me, find no way
  04  [0.40]  from pardon to him, who will stand and wait
  
  
```


# Publication

Lau, Jey Han, Trevor Cohn, Timothy Baldwin, Julian Brooke and Adam Hammond (2018). [Deep-speare: A joint neural model of poetic language, meter and rhyme](XXX). In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), Melbourne, Australia.
