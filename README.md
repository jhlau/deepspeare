# Requirements
- python2.7
- tensorflow 0.12
   - CPU: `pip install tensorflow==0.12.0`
   - GPU: `pip install tensorflow-gpu==0.12.0`
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
- It was trained on 34M Gutenberg poetry data: [download link](https://ibm.box.com/s/yj38zwrk21q584y1y9qkjt1huf5nepuu)
- If you want to train your own word embeddings, you can use the python script (uses gensim's word2vec)
   * `python pretrain_word2vec.py`

# Training the Sonnet Model

1. Extract the data; it should produce the train/valid/test splits
   * `cd datasets/gutenberg; tar -xvzf data.tgz`
1. Unzip the pre-trained word2vec model
   * `gunzip pretrain_word2vec/dim100/*`
1. Set up model hyper-parameters and other settings, which are all defined in **config.py**
   * the default configuration is the optimal configuration used in the paper (documented [here](http://anthology.aclweb.org/attachments/P/P18/P18-1181.Notes.pdf))
1. Run `python sonnet_train.py`
   * takes about 2-3 hours on a single K80 GPU to train 30 epochs

# Generating Sonnet Quatrain

1. Extract the trained model
   * `cd trained_model; tar -xvzf model.tgz`
1. Run `python sonnet_gen.py -m trained_model`
   * the default configuration is the generation configuration used in the paper
   * takes about a minute to generate one quatrain on CPU (GPU not necessary)

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

# Crowdflower and Expert Evaluation

- Annotations can be found in the folder: evaluation_annotation/

# Media Coverage
- [New Scientist](https://www.newscientist.com/article/2175301-ai-creates-shakespearean-sonnets-and-theyre-actually-quite-good/) (or view the full article [here](media_coverage/new_scientist.jpg))
- [Times UK](https://www.thetimes.co.uk/article/computers-produce-poetry-by-the-meter-vk80077zl) (or view the full article without subscription [here](http://htmlpreview.github.io/?https://github.com/jhlau/deepspeare/blob/master/media_coverage/uk_times.html))
- [Daily Mail](http://www.dailymail.co.uk/sciencetech/article-6000619/Can-spot-real-Shakespeare-sonnet-AI-learns-write-poetry.html)
- [Digital Trends](https://www.digitaltrends.com/cool-tech/ai-generates-shakespearean-sonnets/)
- [NVIDIA](https://news.developer.nvidia.com/ai-sonnet-writing-poet-resembles-shakespeare/)
- [ABC Central News Now, Texas Affiliate](https://bit.ly/2M5s0zg)
- [InfoSurHoy](http://infosurhoy.com/cocoon/saii/xhtml/en_GB/health/can-you-spot-the-real-shakespeare-sonnet-ai-learns-how-write-its-own-poetry/)
- [BBC Radio 4 (from 1:45:20 to 1:49:20)](https://www.bbc.co.uk/programmes/b0bcddwc) (or download extracted segment [here](media_coverage/bbc-radio4-20180801.m4a))
- [Replubbica (Italian)](http://www.repubblica.it/scienze/2018/07/31/news/shakespeare_fatti_da_parte_i_sonetti_li_scrive_l_ai-203094778/?utm_source=dlvr.it&utm_medium=twitter)
- [Datanami](https://www.datanami.com/2018/08/01/deep-speare-emulates-the-bard-with-ai/)
- [TechXplore](https://techxplore.com/news/2018-08-sonnet-shakespeare.html)
- [RankRed](https://www.rankred.com/ai-writes-its-own-poetry/)
- [EndGadget](https://www.engadget.com/2018/08/10/Ai-sonnets-shakespeare/)
- [The New York Post](https://nypost.com/2018/08/08/researchers-trained-robots-to-write-poetry/)
- [Dazed](http://www.dazeddigital.com/science-tech/article/40985/1/artificial-intelligence-ai-poetry-sonnet-shakespeare)
- [The Poetry Society](https://poetrysociety.org.uk/news/you-need-a-bit-of-unpredictability-poetry-society-director-judith-palmer-on-ai-poetry-on-talkradio/)
- [IEEE Spectrum](https://spectrum.ieee.org/artificial-intelligence/machine-learning/this-ai-poet-mastered-rhythm-rhyme-and-natural-language-to-write-like-shakespeare)
- [Passw0rd, Creativity and AI](https://www.mixcloud.com/FI_PassW0rd/passw0rd-creativity-and-ai/) (starts at around the 43 minute mark)

# Publication

Jey Han Lau, Trevor Cohn, Timothy Baldwin, Julian Brooke and Adam Hammond (2018). [Deep-speare: A joint neural model of poetic language, meter and rhyme](http://aclweb.org/anthology/P18-1181) ([Supplementary Material](https://www.aclweb.org/anthology/attachments/P18-1181.Notes.pdf)). In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), Melbourne, Australia, pp. 1948--1958.

# Talk

_Creativity, Machine and Poetry_ for a [public forum on language](https://art-museum.unimelb.edu.au/events/language/) [[video](https://www.youtube.com/watch?v=cHUIFKhPPyo)]
