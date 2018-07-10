import gensim.models as g
import logging
import os

#parameters
output_dir="pretrain_word2vec/dim100"
input_doc="datasets/gutenberg/pretrain.txt"

#main
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docs = g.word2vec.LineSentence(input_doc)
m = g.Word2Vec(docs, size=100, alpha=0.025, window=5, min_count=3, \
    sample=1e-5, workers=16, min_alpha=0.0001, sg=1, hs=0, negative=5, iter=200)

#save model
m.save(output_dir + "/word2vec.bin")
m.wv.save_word2vec_format(output_dir + "/word2vec.txt", binary=False)
