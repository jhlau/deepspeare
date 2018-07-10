###preprocessing options###
word_minfreq=3

###hyper-parameters###
seed=0
batch_size=32
keep_prob=0.7
epoch_size=30
max_grad_norm=5
#language model
word_embedding_dim=100
word_embedding_model="pretrain_word2vec/dim100/word2vec.bin"
lm_enc_dim=200
lm_dec_dim=600
lm_dec_layer_size=1
lm_attend_dim=25
lm_learning_rate=0.2
#pentameter model
char_embedding_dim=150
pm_enc_dim=50
pm_dec_dim=200
pm_attend_dim=50
pm_learning_rate=0.001
repeat_loss_scale=1.0
cov_loss_scale=1.0
cov_loss_threshold=0.7
sigma=1.00
#rhyme model
rm_dim=100
rm_neg=5 #extra randomly sampled negative examples
rm_delta=0.5
rm_learning_rate=0.001

###sonnet hyper-parameters###
bptt_truncate=2 #number of sonnet lines to truncate bptt
doc_lines=14 #total number of lines for a sonnet

###misc###
verbose=False
save_model=True

###input/output###
output_dir="output_tmp"
train_data="datasets/gutenberg/sonnet_train.txt"
valid_data="datasets/gutenberg/sonnet_valid.txt"
test_data="datasets/gutenberg/sonnet_test.txt"
output_prefix="wmin%d_sd%d_bat%d_kp%.1f_eph%d_grd%d_wdim%d_lmedim%d_lmddim%d_lmdlayer%d_lmadim%d_lmlr%.1f_cdim%d_pmedim%d_pmddim%d_pmadim%d_pmlr%.1E_loss%.1f-%.1f-%.1f_sm%.2f_rmdim%d_rmn%d_rmd%.1f_rmlr%.1E_son%d-%d" % \
    (word_minfreq, seed, batch_size, keep_prob, epoch_size, max_grad_norm, word_embedding_dim, lm_enc_dim,
    lm_dec_dim, lm_dec_layer_size, lm_attend_dim, lm_learning_rate,
    char_embedding_dim, pm_enc_dim, pm_dec_dim, pm_attend_dim, pm_learning_rate, repeat_loss_scale,
    cov_loss_scale, cov_loss_threshold, sigma, rm_dim, rm_neg, rm_delta, rm_learning_rate, bptt_truncate, doc_lines)
