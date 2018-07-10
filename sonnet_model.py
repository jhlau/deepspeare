import tensorflow as tf
import numpy as np
import math
import time
from util import *
from collections import Counter
#from rnn_cell import ExtendedMultiRNNCell

class SonnetModel(object):

    def get_last_hidden(self, h, xlen):

        ids = tf.range(tf.shape(xlen)[0])
        gather_ids = tf.concat(1, [tf.expand_dims(ids, 1), tf.expand_dims(xlen-1, 1)])
        return tf.gather_nd(h, gather_ids)


    def gated_layer(self, s, h, sdim, hdim):

        update_w = tf.get_variable("update_w", [sdim+hdim, hdim])
        update_b = tf.get_variable("update_b", [hdim], initializer=tf.constant_initializer(1.0))
        reset_w  = tf.get_variable("reset_w", [sdim+hdim, hdim])
        reset_b  = tf.get_variable("reset_b", [hdim], initializer=tf.constant_initializer(1.0))
        c_w      = tf.get_variable("c_w", [sdim+hdim, hdim])
        c_b      = tf.get_variable("c_b", [hdim], initializer=tf.constant_initializer())

        z = tf.sigmoid(tf.matmul(tf.concat(1, [s, h]), update_w) + update_b)
        r = tf.sigmoid(tf.matmul(tf.concat(1, [s, h]), reset_w) + reset_b)
        c = tf.tanh(tf.matmul(tf.concat(1, [s, r*h]), c_w) + c_b)
        
        return (1-z)*h + z*c


    def selective_encoding(self, h, s, hdim):

        h1 = tf.shape(h)[0]
        h2 = tf.shape(h)[1]
        h_ = tf.reshape(h, [-1, hdim])
        s_ = tf.reshape(tf.tile(s, [1, h2]), [-1, hdim])

        attend_w = tf.get_variable("attend_w", [hdim*2, hdim])
        attend_b = tf.get_variable("attend_b", [hdim], initializer=tf.constant_initializer())

        g = tf.sigmoid(tf.matmul(tf.concat(1, [h_, s_]), attend_w) + attend_b)

        return tf.reshape(h_* g, [h1, h2, -1])


    def __init__(self, is_training, batch_size, word_type_size, char_type_size, space_id, pad_id, cf):

        self.config = cf

        ###########
        #constants#
        ###########
        self.pentameter = [0,1]*5
        self.pentameter_len = len(self.pentameter)

        ##############
        #placeholders#
        ##############

        #language model placeholders
        self.lm_x    = tf.placeholder(tf.int32, [None, None])
        self.lm_xlen = tf.placeholder(tf.int32, [None])
        self.lm_y    = tf.placeholder(tf.int32, [None, None])
        self.lm_hist = tf.placeholder(tf.int32, [None, None])
        self.lm_hlen = tf.placeholder(tf.int32, [None])

        #pentameter model placeholders
        self.pm_enc_x    = tf.placeholder(tf.int32, [None, None])
        self.pm_enc_xlen = tf.placeholder(tf.int32, [None])
        self.pm_cov_mask = tf.placeholder(tf.float32, [None, None])

        #rhyme model placeholders
        self.rm_num_context = tf.placeholder(tf.int32)

        ##################
        #pentameter model#
        ##################
        with tf.variable_scope("pentameter_model"):
            self.init_pm(is_training, batch_size, char_type_size, space_id, pad_id)

        ################
        #language model#
        ################
        with tf.variable_scope("language_model"):
            self.init_lm(is_training, batch_size, word_type_size)

        #############
        #rhyme model#
        #############
        with tf.variable_scope("rhyme_model"):
            self.init_rm(is_training, batch_size, char_type_size)

       
    # -- language model network
    def init_lm(self, is_training, batch_size, word_type_size):

        cf = self.config

        #shared word embeddings (used by encoder and decoder)
        self.word_embedding = tf.get_variable("word_embedding", [word_type_size, cf.word_embedding_dim],
            initializer=tf.random_uniform_initializer(-0.05/cf.word_embedding_dim, 0.05/cf.word_embedding_dim))
    
        #########
        #decoder#
        #########

        #define lstm cells
        lm_dec_cell = tf.nn.rnn_cell.LSTMCell(cf.lm_dec_dim, use_peepholes=True, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            lm_dec_cell = tf.nn.rnn_cell.DropoutWrapper(lm_dec_cell, output_keep_prob=cf.keep_prob)
        self.lm_dec_cell = tf.nn.rnn_cell.MultiRNNCell([lm_dec_cell] * cf.lm_dec_layer_size)
        #if cf.lm_dec_layer_size > 1:
        #    self.lm_dec_cell = ExtendedMultiRNNCell([lm_dec_cell] * cf.lm_dec_layer_size, residual_connections=True)
        #else:
        #    self.lm_dec_cell = lm_dec_cell

        #initial states
        self.lm_initial_state = self.lm_dec_cell.zero_state(batch_size, tf.float32)
        state = self.lm_initial_state

        #pad symbol vocab ID = 0; create mask = 1.0 where vocab ID > 0 else 0.0
        lm_mask = tf.cast(tf.greater(self.lm_x, tf.zeros(tf.shape(self.lm_x), dtype=tf.int32)), dtype=tf.float32)

        #embedding lookup
        word_inputs = tf.nn.embedding_lookup(self.word_embedding, self.lm_x)
        if is_training and cf.keep_prob < 1.0:
            word_inputs = tf.nn.dropout(word_inputs, cf.keep_prob)

        #process character encodings
        #concat last hidden state of fw RNN with first hidden state of bw RNN
        fw_hidden = self.get_last_hidden(self.char_encodings[0], self.pm_enc_xlen)
        char_inputs = tf.concat(1, [fw_hidden, self.char_encodings[1][:,0,:]])
        char_inputs = tf.reshape(char_inputs, [batch_size, -1, cf.pm_enc_dim*2]) #reshape into same dimension as inputs
        
        #concat word and char encodings
        inputs = tf.concat(2, [word_inputs, char_inputs])
        #inputs = word_inputs

        #apply mask to zero out pad embeddings
        inputs = inputs * tf.expand_dims(lm_mask, -1)

        #dynamic rnn
        dec_outputs, final_state = tf.nn.dynamic_rnn(self.lm_dec_cell, inputs, sequence_length=self.lm_xlen, \
            dtype=tf.float32, initial_state=self.lm_initial_state)
        self.lm_final_state = final_state

        #reshape output into [batch_size,sent_len,hidden_size] and then into [batch_size*sent_len,hidden_size]
#        hidden = tf.reshape(dec_outputs, [-1, cf.lm_dec_dim])

        #########################
        #encoder (history words)#
        #########################

        #embedding lookup
        hist_inputs = tf.nn.embedding_lookup(self.word_embedding, self.lm_hist)
        if is_training and cf.keep_prob < 1.0:
            hist_inputs = tf.nn.dropout(hist_inputs, cf.keep_prob)

        #encoder lstm cell
        lm_enc_cell = tf.nn.rnn_cell.LSTMCell(cf.lm_enc_dim, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            lm_enc_cell = tf.nn.rnn_cell.DropoutWrapper(lm_enc_cell, output_keep_prob=cf.keep_prob)

        #history word encodings
        hist_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lm_enc_cell, cell_bw=lm_enc_cell,
            inputs=hist_inputs, sequence_length=self.lm_hlen, dtype=tf.float32)

        #full history encoding
        full_encoding = tf.concat(1, [hist_outputs[0][:,-1,:], hist_outputs[1][:,0,:]])

        #concat fw and bw hidden states
        hist_outputs = tf.concat(2, hist_outputs)

        #selective encoding
        with tf.variable_scope("selective_encoding"):
            hist_outputs = self.selective_encoding(hist_outputs, full_encoding, cf.lm_enc_dim*2)

        #attention (concat)
        with tf.variable_scope("lm_attention"):
            attend_w = tf.get_variable("attend_w", [cf.lm_enc_dim*2+cf.lm_dec_dim, cf.lm_attend_dim])
            attend_b = tf.get_variable("attend_b", [cf.lm_attend_dim], initializer=tf.constant_initializer())
            attend_v = tf.get_variable("attend_v", [cf.lm_attend_dim, 1])

        enc_steps = tf.shape(hist_outputs)[1]
        dec_steps = tf.shape(dec_outputs)[1]

        #prepare encoder and decoder
        hist_outputs_t = tf.tile(hist_outputs, [1, dec_steps, 1])
        dec_outputs_t  = tf.reshape(tf.tile(dec_outputs, [1, 1, enc_steps]),
            [batch_size, -1, cf.lm_dec_dim])

        #compute e
        hist_dec_concat = tf.concat(1, [tf.reshape(hist_outputs_t, [-1, cf.lm_enc_dim*2]),
            tf.reshape(dec_outputs_t, [-1, cf.lm_dec_dim])])
        e = tf.matmul(tf.tanh(tf.matmul(hist_dec_concat, attend_w) + attend_b), attend_v)
        e = tf.reshape(e, [-1, enc_steps])

        #mask out pad symbols to compute alpha and weighted sum of history words
        #e_mask   = tf.cast(tf.equal(self.lm_hist, tf.zeros(tf.shape(self.lm_hist), dtype=tf.int32)),
        #    dtype=tf.float32)*-1e20
        #e_mask   = tf.reshape(tf.tile(e_mask, [1,dec_steps]), [-1, enc_steps])
        #alpha    = tf.reshape(tf.nn.softmax(e+e_mask), [batch_size, -1, 1])
        alpha    = tf.reshape(tf.nn.softmax(e), [batch_size, -1, 1])
        context  = tf.reduce_sum(tf.reshape(alpha * hist_outputs_t,
            [batch_size,dec_steps,enc_steps,-1]), 2)

        #save attention weights
        self.lm_attentions = tf.reshape(alpha, [batch_size, dec_steps, enc_steps])

        ##############
        #output layer#
        ##############

        #reshape both into [batch_size*len, hidden_dim]
        dec_outputs = tf.reshape(dec_outputs, [-1, cf.lm_dec_dim])
        context     = tf.reshape(context, [-1, cf.lm_enc_dim*2])
        
        #combine context and decoder hidden state with a gated unit
        with tf.variable_scope("gated_unit"):
            hidden = self.gated_layer(context, dec_outputs, cf.lm_enc_dim*2, cf.lm_dec_dim)
            #hidden = dec_outputs

        #output embeddings
        lm_output_proj = tf.get_variable("lm_output_proj", [cf.word_embedding_dim, cf.lm_dec_dim])
        lm_softmax_b   = tf.get_variable("lm_softmax_b", [word_type_size], initializer=tf.constant_initializer())
        lm_softmax_w   = tf.transpose(tf.tanh(tf.matmul(self.word_embedding, lm_output_proj)))

        #compute logits and cost
        lm_logits     = tf.matmul(hidden, lm_softmax_w) + lm_softmax_b
        lm_crossent   = tf.nn.sparse_softmax_cross_entropy_with_logits(lm_logits, tf.reshape(self.lm_y, [-1]))
        lm_crossent_m = lm_crossent * tf.reshape(lm_mask, [-1])
        self.lm_cost  = tf.reduce_sum(lm_crossent_m) / batch_size

        if not is_training:
            self.lm_probs = tf.nn.softmax(lm_logits)
            return

        #run optimiser and backpropagate (clipped) gradients for lm loss
        lm_tvars = tf.trainable_variables()
        lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_cost, lm_tvars), cf.max_grad_norm)
        #self.lm_train_op = tf.train.GradientDescentOptimizer(self.lr).apply_gradients(zip(lm_grads, lm_tvars))
        self.lm_train_op = tf.train.AdagradOptimizer(cf.lm_learning_rate).apply_gradients(zip(lm_grads, lm_tvars))


    # -- pentameter model network
    def init_pm(self, is_training, batch_size, char_type_size, space_id, pad_id):

        cf = self.config

        #character embeddings
        self.char_embedding = tf.get_variable("char_embedding", [char_type_size, cf.char_embedding_dim],
            initializer=tf.random_uniform_initializer(-0.05/cf.char_embedding_dim, 0.05/cf.char_embedding_dim))

        #get bidirectional rnn states of the encoder
        enc_cell = tf.nn.rnn_cell.LSTMCell(cf.pm_enc_dim)#, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=cf.keep_prob)

        char_inputs    = tf.nn.embedding_lookup(self.char_embedding, self.pm_enc_x)
        enc_hiddens, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell, cell_bw=enc_cell, inputs=char_inputs,
            sequence_length=self.pm_enc_xlen, dtype=tf.float32)

        #save enc_hiddens
        self.char_encodings = enc_hiddens

        #reshape enc_hiddens
        enc_hiddens  = tf.reshape(tf.concat(2, enc_hiddens), [-1, cf.pm_enc_dim*2]) #[batch_size*num_steps, hidden]

        #get decoder hidden states
        dec_cell = tf.nn.rnn_cell.LSTMCell(cf.pm_dec_dim)#, forget_bias=1.0)

        if is_training and cf.keep_prob < 1.0:
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=cf.keep_prob)

        #compute loss for pentameter
        self.pm_costs     = self.compute_pm_loss(is_training, batch_size, enc_hiddens, dec_cell, space_id, pad_id)
        self.pm_mean_cost = tf.reduce_sum(self.pm_costs) / batch_size

        if not is_training:
            return

        #run optimiser and backpropagate (clipped) gradients for pm loss
        pm_tvars         = tf.trainable_variables()
        pm_grads, _      = tf.clip_by_global_norm(tf.gradients(self.pm_mean_cost, pm_tvars), cf.max_grad_norm)
        self.pm_train_op = tf.train.AdamOptimizer(cf.pm_learning_rate).apply_gradients(zip(pm_grads, pm_tvars))


    # -- rhyme model network
    def init_rm(self, is_training, batch_size, char_type_size):

        cf = self.config

        #get char encodings
        rnn_cell = tf.nn.rnn_cell.LSTMCell(cf.rm_dim)#, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=cf.keep_prob)

        initial_state = rnn_cell.zero_state(tf.shape(self.pm_enc_x)[0], tf.float32)
        char_enc, _   = tf.nn.dynamic_rnn(rnn_cell, tf.nn.embedding_lookup(self.char_embedding, self.pm_enc_x),
            sequence_length=self.pm_enc_xlen, dtype=tf.float32, initial_state=initial_state)

        #get last hidden states
        char_enc = self.get_last_hidden(char_enc, self.pm_enc_xlen)
        
        #slice it into target_words and context words
        target  = char_enc[:batch_size,:]
        context = char_enc[batch_size:,:]

        target_tiled   = tf.reshape(tf.tile(target, [1,self.rm_num_context]), [-1, cf.rm_dim])
        target_context = tf.concat(1, [target_tiled, context])

        #cosine similarity
        e = tf.reduce_sum(tf.nn.l2_normalize(target_tiled, 1) * tf.nn.l2_normalize(context, 1), 1)
        e = tf.reshape(e, [batch_size, -1])

        #save the attentions
        self.rm_attentions = e

        #max margin loss
        min_cos = tf.nn.top_k(e, 2)[0][:, -1] #second highest cos similarity
        max_cos = tf.reduce_max(e, 1)
        self.rm_cost = tf.reduce_mean(tf.maximum(0.0, cf.rm_delta - max_cos + min_cos))

        if not is_training:
            return
        
        self.rm_train_op = tf.train.AdamOptimizer(cf.rm_learning_rate).minimize(self.rm_cost)


    # -- compute pentameter model loss, given a pentameter input
    def compute_pm_loss(self, is_training, batch_size, enc_hiddens, dec_cell, space_id, pad_id):

        cf             = self.config
        xlen_max       = tf.reduce_max(self.pm_enc_xlen)

        #use decoder hidden states to select encoder hidden states to predict stress for next time step
        repeat_loss    = tf.zeros([batch_size])
        attentions     = tf.zeros([batch_size, xlen_max]) #historical attention weights
        prev_miu       = tf.zeros([batch_size,1])
        outputs        = []
        attention_list = []
        miu_list       = []

        #initial inputs (learnable) and state
        initial_inputs = tf.get_variable("dec_init_input", [cf.pm_enc_dim*2])
        inputs         = tf.reshape(tf.tile(initial_inputs, [batch_size]), [batch_size, -1])
        state          = dec_cell.zero_state(batch_size, tf.float32)

        #manual unroll of time steps because attention depends on previous attention weights
        with tf.variable_scope("RNN"):
            for time_step in range(self.pentameter_len):

                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                def attend(enc_hiddens, dec_hidden, attn_hist, prev_miu):
                    with tf.variable_scope("pm_attention"):
                        attend_w = tf.get_variable("attend_w", [cf.pm_enc_dim*2+cf.pm_dec_dim, cf.pm_attend_dim])
                        attend_b = tf.get_variable("attend_b", [cf.pm_attend_dim], initializer=tf.constant_initializer())
                        attend_v = tf.get_variable("attend_v", [cf.pm_attend_dim, 1])
                        miu_w    = tf.get_variable("miu_w", [cf.pm_dec_dim+1, cf.pm_attend_dim])
                        miu_b    = tf.get_variable("miu_b", [cf.pm_attend_dim], initializer=tf.constant_initializer())
                        miu_v    = tf.get_variable("miu_v", [cf.pm_attend_dim, 1])
                    
                    #position attention
                    miu     = tf.minimum(tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(tf.concat(1,
                        [dec_hidden, prev_miu]), miu_w) + miu_b), miu_v)) + prev_miu, tf.ones([batch_size, 1]))
                    miu_p   = miu * tf.reshape(tf.cast(self.pm_enc_xlen-1, tf.float32), [-1, 1])
                    pos     = tf.cast(tf.reshape(tf.tile(tf.range(xlen_max), [batch_size]), [batch_size, -1]),
                        dtype=tf.float32)
                    pos_lp  = -(pos - miu_p)**2 / (2 * tf.reshape(tf.tile([tf.square(cf.sigma)], [batch_size]),
                        [batch_size,-1]))
            
                    #char encoding attention
                    pos_weight = tf.reshape(tf.exp(pos_lp), [-1, 1])
                    inp_concat = tf.concat(1, [enc_hiddens * pos_weight,
                        tf.reshape(tf.tile(dec_hidden, [1,xlen_max]), [-1,cf.pm_dec_dim])])
                    x       = self.pm_enc_x
                    e       = tf.matmul(tf.tanh(tf.matmul(inp_concat, attend_w) + attend_b), attend_v)
                    e       = tf.reshape(e, [batch_size, xlen_max])
                    mask1   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), pad_id)), dtype=tf.float32)
                    mask2   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), space_id)), dtype=tf.float32)
                    e_mask  = tf.maximum(mask1, mask2)
                    e_mask *= tf.constant(-1e20)

                    #combine alpha with position probability
                    alpha   = tf.nn.softmax(e + e_mask + pos_lp)
                    #alpha   = tf.nn.softmax(e + e_mask)

                    #weighted sum
                    c       = tf.reduce_sum(tf.expand_dims(alpha, 2)*tf.reshape(enc_hiddens,
                        [batch_size, xlen_max, cf.pm_enc_dim*2]), 1)

                    return c, alpha, miu

                dec_hidden, state               = dec_cell(inputs, state)
                enc_hiddens_sum, attn, prev_miu = attend(enc_hiddens, dec_hidden, attentions, prev_miu)

                repeat_loss += tf.reduce_sum(tf.minimum(attentions, attn), 1)
                attentions  += attn
                inputs       = enc_hiddens_sum

                attention_list.append(attn)
                miu_list.append(prev_miu)
                outputs.append(enc_hiddens_sum)

        #reshape output into [batch_size*num_steps,hidden_size]
        outputs = tf.reshape(tf.concat(1, outputs), [-1, cf.pm_enc_dim*2])

        #compute loss
        pm_softmax_w = tf.get_variable("pm_softmax_w", [cf.pm_enc_dim*2, 1])
        pm_softmax_b = tf.get_variable("pm_softmax_b", [1], initializer=tf.constant_initializer())
        pm_logit     = tf.squeeze(tf.matmul(outputs, pm_softmax_w) + pm_softmax_b)
        pm_crossent  = tf.nn.sigmoid_cross_entropy_with_logits(pm_logit,
            tf.tile(tf.cast(self.pentameter, tf.float32), [batch_size]))
        cov_loss     = tf.reduce_sum(tf.nn.relu(self.pm_cov_mask*cf.cov_loss_threshold - attentions), 1)
        pm_cost      = tf.reduce_sum(tf.reshape(pm_crossent, [batch_size, -1]), 1) + \
            cf.repeat_loss_scale*repeat_loss + cf.cov_loss_scale*cov_loss

        #save some variables
        self.pm_logits     = tf.sigmoid(tf.reshape(pm_logit, [batch_size, -1]))
        self.pm_attentions = attention_list
        self.mius          = miu_list

        return pm_cost


    # -- sample a word given probability distribution (with option to normalise the distribution with temperature)
    # -- temperature = 0 means argmax
    def sample_word(self, sess, probs, temperature, unk_symbol_id, pad_symbol_id, wordxchar, idxword,
        rm_target_pos, rm_target_neg, rm_threshold_pos, avoid_words):

        def rhyme_pair_to_char(rhyme_pair):
            char_ids, char_id_len = [], []

            for word in rhyme_pair:
                char_ids.append(wordxchar[word])
                char_id_len.append(len(char_ids[-1]))

            #pad char_ids
            for ci, c in enumerate(char_ids):
                char_ids[ci] = pad(c, max(char_id_len), pad_symbol_id)

            return char_ids, char_id_len

        def rhyme_cos(x, y):
            rm_char, rm_char_len = rhyme_pair_to_char([x, y])

            feed_dict = {self.pm_enc_x: rm_char, self.pm_enc_xlen: rm_char_len, self.rm_num_context: 1}
            rm_attns  = sess.run(self.rm_attentions, feed_dict)

            return rm_attns[0][0]

        rm_threshold_neg = 0.7 #non-rhyming words A and B shouldn't have similarity larger than this threshold

        if temperature == 0:
            return np.argmax(probs)

        probs = probs.astype(np.float64) #convert to float64 for higher precision
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / math.fsum(np.exp(probs))

        #avoid unk_symbol_id if possible
        sampled = None
        pw      = idxword[rm_target_pos] if rm_target_pos != None else "None"
        nw      = idxword[rm_target_neg] if rm_target_neg != None else "None"
        for i in range(1000):
            sampled = np.argmax(np.random.multinomial(1, probs, 1))

            #resample if it's a word to avoid
            if sampled in avoid_words:
                continue

            #if it needs to rhyme, resample until we find a rhyming word
            if rm_threshold_pos != 0.0 and rm_target_pos != None and rhyme_cos(sampled, rm_target_pos) < rm_threshold_pos:
                continue

            if rm_target_neg != None and rhyme_cos(sampled, rm_target_neg) > rm_threshold_neg:
                continue

            return sampled

        return None


    # -- generate a sentence by sampling one word at a time
    def sample_sent(self, sess, state, x, hist, hlen, xchar, xchar_len, avoid_symbols, stopwords,temp_min, temp_max,
        unk_symbol_id, pad_symbol_id, end_symbol_id, space_id, idxchar, charxid, idxword, wordxchar,
        rm_target_pos, rm_target_neg, rm_threshold, last_words, max_words):

        def filter_stop_symbol(word_ids):
            cleaned = set([])
            for w in word_ids:
                if w not in (stopwords | set([pad_symbol_id, end_symbol_id])) and not only_symbol(idxword[w]):
                    cleaned.add(w)
            return cleaned

        def get_freq_words(word_ids, freq_threshold):
            words     = []
            word_freq = Counter(word_ids)
            for k, v in word_freq.items():
                #if v >= freq_threshold and not only_symbol(idxword[k]) and k != end_symbol_id:
                if v >= freq_threshold and k != end_symbol_id:
                    words.append(k)
            return set(words)

        sent   = []

        while True:
            probs, state = sess.run([self.lm_probs, self.lm_final_state],
                {self.lm_x: x, self.lm_initial_state: state, self.lm_xlen: [1],
                self.lm_hist: hist, self.lm_hlen: hlen,
                self.pm_enc_x: xchar, self.pm_enc_xlen: xchar_len})

            #avoid words previously generated            
            avoid_words = filter_stop_symbol(sent + hist[0])
            freq_words  = get_freq_words(sent + hist[0], 2) #avoid any words that occur >= N times
            avoid_words = avoid_words | freq_words | set(sent[-3:] + last_words + avoid_symbols + [unk_symbol_id])
            #avoid_words = set(sent[-3:] + last_words + avoid_symbols + [unk_symbol_id])

            word = self.sample_word(sess, probs[0], np.random.uniform(temp_min, temp_max), unk_symbol_id,
                pad_symbol_id, wordxchar, idxword, rm_target_pos, rm_target_neg, rm_threshold, avoid_words)

            if word != None:
                sent.append(word)
                x             = [[ sent[-1] ]]
                xchar         = [wordxchar[sent[-1]]]
                xchar_len     = [len(xchar[0])]
                rm_target_pos = None
                rm_target_neg = None
            else:
                return None, None, None

            if sent[-1] == end_symbol_id or len(sent) >= max_words:

                if len(sent) > 1:
                    pm_loss  = self.eval_pm_loss(sess, sent, end_symbol_id, space_id, idxchar, charxid, idxword, wordxchar)
                    return sent, state, pm_loss
                else:
                    return None, None, None
    

    # -- compute pentameter loss
    def eval_pm_loss(self, sess, sent, end_symbol, space_id, idxchar, charxid, idxword, wordxchar):

        def sent_to_char(words):
            char_ids = []

            for word in reversed(words):
                if word != end_symbol:
                    char_ids.extend(wordxchar[word])
                    char_ids.append(space_id)

            #remove punctuation
            chars = "".join([ idxchar[item] for item in char_ids])
            char_ids = [charxid[item] for item in remove_punct(chars)]

            return char_ids

        #pentameter check
        pm_char  = sent_to_char(sent)
        cov_mask = coverage_mask(pm_char, idxchar)

        #create pseudo batch
        b = ([pm_char], [len(pm_char)], [cov_mask])

        feed_dict = {self.pm_enc_x: b[0], self.pm_enc_xlen: b[1], self.pm_cov_mask: b[2]}
        pm_attns, pm_costs, logits, mius = sess.run([self.pm_attentions, self.pm_costs, self.pm_logits,
            self.mius], feed_dict)

        return pm_costs[0]


    # -- quatrain generation
    def generate(self, sess, idxword, idxchar, charxid, wordxchar, pad_symbol_id, end_symbol_id, unk_symbol_id, space_id,
        avoid_symbols, stopwords, temp_min, temp_max, max_lines, max_words, sent_sample, rm_threshold, verbose=False):

        def reset():
            rhyme_aabb  = ([None, 0, None, 2], [None, None, 1, None])
            rhyme_abab  = ([None, None, 0, 1], [None, 0, None, None])
            rhyme_abba  = ([None, None, 1, 0], [None, 0, None, None])
            rhyme_pttn  = random.choice([rhyme_aabb, rhyme_abab, rhyme_abba])
            #rhyme_pttn  = random.choice([rhyme_aabb])
            state       = sess.run(self.lm_dec_cell.zero_state(1, tf.float32))
            prev_state  = state
            x           = [[end_symbol_id]]
            xchar       = [wordxchar[end_symbol_id]]
            xchar_len   = [1]
            sonnet      = []
            sent_probs  = []
            last_words  = []
            total_words = 0
            total_lines = 0
            
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
                one_sent, one_state, one_pl = self.sample_sent(sess, state, x, hist, hlen, xchar, xchar_len,
                    avoid_symbols, stopwords, temp_min, temp_max, unk_symbol_id, pad_symbol_id, end_symbol_id, space_id,
                    idxchar, charxid, idxword, wordxchar, rm_target_pos, rm_target_neg, rm_threshold, last_words, max_words)
                if one_sent != None:
                    all_sent.append(one_sent)
                    all_state.append(one_state)
                    all_pl.append(-one_pl)
                else:
                    all_sent = []
                    break

            #unable to generate sentences; reset whole quatrain
            if len(all_sent) == 0:

                state, prev_state, x, xchar, xchar_len, sonnet, sent_probs, last_words, total_words, total_lines, \
                    rhyme_pttn_pos, rhyme_pttn_neg = reset()

            else:

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
