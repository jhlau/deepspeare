import codecs
import operator
import numpy as np
import random
import math
import codecs
import sys
from collections import defaultdict


def load_vocab(corpus, word_minfreq, dummy_symbols):
    idxword, idxchar = [], []
    wordxid, charxid = defaultdict(int), defaultdict(int)
    word_freq, char_freq = defaultdict(int), defaultdict(int)
    wordxchar = defaultdict(list)

    def update_dic(symbol, idxvocab, vocabxid):
        if symbol not in vocabxid:
            idxvocab.append(symbol)
            vocabxid[symbol] = len(idxvocab) - 1 

    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            word_freq[word] += 1
        for char in line.strip():
            char_freq[char] += 1

    #add in dummy symbols into dictionaries
    for s in dummy_symbols:
        update_dic(s, idxword, wordxid)
        update_dic(s, idxchar, charxid)

    #remove low fequency words/chars
    def collect_vocab(vocab_freq, idxvocab, vocabxid):
        for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
            if f < word_minfreq:
                break
            else:
                update_dic(w, idxvocab, vocabxid)

    collect_vocab(word_freq, idxword, wordxid)
    collect_vocab(char_freq, idxchar, charxid)

    #word id to [char ids]
    dummy_symbols_set = set(dummy_symbols)
    for wi, w in enumerate(idxword):
        if w in dummy_symbols:
            wordxchar[wi] = [wi]
        else:
            for c in w:
                wordxchar[wi].append(charxid[c] if c in charxid else charxid[dummy_symbols[2]])

    return idxword, wordxid, idxchar, charxid, wordxchar


def only_symbol(word):
    for c in word:
        if c.isalpha():
            return False

    return True


def remove_punct(string):
    return " ".join("".join([ item for item in string if (item.isalpha() or item == " ") ]).split())


def load_data(corpus, wordxid, idxword, charxid, idxchar, (pad_symbol, end_symbol, unk_symbol)):
    nwords     = [] #number of words for each line
    nchars     = [] #number of chars for each line
    word_data  = [] #data[doc_id][0][line_id] = list of word ids; data[doc_id][1][line_id] = list of [char_ids]
    char_data  = [] #data[line_id] = list of char ids
    rhyme_data = [] #list of ( target_word, [candidate_words], target_word_line_id ); word is a list of characters

    def word_to_char(word):
        if word in set([pad_symbol, end_symbol, unk_symbol]):
            return [ wordxid[word] ]
        else:
            return [ charxid[item] if item in charxid else charxid[unk_symbol] for item in word ]


    for doc in codecs.open(corpus, "r", "utf-8"):

        word_lines, char_lines = [[], []], []
        last_words = []

         #reverse the order of lines and words as we are generating from end to start
        for line in reversed(doc.strip().split(end_symbol)):

            if len(line.strip()) > 0:

                word_seq = [ wordxid[item] if item in wordxid else wordxid[unk_symbol] \
                    for item in reversed(line.strip().split()) ] + [wordxid[end_symbol]]

                char_seq = [ word_to_char(item) for item in reversed(line.strip().split()) ] + [word_to_char(end_symbol)]

                word_lines[0].append(word_seq)
                word_lines[1].append(char_seq)
                char_lines.append([ charxid[item] if item in charxid else charxid[unk_symbol] \
                    for item in remove_punct(line.strip())])
                nwords.append(len(word_lines[0][-1]))
                nchars.append(len(char_lines[-1]))

                last_words.append(line.strip().split()[-1])

        if len(word_lines[0]) == 14: #14 lines for sonnets

            word_data.append(word_lines)
            char_data.extend(char_lines)

            #last_words = last_words[:12] #remove couplets (since they don't always rhyme)
            last_words = last_words[2:] #remove couplets (since they don't always rhyme)

            for wi, w in enumerate(last_words):
                rhyme_data.append( (word_to_char(w), [ word_to_char(item)
                    for item_id, item in enumerate(last_words[(wi/4)*4:(wi/4+1)*4]) if item_id != (wi%4) ], (11-wi)) )

    return word_data, char_data, rhyme_data, nwords, nchars
            

def print_stats(partition, word_data, rhyme_data, nwords, nchars):
    print partition, "statistics:"
    print "  Number of documents         =", len(word_data)
    print "  Number of rhyme examples    =", len(rhyme_data)
    print "  Total number of word tokens =", sum(nwords)
    print "  Mean/min/max words per line = %.2f/%d/%d" % (np.mean(nwords), min(nwords), max(nwords))
    print "  Total number of char tokens =", sum(nchars)
    print "  Mean/min/max chars per line = %.2f/%d/%d" % (np.mean(nchars), min(nchars), max(nchars))


def init_embedding(model, idxword):
    word_emb = []
    for vi, v in enumerate(idxword):
        if v in model:
            word_emb.append(model[v])
        else:
            word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
    return np.array(word_emb)


def pad(lst, max_len, pad_symbol):
    if len(lst) > max_len:
        print "\nERROR: padding"
        print "length of list greater than maxlen; list =", lst, "; maxlen =", max_len
        raise SystemExit
    return lst + [pad_symbol] * (max_len - len(lst))


def get_vowels():
    return set(["a", "e", "i", "o", "u"])


def coverage_mask(char_ids, idxchar):
    vowels = get_vowels()
    return [ float(idxchar[c] in vowels) for c in char_ids ]


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def create_word_batch(data, batch_size, lines_per_doc, nlines_per_batch, pad_symbol, end_symbol, unk_symbol, shuffle_data):
    docs_per_batch = len(data) / batch_size
    batches = []
    doc_ids = range(len(data))
    if shuffle_data:
        random.shuffle(doc_ids)

    if lines_per_doc % nlines_per_batch != 0:
        print "\nERROR:"
        print "lines_per_doc (%d) %% nlines_per_batch (%d) must equal 0" % (lines_per_doc, nlines_per_batch)
        raise SystemExit

    for i in range(docs_per_batch):

        for j in range(lines_per_doc / nlines_per_batch):

            docs       = []
            doc_lens   = []
            doc_lines  = []
            x          = []
            y          = []
            xchar      = []
            xchar_lens = []
            hist       = []
            hist_lens  = []

            for k in range(batch_size):

                d       = doc_ids[i*batch_size+k]
                wordseq = flatten_list(data[d][0][j*nlines_per_batch:(j+1)*nlines_per_batch])
                charseq = flatten_list(data[d][1][j*nlines_per_batch:(j+1)*nlines_per_batch])
                histseq = flatten_list(data[d][0][:j*nlines_per_batch])
    
                x.append([end_symbol] + wordseq[:-1])
                y.append(wordseq)

                docs.append(d)
                doc_lens.append(len(wordseq))
                doc_lines.append(range(j*nlines_per_batch, (j+1)*nlines_per_batch))

                xchar.append([[end_symbol]] + charseq[:-1])
                xchar_lens.append( [1] + [ len(item) for item in charseq[:-1] ])

                hist.append(histseq if len(histseq) > 0 else [unk_symbol])
                hist_lens.append(len(histseq) if len(histseq) > 0 else 1)

            #pad the data
            word_pad_len = max(doc_lens)
            char_pad_len = max(flatten_list(xchar_lens))
            hist_pad_len = max(hist_lens)
            for k in range(batch_size):

                x[k] = pad(x[k], word_pad_len, pad_symbol)
                y[k] = pad(y[k], word_pad_len, pad_symbol)

                xchar_lens[k].extend( [1]*(word_pad_len-len(xchar[k])) ) #add len for pad symbols

                xchar[k] = pad(xchar[k], word_pad_len, [pad_symbol]) #pad the word lengths
                xchar[k] = [pad(item, char_pad_len, pad_symbol) for item in xchar[k]] #pad the characters

                hist[k] = pad(hist[k], hist_pad_len, pad_symbol)

            batches.append((x, y, docs, doc_lens, doc_lines, xchar, xchar_lens, hist, hist_lens))

    return batches


def create_char_batch(data, batch_size, pad_symbol, pentameter, idxchar, shuffle_data):
    batches = []
    batch_len = len(data) / batch_size

    if shuffle_data:
        random.shuffle(data)

    for i in range(batch_len):
        enc_x    = []
        enc_xlen = []

        for j in range(batch_size):
            enc_x.append(data[i*batch_size+j])
            enc_xlen.append(len(data[i*batch_size+j]))

        xlen_max = max(enc_xlen)
        cov_mask = np.zeros((batch_size, xlen_max)) #coverage mask

        for j in range(batch_size):
            enc_x[j]    = pad(enc_x[j], xlen_max, pad_symbol)
            cov_mask[j] = coverage_mask(enc_x[j], idxchar)
            
        batches.append((enc_x, enc_xlen, cov_mask))

    return batches


def create_rhyme_batch(data, batch_size, pad_symbol, wordxchar, num_neg, shuffle_data):
    if shuffle_data:
        random.shuffle(data)

    batches = []

    for i in range(len(data) / batch_size):
        x, xid, c  = [], [], []
        xlen, clen = [], []

        for j in range(batch_size):
            x.append(data[i*batch_size+j][0])
            xid.append(data[i*batch_size+j][2])
            xlen.append(len(data[i*batch_size+j][0]))
            for context in data[i*batch_size+j][1]:
                c.append(context)
                clen.append(len(context))
            for _ in range(num_neg):
                c.append(wordxchar[random.randrange(3, len(wordxchar))])
                clen.append(len(c[-1]))

        #merging target and context words
        # (first batch_size = target words; following batch_size*(3+num_neg) = context words)
        xc        = x + c
        xclen     = xlen + clen
        xclen_max = max(xclen)

        #pad the target words and context words
        for xci, xcv in enumerate(xc):
            xc[xci] = pad(xcv, xclen_max, pad_symbol)

        batches.append((xc, xclen, xid))

    return batches
        

def print_lm_attention(bi, b, attentions, idxword, cf):

    print "\n", "="*100
    for ex in range(cf.batch_size)[-1:]:
        xword = [ idxword[item] for item in b[1][ex] ]
        hword = [ idxword[item] for item in b[7][ex] ]
        print "\nBatch ID =", bi
        print "Example =", ex
        print "x_word =", " ".join(xword)
        print "hist_word=", " ".join(hword)
        for xi, x in enumerate(xword):
            print "\nWord =", x
            print "\tSum dist =", sum(attentions[ex][xi])
            attn_dist_sort = np.argsort(-attentions[ex][xi])
            print "\t", 
            for hi in attn_dist_sort[:5]:
                print ("[%d]%s:%.3f  " % (hi, hword[hi], attentions[ex][xi][hi])),
            print


def print_pm_attention(b, batch_size, costs, logits, attentions, mius, idxchar):

    print "\n", "="*100
    for ex in range(batch_size)[-10:]:
        print "\nSentence =", ex
        print "x =", b[0][ex]
        print "x len =", b[1][ex]
        print "x char=", "".join(idxchar[item] for item in b[0][ex])
        print "losses =", costs[ex]
        print "pentameter output =", logits[ex]
        print "coverage mask =", b[2][ex]
        for attni, attn in enumerate(attentions):
            print "attention at time step", attni, ":"
            print "\tmiu_p =", mius[attni][ex] * (b[1][ex] - 1.0)
            for xid in reversed(np.argsort(attn[ex])):
                if attn[ex][xid] > 0.05:
                    print "\t%.3f %d %s" % (attn[ex][xid], xid, (idxchar[b[0][ex][xid]]))


def print_rm_attention(b, batch_size, num_context, attentions, pad_id, idxchar):

    print "\n", "="*100
    for exid in range(batch_size)[-10:]:
        print "\nTarget word =", "".join([idxchar[item] for item in b[0][exid] if item != pad_id])
        for ci, c in enumerate(b[0][(exid*num_context+batch_size):((exid+1)*num_context+batch_size)]):
            print "\t", ("%.2f" % attentions[exid][ci]), "=", "".join([idxchar[item] for item in c if item != pad_id])


def get_word_stress(cmu, word):

    stresses = set([])

    def valid(stress):
        for sti in range(len(stress)-1):
            if abs(int(stress[sti]) - int(stress[sti+1])) != 1:
                return False
        return True

    if word in cmu:
        for res in cmu[word]:
            stress = ""
            for syl in res:
                if syl[-1] == "0":
                    stress += "0"
                elif syl[-1] == "1" or syl[-1] == "2":
                    stress += "1"

            if valid(stress):
                stresses.add(stress)

    return stresses


def update_stress_accs(accs, word_len, score):

    word_buckets   = [4,8,float("inf")]

    for wi, wb in enumerate(word_buckets):
        if word_len <= wb:
            accs[wi].append(score)
            break


def eval_stress(accs, cmu, attns, pentameter, batch, idxchar, charxid, pad_symbol, cf):

    attn_threshold = 0.2

    for ex in range(cf.batch_size):

        chars     = [idxchar[item] for item in batch[ex]]
        space_ids = [-1] + [i for i, ch in enumerate(chars) if ch == " "] + \
            [batch[ex].index(charxid[pad_symbol]) if charxid[pad_symbol] in batch[ex] else len(batch[ex])]

        """
        print "="*100
        print "batch id =", ex
        print "chars =", "".join(chars)
        print "space_ids =", space_ids
        """

        for spi, sp in enumerate(space_ids[:-1]):

            start  = sp+1
            end    = space_ids[spi+1]
            word   = "".join(chars[start:end])

            gold_stress = get_word_stress(cmu, word)
            sys_stress  = ""

            """
            print "\nword =", word
            print "start, end =", start, end
            print "gold stress =", gold_stress
            """

            if len(gold_stress) == 0:
                continue

            for attni, attn in enumerate(attns):

                #print "\tattention for start,end =", attn[ex][start:end]

                for ch in range(start, end):

                    if attn[ex][ch] >= attn_threshold: 
                        sys_stress += str(pentameter[attni])
                        break

            #print "sys stress =", sys_stress

            update_stress_accs(accs, (end-start), float(sys_stress in gold_stress))


def eval_rhyme(pr, thresholds, cmu, attns, b, idxchar, charxid, pad_symbol, cf, cmu_rhyme=None, cmu_norhyme=None,
    em_vocab=None, em_theta=None):

    def syllable_to_rhyme(syllable):

        stresses = set(["0", "1", "2"])
        r = []
        for s in reversed(syllable):
            if s[-1] in stresses:
                r.append(s[:-1])
                break
            else:
                r.append(s)

        return "-".join(list(reversed(r)))

    def get_rhyme(words):

        rhymes = []
        
        for word in words:
            word_rhymes = set([])
            if word in cmu:
                for res in cmu[word]:
                    word_rhymes.add(syllable_to_rhyme(res))
            rhymes.append(word_rhymes)

        return rhymes

    def rhyme_score(target_rhymes, context_rhymes):
        
        if len(target_rhymes) == 0 or len(context_rhymes) == 0:
            return None
        else:
            for tr in target_rhymes:
                if tr in context_rhymes: 
                    return 1.0
            return 0.0

    def last_syllable(word):

        i = len(word)
        for c in reversed(word):
            i -= 1
            if c in get_vowels():
                break

        return word[i:]

    def em_rhyme_score(x, y):

        if x in em_vocab and y in em_vocab:
            xi = em_vocab.index(x)
            yi = em_vocab.index(y)

            return max(em_theta[xi][yi], em_theta[yi][xi])
    
        return 0.0


    num_c = 3 + cf.rm_neg
    for ex in range(cf.batch_size):
        target  = "".join([idxchar[item] for item in b[0][ex][:b[1][ex]]])
        context = []
        #for ci, c in enumerate(b[0][(ex*num_c+cf.batch_size):((ex+1)*num_c+cf.batch_size)]):
        for ci, c in enumerate(b[0][(ex*num_c+cf.batch_size):(ex*num_c+cf.batch_size+3)]):
            context.append("".join([idxchar[item] for item in c[:b[1][ex*num_c+cf.batch_size+ci]]]))

        target_rhyme  = get_rhyme([target])[0]
        context_rhyme = get_rhyme(context)

        """
        print "\n", "="*80
        print "batch id =", ex
        print target, "=", target_rhyme
        """

        for t in thresholds:
            #print "\nThreshold =", t
            for ci, c in enumerate(context):
                score = rhyme_score(target_rhyme, context_rhyme[ci])
                system_score = attns[ex][ci] if em_vocab == None else em_rhyme_score(target, context[ci])
                #rhyme_baseline = (last_syllable(target) == last_syllable(context[ci]))
                #print "\n\t", c, "=", context_rhyme[ci]
                #print "\t\tscore =", score
                #print "\t\tattn  =", attns[ex][ci]

                #precision
                if system_score >= t:
                #if rhyme_baseline:
                    if score != None:
                        pr[t][0].append(score)
                        #print "\t\t\tupdating precision!"
                        
                #recall
                if score == 1.0:
                    pr[t][1].append(float(system_score >= t))
                    #pr[t][1].append(float(rhyme_baseline))
                    #print "\t\t\tupdating recall!"
            
                #print "\tpr[t] =", pr[t]

                if cmu_rhyme != None and cmu_norhyme != None:
                    if score == 1.0:
                        if (target, context[ci]) not in cmu_rhyme and (context[ci], target) not in cmu_rhyme:
                            cmu_rhyme[(target, context[ci])] = system_score
                    elif score == 0.0:
                        if (target, context[ci]) not in cmu_norhyme and (context[ci], target) not in cmu_norhyme:
                            cmu_norhyme[(target, context[ci])] = system_score


def collect_rhyme_pattern(rhyme_pattern, attentions, b, batch_size, num_context, idxchar, pad_id):

    def get_context_line_id(target_line_id, context_id):
        p = target_line_id % 4
        q = 3-context_id
        if q <= p:
            q -= 1
        return q

    #print "\n", "="*100
    for exid in range(batch_size):
        target_line_id = b[2][exid]
        #print "\nTarget word =", "".join([idxchar[item] for item in b[0][exid] if item != pad_id])
        #print "Target word line id =", target_line_id
        for ci, c in enumerate(b[0][(exid*num_context+batch_size):((exid+1)*num_context+batch_size)][:3]):
            context_line_id = get_context_line_id(target_line_id, ci)
            #print "\t", ("%.2f" % attentions[exid][ci]), "=", "".join([idxchar[item] for item in c if item != pad_id])
            #print "\t\tContext word line id =", context_line_id
            rhyme_pattern[target_line_id][context_line_id].append(attentions[exid][ci])


def postprocess_sentence(line):
    cleaned_sent = ""
    for w in line.strip().split():
        spacing = " "
        if w.startswith("'") or only_symbol(w):
            spacing = ""
        cleaned_sent += spacing + w
    return cleaned_sent.strip()
