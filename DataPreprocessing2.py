# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python [conda env:llms]
#     language: python
#     name: conda-env-llms-py
# ---

from Vocabulary import normalizeString, vocabulary, unicodetoascii
import itertools
import torch
import random

PAD = 0
SOS = 1
EOS = 2

datafile = "data/formatted_movie_lines.txt"

lines = open(datafile, encoding="utf-8").read().strip().split("\n\n")
pairs = [[normalizeString(s) for s in pair.split("\t")] for pair in lines]

len(pairs)

corpus = vocabulary("Cornell Movie Dialogues")


# +
#a bit more cleaning, so well remove any sentances that are too long
def filterpair(p, max_length=10):
    return len(p[0].split()) <= max_length and len(p[1].split()) <= max_length

pairs = [pair for pair in pairs if filterpair(pair)]
# -

len(pairs)

pairs[:10]


def trimRareWords(vocab, pairs, min_count = 3):
    
    vocab.trim(min_count=min_count)
    keep_pairs = []
    for pair in pairs:
        input_ = pair[0]
        reply_ = pair[1]
        keepinput, keepreply = True, True
        for word in input_.split(" "):
            if word not in vocab.word2index:
                keepinput = False
                break
        for word in reply_.split(" "):
            if word not in vocab.word2index:
                keepreply = False
                break
        if keepinput and keepreply:
            keep_pairs.append(pair)

    print(f"After trimming kept {len(keep_pairs)} out of {len(pairs)}")
    
    return keep_pairs


# +
for pair in pairs:
    corpus.addSentance(pair[0])
    corpus.addSentance(pair[1])

print(corpus.num_words)
# -

cleaned_pairs = trimRareWords(corpus, pairs)

cleaned_pairs[:10]


def indexfromSentance(vocab:vocabulary, sentance:str):
    return [vocab.word2index[word] for word in sentance.split(" ")] + [EOS]


indexfromSentance(corpus, cleaned_pairs[1][0])

inputs = []
for pair in cleaned_pairs[:10]:
    inputs.append(indexfromSentance(corpus, pair[0]))


def zeropading(l, fillvalue = 0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binarymatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD:
                m[i].append(0)
            else:
                m[i].append(1)

    return m


l = zeropading(inputs)

inputs

l

binary = binarymatrix(l)

binary


#Returns padded input sequence tensor as well as tensor of lengths for each of the padded seq in the batc
def inputVar(l:list, vocab:vocabulary):
    indexes_batch = [indexfromSentance(vocab, sentance) for sentance in l]
    lengths = torch.tensor([len(index_array) for index_array in indexes_batch])
    padlist = zeropading(indexes_batch)
    padvar = torch.LongTensor(padlist)
    return padvar, lengths


# Returns padded target sequence tensor, padding mask and maax target length
def outputVar(l:list, vocab:vocabulary):
    indexes_batch = [indexfromSentance(vocab, sentance) for sentance in l]
    max_target_len = max([len(index_array) for index_array in indexes_batch])
    padlist = zeropading(indexes_batch)
    mask = binarymatrix(padlist)
    mask = torch.ByteTensor(mask)
    padvar = torch.LongTensor(padlist)
    return padvar, mask, max_target_len


#Prepares the data for training for a given batch of pairs
def batch2traindata(vocab, pair_batch):
    #Sort the question answers pairs in descending order
    pair_batch.sort(key=lambda x:len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, vocab)
    output, mask, max_target_len = outputVar(output_batch, vocab)
    return inp, lengths, output, mask, max_target_len


#Validation of preprocessing steps
batch_size = 5
input_seq, lengths, target_seq, target_mask, max_target_length = batch2traindata(corpus, [random.choice(cleaned_pairs) for _ in range(batch_size)])

print(input_seq)

print(lengths)

print(target_seq)

print(target_mask)

print(max_target_length)


