from Vocabulary import vocabulary
import itertools
import torch
import random
import re
import unicodedata

def filterpair(p:list[str], max_length=10):
    """
    Clip Sentances above a certain length
    """
    return len(p[0].split()) <= max_length and len(p[1].split()) <= max_length

def trimRareWords(vocab:vocabulary, pairs:list[list[str]], min_count=3) -> list[list[str]]:
    """
    Removes coversations that have words that occur less frequently
    """
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

def unicodetoascii(s:str) -> str:
    """
    Removes things like accents on charecters e with a ` becomes just e
    """
    return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn")

def normalizeString(s:str) -> str:
    """
    Does stuff like removes nonalphanumeric charectrs then removes extra whitespace and also .!? are made into their own charecters
    """
    s = unicodetoascii(s.lower().strip())
    # Replace any .!? by a whitespace + the charecter --> "!" will be replaced by " !". \1 means the first bracketed group --> [,!?]. r is to not consider
    # \1 as a charecter (r is to escape backslash)
    s = re.sub(r"([.!?])", r" \1", s)
    # Remove any sequence of white space charecters. + means more
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # Remove any sequence of whitespace charecters
    s = re.sub(r"\s+", r" ", s).strip()

    return s

def zeropading(l, fillvalue = 0) -> list:
    """
    Takes a List of list of indexes after generating indices from the words and pads the index arrays that are smaller than the max size of a sequence
    Also from (batch_size, seq_len) -> (maax_seq_len, batch_size)
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binarymatrix(l:list[list[int]], value=0):
    """
    Returns a masked output of the padded index list so that we can compute loss later down the line
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)

    return m

def inputVar(l:list, vocab:vocabulary):
    """
    Returns padded input sequence tensor as well as tensor of lengths for each of the padded seq in the batch for training preperation
    """
    indexes_batch = [vocab.indexfromSentance(sentance) for sentance in l]
    lengths = torch.tensor([len(index_array) for index_array in indexes_batch])
    padlist = zeropading(indexes_batch)
    padvar = torch.LongTensor(padlist)
    return padvar, lengths

def outputVar(l:list, vocab:vocabulary):
    """
    Returns padded target sequence tensor, padding mask and maax target length for loss calculation during training
    Input: Batches of list of indexes formed from the sentances
    Output: Padded indexes to maintain max seq len
            Masked 1 and 0 version of the padded indexes
            maximum length of any sentance
    """
    indexes_batch = [vocab.indexfromSentance(sentance) for sentance in l]
    max_target_len = max([len(index_array) for index_array in indexes_batch])
    padlist = zeropading(indexes_batch)
    mask = binarymatrix(padlist)
    mask = torch.ByteTensor(mask)
    padvar = torch.LongTensor(padlist)
    return padvar, mask, max_target_len

def batch2traindata(vocab, pair_batch):
    """
    Prepares the data for training for a given batch of pairs
    Output: 0 - train input padded transposed index arrays
            1 - tensor of lengths of each sentance
            2 - targets padded transposed index arrays
            3 - Masked 1 and 0 of targets transposed index arrays
            4 - Max length of the sequences
    """
    #Sort the question answers pairs in descending order
    pair_batch.sort(key=lambda x:len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, vocab)
    output, mask, max_target_len = outputVar(output_batch, vocab)
    return inp, lengths, output, mask, max_target_len

if __name__ == "__main__":

    s = "aa123aa!s's dd?"

    print(normalizeString(s))
