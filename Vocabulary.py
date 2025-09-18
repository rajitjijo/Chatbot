import re
import unicodedata

class vocabulary:

    def __init__(self, name):

        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.num_words = 3

    def addSentance(self, sentance):

        for word in sentance.split(" "):
            self.addWord(word)


    def addWord(self, word):

        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
            
    # Remove words bewlow a certain count threshold
    def trim(self, min_count):

        keep_words = [k for k,v in self.word2count.items() if v>=min_count]
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

def unicodetoascii(s):

    return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn")

def normalizeString(s):

    s = unicodetoascii(s.lower().strip())
    # Replace any .!? by a whitespace + the charecter --> "!" will be replaced by " !". \1 means the first bracketed group --> [,!?]. r is to not consider
    # \1 as a charecter (r is to escape backslash)
    s = re.sub(r"([.!?])", r" \1", s)
    # Remove any sequence of white space charecters. + means more
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # Remove any sequence of whitespace charecters
    s = re.sub(r"\s+", r" ", s).strip()

    return s

if __name__ == "__main__":

    s = "aa123aa!s's dd?"

    print(normalizeString(s))