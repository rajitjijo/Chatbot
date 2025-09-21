class vocabulary:

    PAD = 0
    SOS = 1
    EOS = 2

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

    def indexfromSentance(self, sentance:str) -> list:
        """
        Convers a sentance of words into a list of indexes from the corpus
        """
        return [self.word2index[word] for word in sentance.split(" ")] + [self.EOS]
    
    def __len__(self):
        return self.num_words

if __name__ == "__main__":

    pass