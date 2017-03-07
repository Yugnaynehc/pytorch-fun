

class Lang():
    def __init__(self, name):
        self.name = name
        self.word2idx = {'SOS': 0, 'EOS': 1}
        self.word2cnt = {}
        self.idx2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # include 'SOS' and 'EOS'

    def add_word(self, w):
        if w in self.word2idx:
            self.word2cnt[w] += 1
        else:
            self.word2idx[w] = self.n_words
            self.idx2word[self.n_words] = w
            self.word2cnt[w] = 1
            self.n_words += 1

    def add_word_list(self, l):
        for w in l:
            self.add_word(w)
