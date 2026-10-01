"""Word tokenization for the compact VQA model."""
import re


class Vocabulary:
    def __init__(self, words=None):
        self.words = words or ['<pad>', '<unk>', '<bos>', '<eos>']
        self.ids = {word: index for index, word in enumerate(self.words)}

    @staticmethod
    def split(text):
        return re.findall(r'\w+|[^\w\s]', str(text).lower())

    def fit(self, rows):
        words = {word for row in rows for key in ('question', 'answer') for word in self.split(row[key])}
        self.__init__(self.words + sorted(words - set(self.words)))
        return self

    def encode(self, text):
        return [self.ids.get(word, 1) for word in self.split(text)] or [1]

    def decode(self, tokens):
        result = []
        for token in tokens:
            if token == 3:
                break
            if token > 3:
                result.append(self.words[token])
        return ' '.join(result)
