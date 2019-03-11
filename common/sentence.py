from typing import List


class Sentence:

    def __init__(self, words: List[str]):
        self.words = words

    def get_words(self):
        return self.words

    def __len__(self):
        return len(self.words)
