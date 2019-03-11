
from typing import List
from common.sentence import Sentence

class Instance:

    def __init__(self, input: Sentence, output: List[str]):
        self.input = input
        self.output = output
        self.id = None
        self.marginals = None

    def set_id(self, id: int):
        self.id = id

    def __len__(self) -> int:
        return len(self.input)

