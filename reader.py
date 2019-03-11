
from tqdm import tqdm
from common.sentence import Sentence
from common.instance import Instance
import re

class Reader:


    def __init__(self, digit2zero: bool, dataset: str):
        self.digit2zero = digit2zero
        self.train_vocab = []
        self.test_vocab = []
        self.dataset = dataset


    def read_from_file(self, file, number=-1, is_train=True):
        print("Reading file: " + file)
        insts = []
        # vocab = set() ## build the vocabulary
        id = 0
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            # for line in f.readlines():
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    inst = Instance(Sentence(words), labels)
                    inst.set_id(id)
                    id += 1
                    insts.append(inst)
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                if self.dataset == "conll2003":
                    word, _, label = line.split()
                elif self.dataset == "conll2002" or self.dataset == "ecommerce" or self.dataset == "youku":
                    x = line.split()
                    if len(x) == 1:
                        word = ' '
                    else:
                        word = x[0]
                        label = x[1]
                    # word, label = line.split()
                else:
                    raise Exception("unknown dataset: " + self.dataset + " during read data")
                if self.digit2zero:
                    word = re.sub('\d', '0', word)
                words.append(word)
                if is_train:
                    if word not in self.train_vocab:
                        self.train_vocab.append(word)
                else:
                    if word not in self.test_vocab:
                        self.test_vocab.append(word)
                labels.append(label)
        return insts


