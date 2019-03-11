
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from typing import List
from common.instance import Instance

import random


class Config:
    def __init__(self, args):
        self.seed = args.seed
        self.setSeed()

        self.PAD = "<PAD>"
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"


        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None

        self.digit2zero = args.digit2zero

        self.dataset = args.dataset
        self.train_file = "data/"+self.dataset+"/train.txt"
        self.dev_file = "data/"+self.dataset+"/dev.txt"
        self.test_file = "data/"+self.dataset+"/test.txt"
        print("train_file: %s" % (self.train_file))
        print("dev_file: %s" % (self.dev_file))
        print("test_file: %s" % (self.test_file))

        self.unk = "</s>"
        self.unk_id = -1

        if self.dataset == "ecommerce" or self.dataset == "youku" or self.dataset== "conll2002":
            self.unk = "</s>"

        self.label2idx = {}
        self.idx2labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0
        self.use_dev = True
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num

        ### optimization parameter
        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        # self.lr_decay = 0.05
        self.eval_freq = args.eval_freq

        self.hidden_dim = args.hidden_dim
        # self.tanh_hidden_dim = args.tanh_hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn

        ## task specific
        self.entity_keep_ratio = args.entity_keep_ratio
        self.num_folds = args.kfold
        self.model_type = args.model_type
        self.large_iter = args.large_iter

    def setSeed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)


    '''
      read all the  pretrain embeddings
    '''
    def read_pretrain_embedding(self):
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        embedding_dim = -1
        embedding = dict()
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if len(tokens) == 2:
                    continue
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
                    if embedding_dim + 1 != len(tokens):
                        continue
                    # assert (embedding_dim + 1 == len(tokens))
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim


    '''
        build the embedding table
        obtain the word2idx and idx2word as well.
    '''
    def build_emb_table(self, train_vocab, test_vocab):
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)

        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.unk] = 0
        self.unk_id = 0
        self.idx2word.append(self.unk)

        self.char2idx[self.unk] = 0
        self.idx2char.append(self.unk)

        for word in train_vocab:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            for c in word:
                if c not in self.char2idx:
                    self.char2idx[c] = len(self.idx2char)
                    self.idx2char.append(c)

        for word in test_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)

        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    self.word_embedding[self.word2idx[word], :] = np.copy(self.embedding[self.unk])
                    # self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])


    def build_label_idx(self, insts):
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        print("#labels: " + str(self.label_size))
        print("label 2idx: " + str(self.label2idx))

    def use_iobes(self, insts):
        for inst in insts:
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)

    def map_insts_ids(self, insts: List[Instance]):
        for inst in insts:
            curr_words = inst.input.words
            inst.input.word_ids = []
            inst.input.char_ids = []
            inst.label_ids = []
            for word in curr_words:
                if word in self.word2idx:
                    inst.input.word_ids.append(self.word2idx[word])
                else:
                    inst.input.word_ids.append(self.word2idx[self.unk])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.unk])
                inst.input.char_ids.append(char_id)
            for label in inst.output:
                inst.label_ids.append(self.label2idx[label])


    def find_singleton(self, train_insts):
        freq = {}
        self.singleton = set()
        for inst in train_insts:
            words = inst.input.words
            for w in words:
                if w in freq:
                    freq[w] += 1
                else:
                    freq[w] = 1
        for w in freq:
            if freq[w] == 1:
                self.singleton.add(self.word2idx[w])

    def insert_singletons(self, words, p=0.5):
        """
        Replace singletons by the unknown word with a probability p.
        """
        new_words = []
        for word in words:
            if word in self.singleton and np.random.uniform() < p:
                new_words.append(self.unk_id)
            else:
                new_words.append(word)
        return new_words

    def build_prior_for_soft(self, train_insts):
        word_dict = {}
        for inst in train_insts:
            for word in inst.input.words:
                if word not in word_dict:
                    label_dict = {}
                    label_dict[self.label2idx["O"]] = 1
                    word_dict[word] = label_dict
        for inst in train_insts:
            pos = 0
            for word, label in zip(inst.input.words, inst.output):
                if not inst.is_prediction[pos]:
                    if self.label2idx[label] not in word_dict[word]:
                        word_dict[word][self.label2idx[label]] = 1
                    else:
                        word_dict[word][self.label2idx[label]] += 1
                pos += 1
        for word in word_dict:
            sum = 0
            for label_id in word_dict[word]:
                sum += word_dict[word][label_id]
            for label_id in word_dict[word]:
                word_dict[word][label_id] = word_dict[word][label_id] / sum

        return word_dict