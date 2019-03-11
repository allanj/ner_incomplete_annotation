import dynet_config


import argparse
import random
import numpy as np
from config import Config
from reader import Reader
from lstmcrf import BiLSTM_CRF
from partial_lstmcrf import Partial_BiLSTM_CRF
from partial_perceptron import Partial_Perceptron
from transductive_perceptron import Transductive_Perceptron
import eval
from tqdm import tqdm
import math
from utils import remove_entites,build_insts_mask
import time
import dynet as dy






def parse_arguments(parser):
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]
    for arg in dynet_args:
        parser.add_argument(arg)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--digit2zero', action="store_true", default=True)
    parser.add_argument('--dataset', type=str,default='conll2003')
    # parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt")
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.05) ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)

    ##model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--tanh_hidden_dim', type=int, default=100)
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0,1]) ##1 means use, 0 means false

    parser.add_argument('--train_num', type=int, default=-1)
    parser.add_argument('--dev_num', type=int, default=-1)
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=2000)

    ## task-specific model parameters
    parser.add_argument('--entity_keep_ratio', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='simple')
    parser.add_argument('--kfold', type=int, default=2)  ### only support 2 fold
    parser.add_argument('--large_iter', type=int, default=10)



    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def get_optimizer(model):

    if config.optimizer == "sgd":
        return dy.SimpleSGDTrainer(model, learning_rate=config.learning_rate)
    elif config.optimizer == "adam":
        return dy.AdamTrainer(model)

def get_model(config, model, insts):
    if config.model_type == "simple":
        return BiLSTM_CRF(config, model)
    elif config.model_type == "partial":
        mask = build_insts_mask(insts, config.label2idx, len(config.label2idx))
        return Partial_BiLSTM_CRF(config, model, mask)
    elif config.model_type == "partial_perceptron":
        return Partial_Perceptron(config, model)
    elif config.model_type == "transductive_perceptron":
        return Transductive_Perceptron(config, model)

    else:
        raise  Exception ("unknown model type: %s" % (config.model_type))

def train(epoch, insts, dev_insts, test_insts):

    model = dy.ParameterCollection()
    trainer = get_optimizer(model)

    bicrf = get_model(config, model, insts)
    trainer.set_clip_threshold(5)
    print("number of instances: %d" % (len(insts)))

    best_dev = [-1, -1, -1]
    best_iter = -1
    decode_test_metrics = None
    for id, inst in enumerate(insts):
        inst.id = id
    model_name = "models/"+config.dataset+"."+config.model_type+"."+str(config.entity_keep_ratio)+".m"
    for i in range(epoch):
        epoch_loss = 0
        start_time = time.time()
        k = 0
        for index in np.random.permutation(len(insts)):
            inst = insts[index]
            # for inst in tqdm(insts):
            dy.renew_cg()
            input = inst.input.word_ids
            # input = config.insert_singletons(inst.input.word_ids)
            loss = bicrf.negative_log(inst.id, input, inst.label_ids, x_chars=inst.input.char_ids)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            epoch_loss += loss_value
            k = k + 1
            if k % config.eval_freq == 0 or k == len(insts) :
                dev_metrics = decode_and_evaluate(bicrf, dev_insts, "Dev")
                test_metrics = decode_and_evaluate(bicrf, test_insts, "Test")
                if dev_metrics[2] > best_dev[2]:
                    best_dev = dev_metrics
                    best_iter = i + 1
                    decode_test_metrics = test_metrics
                    model.save(model_name)
                k = 0
        end_time = time.time()
        print("Epoch %d: %.5f, Time: %.2fs" % (i + 1, epoch_loss, end_time-start_time), flush=True)
    print("The best dev: Precision: %.2f, Recall: %.2f, F1: %.2f" % (best_dev[0], best_dev[1], best_dev[2]))
    print("The best Epoch: %d" % best_iter)
    print("The best test: Precision: %.2f, Recall: %.2f, F1: %.2f" % (decode_test_metrics[0], decode_test_metrics[1], decode_test_metrics[2]))


    model.populate(model_name)
    decode_and_evaluate(bicrf, test_insts, "Final Test")
    eval.save_results(test_insts, model_name + ".res")

def decode_and_evaluate(model, insts, name):
    ##decode step
    for inst in insts:
        dy.renew_cg()
        inst.prediction = model.decode(inst.input.word_ids, inst.input.char_ids)
    ## evaluation
    metrics = eval.evaluate(insts)
    print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, metrics[0], metrics[1], metrics[2]))
    return metrics

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    config = Config(opt)





    reader = Reader(config.digit2zero, config.dataset)

    train_insts = reader.read_from_file(config.train_file, config.train_num, True)
    dev_insts = reader.read_from_file(config.dev_file, config.dev_num, False)
    test_insts = reader.read_from_file(config.test_file, config.test_num, False)

    config.use_iobes(train_insts)
    config.use_iobes(dev_insts)
    config.use_iobes(test_insts)
    config.build_label_idx(train_insts)
    # print("All vocabulary")
    # print(reader.all_vocab)



    config.build_emb_table(reader.train_vocab, reader.test_vocab)

    config.find_singleton(train_insts)



    if config.entity_keep_ratio < 1.0:
        # print(train_insts[0].output)
        # for inst in train_insts:
        #     print(inst.output)
        print("[Info] Removing the entities")
        remove_entites(train_insts, config)
        # for inst in train_insts:
        #     print(inst.output)
        # print(train_insts[0].output)

    config.map_insts_ids(train_insts)
    config.map_insts_ids(dev_insts)
    config.map_insts_ids(test_insts)


    print("num chars: " + str(config.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(config.word2idx)))
    # print(config.word2idx)

    train(config.num_epochs, train_insts, dev_insts, test_insts)

    print(opt.mode)