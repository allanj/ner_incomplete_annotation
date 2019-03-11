import dynet_config


import argparse
import random
import numpy as np
from config import Config
from reader import Reader
from lstmcrf import BiLSTM_CRF
from soft_lstmcrf import Soft_BiLSTM_CRF
import eval
from tqdm import tqdm
import math
from utils import remove_entites,build_insts_mask
import time
import dynet as dy
import itertools




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
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt")
    # parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.05) ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=30)

    ##model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--tanh_hidden_dim', type=int, default=100)
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0,1])

    parser.add_argument('--train_num', type=int, default=-1)
    parser.add_argument('--dev_num', type=int, default=-1)
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--eval_freq', type=int, default=2000)

    ## task-specific model parameters
    parser.add_argument('--entity_keep_ratio', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='our_hard')
    parser.add_argument('--kfold', type=int, default=2)  ### only support 2 fold
    parser.add_argument('--large_iter', type=int, default=10)
    parser.add_argument('--initialization',type=str,default="o_prefer",help="o_prefer or prior")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def get_optimizer(model):

    if config.optimizer == "sgd":
        return dy.SimpleSGDTrainer(model, learning_rate=config.learning_rate)
    elif config.optimizer == "adam":
        return dy.AdamTrainer(model)


def get_model(config, model):
    if config.model_type == "our_hard":
        return BiLSTM_CRF(config, model)
    elif config.model_type == "our_soft":
        return Soft_BiLSTM_CRF(config, model)
    else:
        raise  Exception ("unknown model type: %s" % (config.model_type))

def train(iterations, epoch, all_insts, dev_insts, test_insts, init):


    print("number of folds: %d" % (len(all_insts)))
    for id, insts in enumerate(all_insts):
        print("fold %d has %d insts." % (id, len(insts)))



    print("[Info] Running for %d large iterations." % (iterations))
    for iteration in range(iterations):
        m_names = []
        print("[Info] Running for %dth large iterations." % (iteration))
        for fold_id, insts in enumerate(all_insts):
            model = dy.ParameterCollection()
            trainer = get_optimizer(model)

            bicrf = get_model(config, model)
            trainer.set_clip_threshold(5)
            best_dev = [-1, -1, -1]
            best_iter = -1
            print("[Info] Training for %dth Fold." % (fold_id))
            model_name = "models/" + config.dataset + "." + config.model_type + ".fold_" + str(fold_id) + "_"+str(config.entity_keep_ratio)\
                         + "_"+init+".m"
            m_names.append(model_name)
            for i in range(epoch):
                epoch_loss = 0
                start_time = time.time()
                k = 0
                for id, inst in enumerate(insts):
                    # for inst in tqdm(insts):
                    dy.renew_cg()
                    input = inst.input.word_ids
                    # input = config.insert_singletons(inst.input.word_ids)
                    loss = bicrf.negative_log(id, input, inst.label_ids, x_chars=inst.input.char_ids, marginals=inst.marginals)
                    loss_value = loss.value()
                    loss.backward()
                    trainer.update()
                    epoch_loss += loss_value
                    k = k + 1
                    if k % config.eval_freq == 0 or k == len(insts) :
                        dev_metrics, _ = evaluate(bicrf, dev_insts)
                        if dev_metrics[2] > best_dev[2]:
                            best_dev = dev_metrics
                            best_iter = i + 1
                            model.save(model_name)
                        k = 0
                end_time = time.time()
                print("Epoch %d: %.5f, Time: %.2fs" % (i + 1, epoch_loss, end_time-start_time), flush=True)

            print("The best dev: Precision: %.2f, Recall: %.2f, F1: %.2f" % (best_dev[0], best_dev[1], best_dev[2]))
            print("The best Epoch: %d" % best_iter)
            # print("The best test: Precision: %.2f, Recall: %.2f, F1: %.2f" % (decode_test_metrics[0], decode_test_metrics[1], decode_test_metrics[2]))


        print("Assigning Marginal for soft or labels for Hard")
        model = dy.ParameterCollection()
        bicrf = get_model(config, model)
        for fold_id, insts in enumerate(all_insts):
            model.populate(m_names[fold_id])
            evaluate_and_predict(bicrf, all_insts[1-fold_id], config.model_type) ## set a new label id
        # print(all_insts[0][0].label_ids)
        # print(all_insts[1][0].label_ids)


        ##train all
    all_train_insts = list(itertools.chain.from_iterable(all_insts))
    # print("Finish all iterations. Now train all datasets: " + str(len(all_train_insts)))
    print("Finish %dth iterations. Now train all datasets: %d" % (iterations, len(all_train_insts)))
    model = dy.ParameterCollection()
    trainer = get_optimizer(model)

    bicrf = get_model(config, model)
    trainer.set_clip_threshold(5)
    best_dev = [-1, -1, -1]
    best_iter = -1
    decode_test_metrics = None
    model_name = "models/" + config.dataset + "." + config.model_type +"."+str(config.entity_keep_ratio)\
                         + "_"+init+".final.m"
    for i in range(epoch):
        epoch_loss = 0
        start_time = time.time()
        k = 0
        for id, inst in enumerate(all_train_insts):
            # for inst in tqdm(insts):
            dy.renew_cg()
            input = inst.input.word_ids
            # input = config.insert_singletons(inst.input.word_ids)
            loss = bicrf.negative_log(id, input, inst.label_ids, x_chars=inst.input.char_ids, marginals=inst.marginals)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            epoch_loss += loss_value
            k = k + 1
            if k % config.eval_freq == 0 or k == len(all_train_insts):
                dev_metrics, test_metrics = evaluate(bicrf, dev_insts, test_insts)
                if dev_metrics[2] > best_dev[2]:
                    best_dev = dev_metrics
                    best_iter = i + 1
                    decode_test_metrics = test_metrics
                    model.save(model_name)
                k = 0
        end_time = time.time()
        print("Epoch %d: %.5f, Time: %.2fs" % (i + 1, epoch_loss, end_time - start_time), flush=True)

    print("[Final] The best dev: Precision: %.2f, Recall: %.2f, F1: %.2f" % (best_dev[0], best_dev[1], best_dev[2]))
    print("[Final] The best Epoch: %d" % best_iter)
    print("[Final] The best test: Precision: %.2f, Recall: %.2f, F1: %.2f" % (
    decode_test_metrics[0], decode_test_metrics[1], decode_test_metrics[2]))

    ##Saving the test results
    model = dy.ParameterCollection()
    bicrf = get_model(config, model)
    model.populate(model_name)
    for test_inst in test_insts:
        dy.renew_cg()
        test_inst.prediction = bicrf.decode(test_inst.input.word_ids, test_inst.input.char_ids)
    test_metrics = eval.evaluate(test_insts)
    print("[Final Test set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (test_metrics[0], test_metrics[1], test_metrics[2]))
    eval.save_results(test_insts, model_name + ".res")


def evaluate_and_predict(model, insts, model_type):
    for inst in insts:
        dy.renew_cg()
        if model_type == "our_hard":
            inst.label_ids = model.decode(inst.input.word_ids, inst.input.char_ids, True,
                                           inst.label_ids, inst.is_prediction)
        elif model_type == "our_soft":
            inst.marginals = model.max_marginal_decode(inst.input.word_ids, inst.input.char_ids,
                                                       inst.label_ids, inst.is_prediction)
        else:
            raise Exception("unknown model type: " + model_type)
    # dev_metrics = eval.evaluate(dev_insts)
    # print("precision "+str(metrics[0]) + " recall:" +str(metrics[1])+" f score : " + str(metrics[2]))
    # print("[Dev set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (dev_metrics[0], dev_metrics[1], dev_metrics[2]))


def evaluate(model, dev_insts, test_insts = None):
    ## evaluation
    for dev_inst in dev_insts:
        dy.renew_cg()
        dev_inst.prediction = model.decode(dev_inst.input.word_ids, dev_inst.input.char_ids)
    dev_metrics = eval.evaluate(dev_insts)
    # print("precision "+str(metrics[0]) + " recall:" +str(metrics[1])+" f score : " + str(metrics[2]))
    print("[Dev set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (dev_metrics[0], dev_metrics[1], dev_metrics[2]))
    ## evaluation
    test_metrics = None
    if test_insts != None:
        for test_inst in test_insts:
            dy.renew_cg()
            test_inst.prediction = model.decode(test_inst.input.word_ids, test_inst.input.char_ids)
        test_metrics = eval.evaluate(test_insts)
        print("[Test set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (test_metrics[0], test_metrics[1], test_metrics[2]))
    return dev_metrics, test_metrics

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


    if config.entity_keep_ratio < 1.0:
        # print(train_insts[0].output)
        # for inst in train_insts:
        #     print(inst.output)
        print("[Info] Removing the entities")
        span_set = remove_entites(train_insts, config)
        eval.save_spans(span_set, "models/" + config.dataset + "." + config.model_type +"."+str(config.entity_keep_ratio)+ ".spans.txt")


        config.find_singleton(train_insts)
        config.map_insts_ids(train_insts)
        config.map_insts_ids(dev_insts)
        config.map_insts_ids(test_insts)

        # print(train_insts[0].output)
        # print(train_insts[0].label_ids)
        # print(train_insts[1].output)
        # print(train_insts[1].label_ids)

        random.shuffle(train_insts)
        for inst in train_insts:
            inst.is_prediction = [False] * len(inst.input)
            for pos, label in enumerate(inst.output):
                if label == config.O:
                    inst.is_prediction[pos] = True

        if config.model_type == "our_soft":
            ## initalize the marginal
            print("[Soft Model] Initializing Marginal Probability")
            if opt.initialization == "prior":
                word_dict = config.build_prior_for_soft(train_insts)
                for inst in train_insts:
                    inst.marginals = []
                    for pos, label in enumerate(inst.output):
                        word = inst.input.words[pos]
                        if inst.is_prediction[pos]:
                            pos_marginal = [-1e10] * len(config.label2idx)
                            for label_id in word_dict[word]:
                                # print(math.log(word_dict[word][label_id]))
                                pos_marginal[label_id] = math.log(word_dict[word][label_id])
                        else:
                            pos_marginal = [-1e10] * len(config.label2idx)
                            pos_marginal[config.label2idx[label]] = 0
                        inst.marginals.append(pos_marginal)
            elif opt.initialization == "random":
                for inst in train_insts:
                    inst.marginals = []
                    for pos, label in enumerate(inst.output):
                        word = inst.input.words[pos]
                        if inst.is_prediction[pos]:
                            pos_marginal = np.random.rand(len(config.label2idx))
                            pos_marginal = np.log(pos_marginal/np.sum(pos_marginal)).tolist()
                        else:
                            pos_marginal = [-1e10] * len(config.label2idx)
                            pos_marginal[config.label2idx[label]] = 0
                        inst.marginals.append(pos_marginal)
            elif opt.initialization == "uniform":
                for inst in train_insts:
                    inst.marginals = []
                    for pos, label in enumerate(inst.output):
                        word = inst.input.words[pos]
                        if inst.is_prediction[pos]:
                            pos_marginal = [0] * len(config.label2idx)
                        else:
                            pos_marginal = [-1e10] * len(config.label2idx)
                            pos_marginal[config.label2idx[label]] = 0
                        inst.marginals.append(pos_marginal)
            else:
                ##o_prefer.
                for inst in train_insts:
                    inst.marginals = []
                    for label in inst.output:
                        pos_marginal = [-1e10] * len(config.label2idx)
                        pos_marginal[config.label2idx[label]] = 0
                        inst.marginals.append(pos_marginal)




        num_insts_in_fold = math.ceil(len(train_insts) / config.num_folds)
        all_insts = [train_insts[i*num_insts_in_fold : (i+1)*num_insts_in_fold]   for i in range(config.num_folds)]



        print("num chars: " + str(config.num_char))
        # print(str(config.char2idx))

        print("num words: " + str(len(config.word2idx)))
        # print(config.word2idx)

        train(config.large_iter, config.num_epochs, all_insts, dev_insts, test_insts, opt.initialization)

        print(opt.mode)
    else:
        print("Our approach only works")