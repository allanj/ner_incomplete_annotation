import argparse
import random
import numpy as np
from typing import Tuple
from config import Reader, Config, ContextEmb, lr_decay, simple_batching, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances
from config import remove_entites
import time
from model.neuralcrf import NNCRF
import torch
from typing import List
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec
import pickle
import tarfile
import shutil
import math
import itertools


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="conll2003")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)  ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--entity_keep_ratio', type=float, default=0.5, help="the percentage of entities to be kept", choices=np.arange(0, 1.1, 0.1))
    parser.add_argument('--num_outer_iterations', type= int , default=10, help="Number of outer iterations for cross validation")


    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="contextual word embedding")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args




def train_model(config: Config, train_insts: List[List[Instance]], dev_insts: List[Instance], test_insts: List[Instance]):
    train_num = sum([len(insts) for insts in train_insts])
    print("[Training Info] number of instances: %d" % (train_num))

    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    # if os.path.exists(model_folder):
    #     raise FileExistsError(f"The folder {model_folder} exists. Please either delete it or create a new one "
    #                           f"to avoid override.")

    print("[Training Info] The model will be saved to: %s.tar.gz" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    num_outer_iterations = config.num_outer_iterations
    for iter in range(num_outer_iterations):
        print(f"[Training Info] Running for {iter}th large iterations.")
        model_names = [] #model names for each fold
        train_batches = [batching_list_instances(config, insts) for insts in train_insts]
        for fold_id, folded_train_insts in enumerate(train_insts):
            print(f"[Training Info] Training fold {fold_id}.")
            model_name = model_folder + f"/lstm_crf_{fold_id}.m"
            model_names.append(model_name)
            train_one(config=config, train_batches = train_batches[fold_id],
                      dev_insts=dev_insts, dev_batches=dev_batches, model_name=model_name)

        # assign hard prediction to other folds
        print("\n\n[Data Info] Assigning labels for the HARD approach")

        for fold_id, folded_train_insts in enumerate(train_insts):
            model = NNCRF(config)
            model_name = model_names[fold_id]
            model.load_state_dict(torch.load(model_name))
            hard_constraint_predict(config=config, model=model,
                                    fold_batches = train_batches[1-fold_id],
                                    folded_insts=train_insts[1 - fold_id])  ## set a new label id
        print("\n\n")

        print("[Training Info] Training the final model" )
        all_train_insts = list(itertools.chain.from_iterable(train_insts))
        model_name = model_folder + "/final_lstm_crf.m"
        config_name = model_folder + "/config.conf"
        res_name = res_folder + "/lstm_crf.results".format()
        all_train_batches = batching_list_instances(config= config, insts=all_train_insts)
        model = train_one(config = config, train_batches=all_train_batches, dev_insts=dev_insts, dev_batches=dev_batches,
                          model_name=model_name, config_name=config_name,test_insts=test_insts, test_batches=test_batches,result_filename=res_name)
        print("Archiving the best Model...")
        with tarfile.open(model_folder + "/" + model_folder + ".tar.gz", "w:gz") as tar:
            tar.add(model_folder, arcname=os.path.basename(model_folder))
        # print("The best dev: %.2f" % (best_dev[0]))
        # print("The corresponding test: %.2f" % (best_test[0]))
        # print("Final testing.")
        model.load_state_dict(torch.load(model_name))
        model.eval()
        evaluate_model(config, model, test_batches, "test", test_insts)
        write_results(res_name, test_insts)

def hard_constraint_predict(config: Config, model: NNCRF, fold_batches: List[Tuple], folded_insts:List[Instance], model_type:str = "hard"):
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    for batch in fold_batches:
        one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(batch)
        batch_max_ids = batch_max_ids.cpu().numpy()
        word_seq_lens = batch[1].cpu().numpy()
        for idx in range(len(batch_max_ids)):
            length = word_seq_lens[idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            one_batch_insts[idx].output_ids = prediction
        batch_id += 1


def train_one(config: Config, train_batches: List[Tuple], dev_insts: List[Instance],
              dev_batches: List[Tuple], model_name: str, test_insts: List[Instance] = None,
              test_batches: List[Tuple] = None, config_name: str = None, result_filename: str = None) -> NNCRF:
    model = NNCRF(config)
    model.train()
    optimizer = get_optimizer(config, model)
    epoch = config.num_epochs
    best_dev_f1 = -1
    saved_test_metrics = None
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in np.random.permutation(len(train_batches)):
            model.train()
            loss = model(*train_batches[index])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        # metric is [precision, recall, f_score]
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        if test_insts is not None:
            test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        if dev_metrics[2] > best_dev_f1:
            print("saving the best model...")
            best_dev_f1 = dev_metrics[2]
            if test_insts is not None:
                saved_test_metrics = test_metrics
            torch.save(model.state_dict(), model_name)
            # # Save the corresponding config as well.
            if config_name:
                f = open(config_name, 'wb')
                pickle.dump(config, f)
                f.close()
            if result_filename:
                write_results(result_filename, test_insts)
        model.zero_grad()
    if test_insts is not None:
        print(f"The best dev F1: {best_dev_f1}" )
        print(f"The corresponding test: {saved_test_metrics}")
    return model

def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(batch)
        metrics += evaluate_batch_insts(batch_insts=one_batch_insts,
                                        batch_pred_ids = batch_max_ids,
                                        batch_gold_ids=batch[-1],
                                        word_seq_lens= batch[1], idx2label=config.idx2labels)
        batch_id += 1
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
    return [precision, recall, fscore]




def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('[Data Info] Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    conf.use_iobes(trains + devs + tests)
    conf.build_label_idx(trains + devs + tests)

    conf.build_word_idx(trains, devs, tests)
    conf.build_emb_table()
    conf.map_insts_ids(devs + tests)
    print("[Data Info] num chars: " + str(conf.num_char))
    print("[Data Info] num words: " + str(len(conf.word2idx)))

    print(f"[Data Info] Removing {conf.entity_keep_ratio*100}% of entities from the training set")

    print("[Data Info] Removing the entities")
    span_set = remove_entites(trains, conf)
    # print(f"entities removed: {span_set}")
    conf.map_insts_ids(trains)
    random.shuffle(trains)
    for inst in trains:
        inst.is_prediction = [False] * len(inst.input)
        for pos, label in enumerate(inst.output):
            if label == conf.O:
                inst.is_prediction[pos] = True

    num_insts_in_fold = math.ceil(len(trains) / conf.num_folds)
    trains = [trains[i * num_insts_in_fold: (i + 1) * num_insts_in_fold] for i in range(conf.num_folds)]
    train_model(config=conf, train_insts=trains, dev_insts=devs, test_insts=tests)

if __name__ == "__main__":
    main()
