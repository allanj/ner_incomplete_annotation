import numpy as np
import dynet as dy
from span import Span
from common.instance import Instance
from typing import List
from config import Config
from random import shuffle

def log_sum_exp(scores, num_labels):
    max_score_expr = dy.max_dim(scores)
    max_score_expr_broadcast = dy.concatenate([max_score_expr] * num_labels)
    return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))


def max_score(scores):
    max_score_expr = dy.max_dim(scores)
    return max_score_expr
    # max_score_expr_broadcast = dy.concatenate([max_score_expr] * num_labels)
    # return max_score_expr + dy.log(dy.sum_dim(dy.exp(scores - max_score_expr_broadcast), [0]))


def remove_entites(train_insts: List[Instance], config: Config) -> None:
    """
    Remove certain number of entities and make them become O label
    :param train_insts:
    :param config:
    :return:
    """
    all_spans = []
    for inst in train_insts:
        output = inst.output
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                all_spans.append(Span(start, end, output[i][2:], inst_id=inst.id))
            if output[i].startswith("S-"):
                all_spans.append(Span(i, i, output[i][2:], inst_id=inst.id))
    shuffle(all_spans)

    span_set = set()
    num_entity_removed = round(len(all_spans) * (1 - config.entity_keep_ratio))
    for i in range(num_entity_removed):
        span = all_spans[i]
        id = span.inst_id
        output = train_insts[id].output
        for j in range(span.left, span.right + 1):
            output[j] = config.O
        span_str = ' '.join(train_insts[id].input.words[span.left:(span.right + 1)])
        span_str = span.type + " " + span_str
        span_set.add(span_str)
    return span_set

def check_bies_constraint(previous: str, next: str) -> bool:
    """
    Return True if condition is satisfied.
    :param previous:
    :param next:
    :return:
    """
    if previous.startswith("B-") or previous.startswith("I-"):
        if next.startswith("I-") or next.startswith("E-"):
            return previous[2:] == next[2:]
        else:
            return False
    elif previous.startswith("S-") or previous.startswith("E-") or previous == "O":
        return next.startswith("B-") or next.startswith("S-") or next == "O"
    else:
        raise Exception("unkown previous type: %s" % (previous))


def build_insts_mask(insts, label2idx, num_labels):
    mask = []
    for id, inst in enumerate(insts):
        labels = inst.output
        inst_mask = []
        for label in labels:
            position_mask = None
            if label != "O":
                position_mask = [-1e10] * num_labels
                position_mask[label2idx[label]] = 0
            else:
                position_mask = [0] * num_labels
            inst_mask.append(position_mask)
        # print(inst_mask)
        mask.append(inst_mask)
    return mask