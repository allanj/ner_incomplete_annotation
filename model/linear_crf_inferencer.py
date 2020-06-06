

import torch.nn as nn
import torch

from config import log_sum_exp_pytorch, START, STOP, PAD
from typing import Tuple
from overrides import overrides

class LinearCRF(nn.Module):


    def __init__(self, config, print_info: bool = True):
        super(LinearCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never -> start. end never -> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0
        self.add_constraint_for_iobes(init_transition)

        self.transition = nn.Parameter(init_transition).to(self.device)

    def add_constraint_for_iobes(self, transition: torch.Tensor):
        print("[Info] Adding IOBES constraints")
        ## add constraint:
        for prev_label in self.labels:
            if prev_label == START or prev_label == STOP or prev_label == PAD:
                continue
            for next_label in self.labels:
                if next_label == START or next_label == STOP or next_label == PAD:
                    continue
                if prev_label == "O" and (next_label.startswith("I-") or next_label.startswith("E-")):
                    transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith("B-") or prev_label.startswith("I-"):
                    if next_label.startswith("O") or next_label.startswith("B-") or next_label.startswith("S-"):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                    elif prev_label[2:] != next_label[2:]:
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
                if prev_label.startswith("S-") or prev_label.startswith("E-"):
                    if next_label.startswith("I-") or next_label.startswith("E-"):
                        transition[self.label2idx[prev_label], self.label2idx[next_label]] = -10000.0
        ##constraint for start and end
        for label in self.labels:
            if label.startswith("I-") or label.startswith("E-"):
                transition[self.start_idx, self.label2idx[label]] = -10000.0
            if label.startswith("I-") or label.startswith("B-"):
                transition[self.label2idx[label], self.end_idx] = -10000.0

    @overrides
    def forward(self, lstm_scores, word_seq_lens, tags, mask, marginals = None):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores=  self.calculate_all_scores(lstm_scores= lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask) if marginals is None else \
            self.forward_labeled_with_marginal(all_scores, word_seq_lens, marginals)
        return unlabed_score, labeled_score

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: (batch_size) for the normalization scores
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)

    def forward_labeled_with_marginal(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, marginals: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels) from lstm scores.
        :param word_seq_lens: (batch_size)
        :param marginals: shape (batch x max_seq_len x num_labels), the log-space score of the marginal probability
        :return: (batch_size) for the normalization scores
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0, self.start_idx, :]  ## the first position of all labels = (the transition from start - > all labels) + current emission.
        alpha[:, 0, :] += marginals[:, 0, :]

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) \
                                 + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            alpha[:, word_idx, :] += marginals[:, word_idx, :]

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1,  word_seq_lens.view(batch_size, 1, 1).
                                  expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, lstm_scores: torch.Tensor, sent_len: torch.Tensor, annotation_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param lstm_scores: (batch_size, sent_len, num_labels)
        :param sent_len: (batch_size)
        :param annotation_mask: (batch_size, sent_len, num_labels)
        """
        all_scores = self.calculate_all_scores(lstm_scores)
        bestScores, decodeIdx = self.constrainted_viterbi_decode(all_scores, sent_len, annotation_mask)
        return bestScores, decodeIdx

    def constrainted_viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, annotation_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        if annotation_mask is not None:
            annotation_mask = annotation_mask.float().log()
        # sent_len =
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        if annotation_mask is not None:
            scoresRecord[:, 0, :] += annotation_mask[:, 0, :]
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            if annotation_mask is not None:
                scoresIdx += annotation_mask[:, wordIdx, :].view(batchSize, 1, self.label_size).expand(batchSize, self.label_size, self.label_size)

            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx

    def compute_constrained_marginal(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor, annotation_mask: torch.LongTensor) -> torch.Tensor:
        """
        Note: This function is not used unless you want to compute the marginal probability
        Forward-backward algorithm to compute the marginal probability (in log space)
        Basically, we follow the `backward` algorithm to obtain the backward scores.
        :param lstm_scores:   shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :param annotation_mask: shape: (batch_size, sent_len, label_size)
        :return: Marginal score. If you want probability, you need to use `torch.exp` to convert it into probability
                shape: (batch_size, sent_len, label_size)
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        mask = annotation_mask.float().log()
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)
        beta = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        perm_idx = torch.zeros(batch_size, seq_len).to(self.device)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        alpha[:, 0, :] = scores[:, 0, self.start_idx, :]  ## the first position of all labels = (the transition from start - > all labels) + current emission.
        alpha[:, 0, :] += mask[:, 0, :]
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) \
                                 + scores[ :, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            alpha[:, word_idx, :] += mask[:, word_idx, :]
            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) \
                                 + rev_score[ :, word_idx, :, :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            beta[:, word_idx, :] += mask[:, word_idx, :]

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1,  word_seq_lens.view(batch_size, 1, 1).
                                  expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size, 1, 1).\
            expand(batch_size, seq_len, self.label_size)

        ## Because we need to use the beta variable later, we need to reverse back
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        # `alpha + beta - last_alpha` is the standard way to obtain the marginal
        # However, we have two emission scores overlap at each position, thus, we need to subtract one emission score
        return alpha + beta - last_alpha - lstm_scores
