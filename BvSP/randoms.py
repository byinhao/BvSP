import argparse
import os
import random
from itertools import permutations

import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from transformers import T5Tokenizer
from model import MyT5ForConditionalGeneration
import codecs as cs

all_templates = [
    '[AT] [OT] [AC] [SP]', '[AT] [OT] [SP] [AC]', '[AT] [AC] [OT] [SP]',
    '[AT] [AC] [SP] [OT]', '[AT] [SP] [OT] [AC]', '[AT] [SP] [AC] [OT]',
    '[OT] [AT] [AC] [SP]', '[OT] [AT] [SP] [AC]', '[OT] [AC] [AT] [SP]',
    '[OT] [AC] [SP] [AT]', '[OT] [SP] [AT] [AC]', '[OT] [SP] [AC] [AT]',
    '[AC] [AT] [OT] [SP]', '[AC] [AT] [SP] [OT]', '[AC] [OT] [AT] [SP]',
    '[AC] [OT] [SP] [AT]', '[AC] [SP] [AT] [OT]', '[AC] [SP] [OT] [AT]',
    '[SP] [AT] [OT] [AC]', '[SP] [AT] [AC] [OT]', '[SP] [OT] [AT] [AC]',
    '[SP] [OT] [AC] [AT]', '[SP] [AC] [AT] [OT]', '[SP] [AC] [OT] [AT]',
    'is because is', '( , , , )'
]


def build_support_sets(sents, labels, few_shot_type):
    cate_dicts = {}
    for i in range(len(sents)):
        for quad in labels[i]:
            a, c, s, o = quad
            if c not in cate_dicts:
                cate_dicts[c] = [i]
            elif i in cate_dicts[c]:
                continue
            else:
                cate_dicts[c].append(i)
    choosed_data = []
    for c in cate_dicts:
        cur_candidate = cate_dicts[c]
        cur_choose_ids = random.sample(range(0, len(cur_candidate)), few_shot_type)
        for ids in cur_choose_ids:
            cur_sent_id = cur_candidate[ids]
            if cur_sent_id not in choosed_data:
                choosed_data.append(cur_sent_id)
    train_sents, train_labels = [], []
    for i in range(len(sents)):
        if i in choosed_data:
            train_sents.append(sents[i])
            train_labels.append(labels[i])
    return train_sents, train_labels, choosed_data


def randoms_chooses(args):
    choose_random = random.sample(range(0, len(all_templates)), args.view_num)
    return choose_random
