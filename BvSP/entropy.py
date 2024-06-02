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
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}


def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum()
    return entropy


def get_template(template_style, at, ot, sp, ac, lens):
    if 'AT' in template_style:

        quad = [f"[AT] {at}",
                f"[OT] {ot}",
                f"[AC] {ac}",
                f"[SP] {sp}"]
        x = permutations(quad)

        for each in x:
            order = []
            content = []
            for e in each:
                order.append(e[0:4])
                content.append(e[4:])
            order_name = " ".join(order)
            if order_name == template_style:
                one_quad_sentence = " ".join(each)
                cur_dtype = []

                index_at = one_quad_sentence.split().index("[AT]")
                index_ot = one_quad_sentence.split().index("[OT]")
                index_ac = one_quad_sentence.split().index("[AC]")
                index_sp = one_quad_sentence.split().index("[SP]")

                combined_list = [index_at, index_ot, index_ac, index_sp]
                arg_index_list = list(np.argsort(combined_list))  # .tolist()

                for ii in range(len(combined_list)):
                    start = combined_list[ii] + 1
                    sort_index = arg_index_list.index(ii)
                    if sort_index < 3:
                        next_ = arg_index_list[sort_index + 1]
                        cur_dtype.append((start + lens, combined_list[next_] + lens))
                    else:
                        cur_dtype.append((start + lens, len(one_quad_sentence.split()) + lens))
    elif template_style == 'is because is':
        one_quad_sentence = f"{ac} is {sp} because {at} is {ot}"

        ac_st, ac_end = lens, lens + len(ac.split())
        sp_st, sp_end = ac_end + 1, ac_end + 1 + len(sp.split())
        at_st, at_end = sp_end + 1, sp_end + 1 + len(at.split())
        ot_st, ot_end = at_end + 1, at_end + 1 + len(ot.split())
        cur_dtype = [(at_st, at_end), (ot_st, ot_end), (ac_st, ac_end), (sp_st, sp_end)]
    elif template_style == '( , , , )':
        one_quad_sentence = f"{ac} , {sp} , {at} , {ot}"
        ac_st, ac_end = lens, lens + len(ac.split())
        sp_st, sp_end = ac_end + 1, ac_end + 1 + len(sp.split())
        at_st, at_end = sp_end + 1, sp_end + 1 + len(at.split())
        ot_st, ot_end = at_end + 1, at_end + 1 + len(ot.split())
        cur_dtype = [(at_st, at_end), (ot_st, ot_end), (ac_st, ac_end), (sp_st, sp_end)]
    else:
        raise ValueError
    return one_quad_sentence, cur_dtype


def get_range(tokens, tokenizer):
    token_start = 0
    token_range = []
    for ii, w, in enumerate(tokens):
        token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
        token_range.append([token_start, token_end - 1])
        token_start = token_end
    return token_range


def get_target_span(target, targets_dtypes, tokenizer, device):

    reorder = torch.zeros(128).long().to(device)

    target_ranges = get_range(target, tokenizer)

    kkk = 0
    for temp in targets_dtypes:
        for ttt in temp:
            token_st = ttt[0]
            token_end = ttt[1] - 1
            ids_st = target_ranges[token_st][0]
            ids_end = target_ranges[token_end][1] + 1
            reorder[kkk: kkk + ids_end - ids_st] = torch.arange(ids_st, ids_end)
            kkk += ids_end - ids_st
    reorder = reorder[:kkk]
    return reorder


def entropy_chooses(args, all_sents, all_labels):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = MyT5ForConditionalGeneration.from_pretrained(args.model_name_or_path, prefix_lenth=15, prefix_dropout=0.1).to(device)
    model.eval()
    all_values = []
    for i in range(len(all_sents)):
        input = ' '.join(all_sents[i])
        labels = all_labels[i]

        input_ids = None
        input_masks = None
        target_ids = None
        target_masks = None
        all_reorder = None

        for template_style in all_templates:
            target = []
            targets_dtypes = []
            lens = 0
            for l_id in range(len(labels)):
                at, ac, sp, ot = labels[l_id]
                man_ot = sentword2opinion[sp]

                if at == 'NULL' or at == 'null':
                    at = 'it'
                tmp_target, cur_dtype = get_template(template_style, at, ot, man_ot, ac, lens)
                target.append(tmp_target)
                targets_dtypes.append(cur_dtype)
                lens += len(tmp_target.split()) + 1
            if template_style == '( , , , )':
                target = ' ; '.join(target)
            else:
                target = ' [SSEP] '.join(target)

            tokenized_input = tokenizer.batch_encode_plus(
                [input], max_length=128, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = tokenizer.batch_encode_plus(
                [target], max_length=256, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            reorder = get_target_span(target.split(), targets_dtypes, tokenizer, device)
            if input_ids is None:
                input_ids = tokenized_input['input_ids']
                input_masks = tokenized_input['attention_mask']
                target_ids = tokenized_target['input_ids']
                target_masks = tokenized_target['attention_mask']
                all_reorder = reorder.unsqueeze(0)
            else:
                input_ids = torch.cat((input_ids, tokenized_input['input_ids']), dim=0)
                input_masks = torch.cat((input_masks, tokenized_input['attention_mask']), dim=0)
                target_ids = torch.cat((target_ids, tokenized_target['input_ids']), dim=0)
                target_masks = torch.cat((target_masks, tokenized_target['attention_mask']), dim=0)
                all_reorder = torch.cat((all_reorder, reorder.unsqueeze(0)), dim=0)

        template_types = torch.arange(len(all_templates))

        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), attention_mask=input_masks.to(device), template_types=template_types.to(device),
                            labels=target_ids.to(device), decoder_attention_mask=target_masks.to(device))
            lm_logits = outputs.logits
            entropy = []
            for i in range(lm_logits.size()[0]):
                cur_length = (all_reorder[i, :] != 0).sum()
                cur_reorder = all_reorder[i, :cur_length]

                ent = calc_entropy(lm_logits[i][cur_reorder])
                entropy.append(ent.item())

        if all_values == []:
            all_values = entropy
        else:
            for i in range(len(entropy)):
                all_values[i] += entropy[i]

    all_values = numpy.array(all_values)
    ttt = numpy.argsort(all_values)
    if 'min' in args.method_name:
        entropy_min_all_ids = []
        for i in range(len(all_templates)):
            ids = int(ttt[i])
            entropy_min_all_ids.append(ids)
        return entropy_min_all_ids
    else:
        ttt = ttt[::-1]
        entropy_max_all_ids = []
        for i in range(len(all_templates)):
            ids = int(ttt[i])
            entropy_max_all_ids.append(ids)
        return entropy_max_all_ids
