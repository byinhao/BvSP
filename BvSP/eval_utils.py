# -*- coding: utf-8 -*-

import numpy as np
# This script handles the decoding functions and performance measurement

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}

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

def extract_spans_para(task, seq, seq_type, ids):
    quads = []
    if ids != 25:
        sents = [s.strip() for s in seq.split('[SSEP]')]
    else:
        sents = [s.strip() for s in seq.split(';')]
    if task == 'asqp':
        if ids <= 23:
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    index_at = s.index("[AT]")
                    index_ot = s.index("[OT]")
                    index_ac = s.index("[AC]")
                    index_sp = s.index("[SP]")

                    combined_list = [index_at, index_ot, index_ac, index_sp]
                    arg_index_list = list(np.argsort(combined_list))  # .tolist()

                    result = []
                    for i in range(len(combined_list)):
                        start = combined_list[i] + 4
                        sort_index = arg_index_list.index(i)
                        if sort_index < 3:
                            next_ = arg_index_list[sort_index + 1]
                            re = s[start: combined_list[next_]]
                        else:
                            re = s[start:]
                        result.append(re.strip())

                    at, ot, ac, sp = result

                    # if the aspect term is implicit
                    if at.lower() == 'it':
                        at = 'null'
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ac, at, sp, ot = '', '', '', ''

                quads.append((ac, at, sp, ot))
        elif ids == 24:
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    ac_sp, at_ot = s.split(' because ')
                    ac, sp = ac_sp.split(' is ')
                    at, ot = at_ot.split(' is ')

                    # if the aspect term is implicit
                    if at.lower() == 'it':
                        at = 'null'
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ac, at, sp, ot = '', '', '', ''

                quads.append((ac, at, sp, ot))
        elif ids == 25:
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    ac, sp, at, ot = s.split(',')
                    ac = ac.strip()
                    sp = sp.strip()
                    at = at.strip()
                    ot = ot.strip()
                    if len(ac) >= 1 and ac[0] == '(':
                        ac = ac[1:].strip()
                    if at.lower() == 'it':
                        at = 'null'
                    if len(ot) >= 1 and ot[-1] == ')':
                        ot = ot[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ac, at, sp, ot = '', '', '', ''

                quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_new_scores(all_preds, all_golds):
    all_final_scores = []
    for i in range(len(all_golds)):
        preds = all_preds[i]
        golds = all_golds[i]
        preds_sets = []
        for quad in preds:
            for element in quad:
                if element not in preds_sets:
                    preds_sets.append(element)
        golds_sets = []
        for quad in golds:
            for element in quad:
                if element not in golds_sets:
                    golds_sets.append(element)
        n_intersection = 0.0
        n_union = 0.0
        for ele in preds_sets:
            if ele in golds_sets:
                n_intersection += 1.0
        temp_sets = []
        for ele in golds_sets + preds_sets:
            if ele not in temp_sets:
                temp_sets.append(ele)
                n_union += 1.0
        all_scores = []
        for pred in preds:
            cur_max_score = 0.0
            for gold in golds:
                n = 0.0
                for i in range(4):
                    if pred[i] == gold[i]:
                        n += 1
                if cur_max_score < n * 0.25:
                    cur_max_score = n * 0.25
            all_scores.append(cur_max_score)
        final_score = (n_intersection / n_union) * sum(all_scores) / len(preds)
        all_final_scores.append(final_score)
    new_scores = sum(all_final_scores) / len(all_final_scores)
    return new_scores


def chooses(temp_pred, threshold):
    choose_pred = []
    nums = []
    for quad in temp_pred:
        if quad not in choose_pred:
            choose_pred.append(quad)
            nums.append(1)
        else:
            k = choose_pred.index(quad)
            nums[k] += 1
    fianl_choose = []
    for i in range(len(nums)):
        if nums[i] >= threshold:
            fianl_choose.append(choose_pred[i])
    return fianl_choose


# def compute_scores(pred_seqs, gold_seqs, sents, choose_lists):
#     """
#     Compute model performance
#     """
#     assert len(pred_seqs[0]) == len(gold_seqs)
#     num_samples = len(gold_seqs)

#     all_labels, all_preds = [], []
#     for i in range(num_samples):
#         gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold', 0)
#         temp_pred = []
#         for j, ids in enumerate(choose_lists):
#             ppp = extract_spans_para('asqp', pred_seqs[j][i], 'pred', ids)
#             temp_pred.extend(ppp)
#         pred_list = chooses(temp_pred, len(choose_lists) // 2 + len(choose_lists) % 2)
#         all_labels.append(gold_list)
#         all_preds.append(pred_list)

#     print("\nResults:")
#     scores = compute_f1_scores(all_preds, all_labels)
#     print(scores)

#     return scores, all_labels, all_preds


def compute_scores(pred_seqs, gold_seqs, sents, choose_lists):
    """
    Compute model performance
    """
    assert len(pred_seqs[0]) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds1, all_preds2, all_preds3, all_preds4, all_preds5, all_preds6, all_preds7, all_preds8, all_preds9, all_preds10 = [], [], [], [], [], [], [], [], [], [], []
    for i in range(num_samples):
        gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold', 0)
        temp_pred = []
        for j, ids in enumerate(choose_lists):
            ppp = extract_spans_para('asqp', pred_seqs[j][i], 'pred', ids)
            temp_pred.extend(ppp)
        # pred_list = chooses(temp_pred, len(choose_lists) // 2 + len(choose_lists) % 2)
        pred_list1 = chooses(temp_pred, 1)
        pred_list2 = chooses(temp_pred, 2)
        pred_list3 = chooses(temp_pred, 3)
        pred_list4 = chooses(temp_pred, 4)
        pred_list5 = chooses(temp_pred, 5)
        pred_list6 = chooses(temp_pred, 6)
        pred_list7 = chooses(temp_pred, 7)
        pred_list8 = chooses(temp_pred, 8)
        pred_list9 = chooses(temp_pred, 9)
        pred_list10 = chooses(temp_pred, 10)
        all_labels.append(gold_list)
        all_preds1.append(pred_list1)
        all_preds2.append(pred_list2)
        all_preds3.append(pred_list3)
        all_preds4.append(pred_list4)
        all_preds5.append(pred_list5)
        all_preds6.append(pred_list6)
        all_preds7.append(pred_list7)
        all_preds8.append(pred_list8)
        all_preds9.append(pred_list9)
        all_preds10.append(pred_list10)

    print("\nResults1:")
    scores = compute_f1_scores(all_preds1, all_labels)
    print(scores)
    print()

    print("\nResults2:")
    scores = compute_f1_scores(all_preds2, all_labels)
    print(scores)
    print()

    print("\nResults3:")
    scores = compute_f1_scores(all_preds3, all_labels)
    print(scores)
    print()

    print("\nResults4:")
    scores = compute_f1_scores(all_preds4, all_labels)
    print(scores)
    print()

    print("\nResults5:")
    scores = compute_f1_scores(all_preds5, all_labels)
    print(scores)
    print()

    print("\nResults6:")
    scores = compute_f1_scores(all_preds6, all_labels)
    print(scores)
    print()

    print("\nResults7:")
    scores = compute_f1_scores(all_preds7, all_labels)
    print(scores)
    print()

    print("\nResults8:")
    scores = compute_f1_scores(all_preds8, all_labels)
    print(scores)
    print()

    print("\nResults9:")
    scores = compute_f1_scores(all_preds9, all_labels)
    print(scores)
    print()

    print("\nResults10:")
    scores = compute_f1_scores(all_preds10, all_labels)
    print(scores)
    print()

    return scores, all_labels, all_preds8
