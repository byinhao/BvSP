# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from itertools import permutations

import torch
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']
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


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            # line = line.strip().lower()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_template(template_style, at, ot, sp, ac):
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

    elif template_style == 'is because is':
        one_quad_sentence = f"{ac} is {sp} because {at} is {ot}"

    elif template_style == '( , , , )':
        one_quad_sentence = f"( {ac} , {sp} , {at} , {ot} )"
    else:
        raise ValueError
    return one_quad_sentence


def get_para_asqp_inputs_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    inputs = []
    template_ids = []
    for i in range(len(labels)):
        cur_sent = sents[i]
        cur_sent = ' '.join(cur_sent)
        cur_inputs = cur_sent

        # cur_inputs = f"{cur_sent}"
        label = labels[i]
        # Our
        # [24, 25, 7, 1, 5, 4, 11, 10, 14, 12, 2, 19, 8, 22, 23]

        # JS+order
        # [7, 19, 4, 8, 5, 11, 10, 1, 14, 6, 2, 12, 17, 13, 9]

        # entropy+all
        # [4, 5, 7, 1, 13, 10, 11, 14, 8, 12, 3, 0, 18, 2, 20]

        # JS min + all
        # [20, 21, 25, 3, 17, 13, 9, 16, 0, 6, 15, 18, 23, 22, 8]

        # random
        # [19, 8, 23, 11, 20, 16, 0, 14, 7, 1, 5, 21, 15, 17, 3]

        for ids in range(len(all_templates)):
            template_style = all_templates[ids]
            all_quad_sentences = []
            for quad in label:
                at, ac, sp, ot = quad

                man_ot = sentword2opinion[sp]

                if at == 'NULL' or at == 'null':
                    at = 'it'

                one_quad_sentence = get_template(template_style, at, ot, man_ot, ac)
                all_quad_sentences.append(one_quad_sentence)
            if template_style == '( , , , )':
                target = ' ; '.join(all_quad_sentences)
            else:
                target = ' [SSEP] '.join(all_quad_sentences)

            inputs.append(cur_inputs)
            targets.append(target)
            template_ids.append(ids)
    return inputs, targets, template_ids


def choose_reserve_data(sents, labels, few_shot_type):
    if few_shot_type == -1:
        return sents, labels, [], []
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
            # 合并
            if cur_sent_id not in choosed_data:
                choosed_data.append(cur_sent_id)
            # 不合并
            # choosed_data.append(cur_sent_id)

    train_sents, train_labels, reserve_sents, reserve_labels = [], [], [], []
    for i in range(len(sents)):
        if i not in choosed_data:
            train_sents.append(sents[i])
            train_labels.append(labels[i])
        else:
            reserve_sents.append(sents[i])
            reserve_labels.append(labels[i])
    return train_sents, train_labels, reserve_sents, reserve_labels


def get_transformed_io(data_path, few_shot_type):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, silence=False)

    sents, labels, reserve_sents, reserve_labels = choose_reserve_data(sents, labels, few_shot_type)

    inputs, targets, template_ids = get_para_asqp_inputs_targets(sents, labels)

    return inputs, targets, template_ids, reserve_sents, reserve_labels


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, few_shot_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'{data_dir}/{data_type}.txt'
        # print(f'data_dir/{data_type}.txt')
        self.max_len = max_len
        self.few_shot_type = few_shot_type
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        template_type = self.template_types[index]
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "template_types": template_type,
                "labels": self.all_labels[index]}

    def _build_examples(self):
        inputs, targets, template_ids, reserve_sents, reserve_labels = get_transformed_io(self.data_path, self.few_shot_type)

        self.reserve_sents = reserve_sents
        self.reserve_labels = reserve_labels
        self.template_types = torch.tensor(template_ids)
        self.all_labels = targets
        for i in range(len(inputs)):
            # change input and target to two strings
            # input = ' '.join(inputs[i])
            input = inputs[i]
            target = targets[i]
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
