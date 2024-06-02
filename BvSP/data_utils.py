# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from itertools import permutations

import torch
from torch.utils.data import Dataset

from js import js_chooses
from entropy import entropy_chooses
from randoms import randoms_chooses
import json

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

with open('choosed_template.json', 'r') as file:
    history_choose_lists = json.load(file)
file.close()

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


def read_line_examples_from_file(data_path, args, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if args.do_lower:
                line = line.lower()
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


def get_para_asqp_inputs_targets_support(sents, labels, choose_lists):
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
        for ids in choose_lists:
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


def get_para_asqp_inputs_targets_quiry(sents, labels):
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
        for ids in [0]:
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


def build_quiry_sets(sents, labels, choosed_data):
    test_sents, test_labels = [], []
    for i in range(len(sents)):
        if i not in choosed_data:
            test_sents.append(sents[i])
            test_labels.append(labels[i])
    return test_sents, test_labels


def check_history(name):
    if name in history_choose_lists:
        return history_choose_lists[name]
    return None


def save_history(name, choose_lists):
    history_choose_lists[name] = choose_lists
    if "js" in name or "entropy" in name:
        if "min" in name:
            _name = "max" + name[3:]
            history_choose_lists[_name] = choose_lists[::-1]
        elif "max" in name:
            _name = "min" + name[3:]
            history_choose_lists[_name] = choose_lists[::-1]
    with open('choosed_template.json', 'w') as file:
        json.dump(history_choose_lists, file, indent=4)
    file.close()


def get_transformed_io_support_sets(data_path, open_train_reserve, args):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, args, silence=False)

    sents, labels, choosed_data = build_support_sets(sents, labels, args.few_shot_type)
    name = f"{args.method_name}_fewshot{args.few_shot_type}_seed{args.seed}"
    if args.do_lower:
        name = name + "_lower"
    choose_lists = check_history(name)
    if choose_lists is None:
        if 'js' in args.method_name:
            choose_lists = js_chooses(args, sents, labels)
        elif 'entropy' in args.method_name:
            choose_lists = entropy_chooses(args, sents, labels)
        else:
            choose_lists = randoms_chooses(args)
        save_history(name, choose_lists)
    choose_lists = choose_lists[:args.view_num]
    inputs, targets, template_ids = get_para_asqp_inputs_targets_support(sents, labels, choose_lists)

    return inputs, targets, template_ids, choosed_data, choose_lists


def get_transformed_io_quiry_sets(data_path, args, choosed_data):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, args=args, silence=False)

    sents, labels = build_quiry_sets(sents, labels, choosed_data)

    inputs, targets, template_ids = get_para_asqp_inputs_targets_quiry(sents, labels)

    return inputs, targets, template_ids


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, args, open_train_reserve=False, max_len=128, choosed_data=None):
        # './data/rest16/train.txt'
        self.data_path = f'{data_dir}/test.txt'
        self.max_len = max_len
        self.open_train_reserve = open_train_reserve
        self.choosed_data = choosed_data
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.args = args
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
        if self.data_type == 'train':
            inputs, targets, template_ids, choosed_data, choose_lists = get_transformed_io_support_sets(self.data_path,
                                                                                          self.open_train_reserve,
                                                                                          self.args)
            self.all_labels = targets
            self.choosed_data = choosed_data
            self.choose_lists = choose_lists
        else:
            inputs, targets, template_ids = get_transformed_io_quiry_sets(self.data_path, self.args, self.choosed_data)
            self.all_labels = targets
        self.template_types = torch.tensor(template_ids)
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
