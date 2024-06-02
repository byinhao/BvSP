# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time

from torch.backends import cudnn
from tqdm import tqdm
import random
import codecs as cs
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AdamW, T5Tokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores
from model import MyT5ForConditionalGeneration

import time

start_time = time.perf_counter()

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--model_name_or_path", default='pre-train/outputs', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        default=True,
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=5,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    # few_shot parameters
    parser.add_argument("--open_train_reserve", default=False, type=bool)
    parser.add_argument("--few_shot_type", default=1, type=int)
    parser.add_argument("--do_lower", default=1, type=int)

    parser.add_argument("--data_dir", default='data/FSQP', type=str)
    parser.add_argument("--output_dir", default='outputs', type=str)

    parser.add_argument("--method_name", default='min_js', type=str)

    parser.add_argument("--view_num", default=3, type=int)
    parser.add_argument("--device", default=0, type=int)

    args = parser.parse_args()

    print(f"few_shot_type={args.few_shot_type}\nmodel_name_or_path={args.model_name_or_path}\ndo_lower={args.do_lower}")
    print(f"method_name={args.method_name}\nview_num={args.view_num}\ndevice={args.device}")

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.data_dir, data_type=type_path, max_len=args.max_seq_length,
                       open_train_reserve=args.open_train_reserve, args=args)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, template_types=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            template_types=template_types
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            template_types=batch['template_types']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):

        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)

        self.choosed_data = train_dataset.choosed_data
        args.choose_lists = train_dataset.choose_lists

        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                shuffle=True, num_workers=4)
        # dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
        #                         drop_last==True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def compute_confidence(scores, sequences):
    scores_tensor = scores[0].unsqueeze(1)
    sequences = sequences[:, 1:]
    for i in range(1, len(scores)):
        cur_scores_tensor = scores[i].unsqueeze(1)
        scores_tensor = torch.cat([scores_tensor, cur_scores_tensor], dim=1)
    softmax = nn.Softmax(dim=-1)
    scores_tensor = softmax(scores_tensor)
    results = torch.gather(scores_tensor, -1, sequences.unsqueeze(-1)).squeeze(-1)
    length = (sequences != 0).sum(-1)
    confidence = []
    for i in range(results.shape[0]):
        _cur = results[i][:length[i]]
        confidence.append(_cur.mean().item())
    return confidence


def evaluate(data_loader, model, sents):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu - 1}')
    model.model.to(device)

    model.model.eval()

    targets = []
    outputs = [[] for i in range(15)]
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # need to push the data to device
            for i, ids in enumerate(args.choose_lists):
                template_types = torch.tensor([ids] * batch['source_ids'].shape[0])
                outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                            attention_mask=batch['source_mask'].to(device),
                                            template_types=template_types.to(device),
                                            max_length=args.max_seq_length)
                dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                outputs[i].extend(dec)
            target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
            targets.extend(target)

    scores, all_labels, all_preds = compute_scores(outputs, targets, sents, args.choose_lists)

    f_gold_w = cs.open(f'{args.output_dir}/{args.few_shot_type}_gold_lower{args.do_lower}_{args.seed}_bs{args.train_batch_size}_lr{args.learning_rate}.txt', 'w')
    f_pred_w = cs.open(f'{args.output_dir}/{args.few_shot_type}_pred_lower{args.do_lower}_{args.seed}_bs{args.train_batch_size}_lr{args.learning_rate}.txt', 'w')
    for i in range(len(all_labels)):
        f_gold_w.write(str(all_labels[i]))
        f_gold_w.write('\n')

        f_pred_w.write(str(all_preds[i]))
        f_pred_w.write('\n')

    return scores


# initialization
args = init_args()
os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.device}"
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.n_gpu > 0:
#     torch.cuda.manual_seed_all(args.seed)
#     cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
set_seed(args.seed)
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
# training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    tfm_model = MyT5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, prefix_lenth=15, prefix_dropout=0.1
    )
    model = T5FineTuner(args, tfm_model, tokenizer)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
    # )

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[LoggingCallback()],
        logger=False,
        checkpoint_callback=False
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the final model
    # model.model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    # f_w = cs.open(f'{args.output_dir}/choosed_data.txt', 'w', encoding='utf-8')
    # f_w.write(str(model.choosed_data))
    # print("Finish training and saving the model!")

if args.do_eval:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    if not args.do_train:
        # path = '../New_style_unified_framework/outputs/rest15'
        f = cs.open(f'{args.output_dir}/choosed_data.txt', 'r', encoding='utf-8').readlines()
        choosed_data = eval(f[0].strip())
        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(
            args.output_dir, prefix_lenth=15, prefix_dropout=0.1
        )
        model = T5FineTuner(args, tfm_model, tokenizer)
        model.choosed_data = choosed_data
    sents, _ = read_line_examples_from_file(f'{args.data_dir}/test.txt', args=args, silence=False)

    print()
    test_dataset = ABSADataset(tokenizer, data_dir=args.data_dir, data_type='test', max_len=args.max_seq_length, args=args, choosed_data=model.choosed_data)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    start_time = time.time()
    scores = evaluate(test_loader, model, sents)
    all_time = time.time() - start_time
    # write to file
    log_file_path = f"{args.output_dir}/results_view{args.view_num}_method_name{args.method_name}_few_shot{args.few_shot_type}_lower{args.do_lower}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    end_time = time.perf_counter()
    run_time = end_time - start_time

    exp_settings = f"Few_shot={args.few_shot_type};" \
                   f"seed={args.seed};" \
                   f"type=tuning;" \
                   f"Batch={args.train_batch_size};" \
                   f"lr={args.learning_rate};"
    exp_results = f"F1 = {scores['f1']:.4f}\nPrecision = {scores['precision']:.4f}\n" \
                  f"Recall = {scores['recall']:.4f}\nall_time={all_time:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)
