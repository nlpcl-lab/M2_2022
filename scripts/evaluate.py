import os
import re
import sys
import math
import torch
import wandb
import logging
import spacy
import transformers
import datasets
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np


class AugmentorEvaluator(object):
    def __init__(self):
        total_users = pd.read_json(os.path.join(self.data_dir, './total_user.json'))
        test_docs = pd.read_pickle('./data/test.pickle')
        user2keyword = pd.read_pickle('./data/user2keyword.pickle')
        user2keyword = {
            user_id: counter_dict2list(likes)
            for user_id, likes in zip(user2keyword.loc['likes'].index, user2keyword.loc['likes'])
        }
        clusters = {}
        for i in [2, 4, 6, 7]:
            clusters[f'docs{i}'] = pd.read_json(
                os.path.join(self.data_dir, 'ver3', f'{i}_cluster_ver3_docs_penguin.json')
            )
            clusters[f'users{i}'] = pd.read_json(
                os.path.join(self.data_dir, 'ver3', f'{i}_cluster_ver3_users_penguin.json')
            )

def main(hparams):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(hparams.seed)
    hparams.version = len(os.listdir(hparams.output_dir)) if os.path.isdir(hparams.output_dir) else 1
    params = vars(hparams)
    module = CredibilityAugmentor(**params)

    # callbacks
    early_stopping, ckpt, lr_monitor = module.get_callback_fn('val/loss', 50)
    callbacks_list = [ckpt, lr_monitor]
    if hparams.use_early_stopping:
        callbacks_list.append(early_stopping)

    logger = module.get_logger(**params)
    hparams.logger = logger

    # trainer
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=callbacks_list)
    trainer.fit(module)
    # if hparams.do_eval:
    #     trainer.validate(module)
    if hparams.do_test:
        trainer.test(module)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # initialization
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--accelerator', default='gpu')

    # model arguments
    parser.add_argument('--model_name_or_path', default='facebook/bart-base')
    parser.add_argument('--task_name', default='m2')
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)
    # parser.add_argument("--train_file", default='train.json', type=str)
    # parser.add_argument("--eval_file", default='eval.json', type=str)
    # parser.add_argument("--test_file", default='test.json', type=str)
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--use_logger", default="wandb", type=str, choices=["tensorboard", "wandb"])
    # parser.add_argument("--use_scheduler", default="linear", type=str, choices=["linear", "onecycle"])
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)

    # training arguments
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--max_src_len", default=50, type=int)
    parser.add_argument("--max_tgt_len", default=100, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--overfit_batches", default=0, type=float,
                        help="Not used, implemented in utils.py")
    parser.add_argument("--gradient_clip_val", default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, default=(0.9, 0.98), nargs='+')
    parser.add_argument('--num_display', type=int, default=3)
    parser.add_argument('--use_early_stopping', action='store_true')

    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()

    main(hparams)
