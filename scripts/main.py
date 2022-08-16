import os
import re
import sys
import math
import numpy as np
import torch
import wandb
import logging
import transformers
import datasets
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser, Namespace
from einops import rearrange
from torch import einsum
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

class CredibilityAugmentor(pl.LightningModule):
    def __init__(self):
        pass

def main(hparams):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(hparams.seed)
    hparams.version = len(os.listdir(hparams.output_dir)) if os.path.isdir(hparams.output_dir) else 1
    module = CredibilityAugmentor()

    # callbacks
    ckpt_callbacks = EarlyStopping(
        monitor='val/f1_weighted',
        patience=30,    # one check happens after every training epoch
        mode='max',
        verbose=True
    )
    early_stopping_callbacks = ModelCheckpoint(
        dirpath=hparams.output_dir,
        filename='{}-{}'.format(hparams.version, 'epoch={epoch}-val_loss={val/loss:.2f}-val_f1_weighted={val/f1_weighted:.2f}'),
        save_last=True,
        save_top_k=1,
        monitor='val/f1_weighted',  #         monitor='val/loss',
        mode='max',
        verbose=True,
        auto_insert_metric_name=False
    )

    # define logger
    if hparams.use_logger == "wandb":
        logger = WandbLogger(project=hparams.task_name)
        logger.watch(module)    # show gradients
        # log_model: Save checkpoints in wandb dir to upload on W&B servers.
    else:
        logger = TensorBoardLogger(f"{hparams.output_dir}/{hparams.task_name}_logs", name=hparams.task_name)

    hparams.logger = logger

    trainer = pl.Trainer.from_argparse_args(
        hparams,
        callbacks=[ckpt_callbacks, early_stopping_callbacks]
    )
    trainer.fit(module)
    # if hparams.do_eval:
    #     trainer.validate(module)
    if hparams.do_test:
        trainer.test(module)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # initialization
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--devices', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--accelerator', default='gpu')

    # model arguments
    parser.add_argument('--model_name_or_path', default='facebook/bart-base')
    parser.add_argument('--task_name', default='m2')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)
    parser.add_argument("--train_file", default='train.json', type=str)
    parser.add_argument("--eval_file", default='eval.json', type=str)
    parser.add_argument("--test_file", default='test.json', type=str)
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--use_logger", default="wandb", type=str, choices=["tensorboard", "wandb"])
    parser.add_argument("--use_scheduler", default="linear", type=str, choices=["linear", "onecycle"])
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)

    # training arguments
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--overfit_batches", default=0, type=float, help="Not used, implemented in utils.py")
    parser.add_argument("--gradient_clip_val", default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, default=(0.9, 0.98), nargs='+')

    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()

    main(hparams)