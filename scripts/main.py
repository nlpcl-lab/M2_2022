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
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser, Namespace
from einops import rearrange
from torch import einsum
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class AugmentModel(nn.Module):
    def __init__(self):
        pass

class CredibilityAugmentor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path,
        task_name,
        accumulate_grad_batches,
        max_steps,
        num_training_cases,
        batch_size,
        max_epochs,
        devices,
        lr,
        eps,
        betas,
        warmup_steps,
        num_workers,
        max_seq_length,
        num_display,
        output_dir,
        data_dir,
        cache_dir,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_steps = max_steps
        self.num_training_cases = num_training_cases
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.devices = devices
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.num_display = num_display
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        pass

    def setup(self, stage):
        total_docs = pd.read_json(os.path.join(self.data_dir, './total_docs.json'))
        total_users = pd.read_json(os.path.join(self.data_dir, './total_user.json'))

        self.datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[],
            num_proc=None,  # default None
            load_from_cache_file=True, # default False
            desc="Running tokenizer on dataset line_by_line",
        )

        print(f'[INFO] {self.dataset_type} dataset loaded.')

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, 'val')

    def training_epoch_end(self, outputs):
        if not self.trainer.overfit_batches > 0.0:
            return
        return self._common_epoch_end(outputs, 'tr')

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, betas=self.betas, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.get_estimated_stepping_batches,
        )

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'frequency': self.trainer.check_val_every_n_epoch
                },
            }

    def get_estimated_stepping_batches(self):
        # check that max_steps is not None and is greater than 0
        if self.max_steps and self.max_steps > 0:
            # pytorch_lightning steps the scheduler every batch but only updates
            # the global_step every gradient accumulation cycle. Therefore, the
            # scheduler needs to have `accumulate_grad_batches` * `max_steps` in
            # order to reach `max_steps`.
            # See: https://github.com/PyTorchLightning/pytorch-lightning/blob/f293c9b5f4b4f9fabb2eec0c369f08a66c57ef14/pytorch_lightning/trainer/training_loop.py#L624
            t_total = self.max_steps * self.accumulate_grad_batches
        else:
            t_total = int(
                (
                    len(self.datasets['train'])
                    // (self.batch_size * max(1, len(self.devices)))
                )
                * self.max_epochs
                // self.accumulate_grad_batches
            )

        return t_total

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.datasets['validation'], self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def _collate_fn(self, batch):
        pass

    def get_callback_fn(self, monitor='val/loss', patience=50):
        early_stopping_callback = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='min',
            verbose=True
        )
        ckpt_callback = ModelCheckpoint(
            filename='epoch={epoch}-val_loss={val/loss:.2f}',
            monitor=monitor,
            save_last=True,
            save_top_k=1,
            mode='min',
            verbose=True,
            auto_insert_metric_name=False
        )
        return early_stopping_callback, ckpt_callback

    def get_logger(self, use_logger='wandb', task_name='m2_2022', **kwargs):
        if use_logger == 'tensorboard':
            logger = TensorBoardLogger("m2_2022_logs", name=task_name)
        elif use_logger == 'wandb':
            logger = WandbLogger(project=task_name)
        else:
            raise NotImplementedError
        return logger


def main(hparams):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(hparams.seed)
    hparams.version = len(os.listdir(hparams.output_dir)) if os.path.isdir(hparams.output_dir) else 1
    params = vars(hparams)
    module = CredibilityAugmentor(**params)

    # callbacks
    early_stopping, ckpt = module.get_callback_fn('val/loss', 50)
    callbacks_list = [ckpt]
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
    parser.add_argument('--devices', nargs='+', type=int, default=[0, 1])
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
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--overfit_batches", default=0, type=float, help="Not used, implemented in utils.py")
    parser.add_argument("--gradient_clip_val", default=0.0, type=float, help="Gradient clipping value")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, default=(0.9, 0.98), nargs='+')

    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()

    main(hparams)