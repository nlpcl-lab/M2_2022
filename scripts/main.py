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

from argparse import ArgumentParser
from einops import rearrange
from torch import einsum
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from augment_testor import AugmentorTester
from utils import counter_dict2list


class AugmentModel(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, config):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
        )

    def forward(self, batch):
        outputs = self.model(**batch)
        loss = outputs[0]  # tensor(0.7937, device='cuda:0')
        logits = outputs[1]  # tensor([[ 0.1360, -0.0559]], device='cuda:0')

        return outputs, loss, logits


class CredibilityAugmentor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path,
        task_name,
        accumulate_grad_batches,
        max_steps,
        batch_size,
        max_epochs,
        devices,
        lr,
        eps,
        betas,
        warmup_steps,
        num_workers,
        max_seq_length,
        output_dir,
        data_dir,
        cache_dir,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.devices = devices
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir

        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AugmentModel(model_name_or_path, cache_dir, config)

    def setup(self, stage):
        total_docs = pd.read_json(os.path.join(self.data_dir, './total_docs.json'))
        # total_users = pd.read_json(os.path.join(self.data_dir, './total_user.json'))
        # test_docs = pd.read_pickle('./data/test.pickle')
        # user2keyword = pd.read_pickle('./data/user2keyword.pickle')
        # user2keyword = {
        #     user_id: counter_dict2list(likes)
        #     for user_id, likes in zip(user2keyword.loc['likes'].index, user2keyword.loc['likes'])
        # }
        # clusters = {}
        # for i in [2, 4, 6, 7]:
        #     clusters[f'docs{i}'] = pd.read_json(
        #         os.path.join(self.data_dir, 'ver3', f'{i}_cluster_ver3_docs_penguin.json')
        #     )
        #     clusters[f'users{i}'] = pd.read_json(
        #         os.path.join(self.data_dir, 'ver3', f'{i}_cluster_ver3_users_penguin.json')
        #     )
        df_train = pd.DataFrame({'input': total_docs.loc[0, :800], 'output': total_docs.loc[1, :800]})
        df_validation = pd.DataFrame({'input': total_docs.loc[0, 800:900], 'output': total_docs.loc[1, 800:900]})
        df_test = pd.DataFrame({'input': total_docs.loc[0, 900:1000], 'output': total_docs.loc[1, 900:1000]})

        datasets = DatasetDict({
            'train': Dataset.from_pandas(df_train),
            'validation': Dataset.from_pandas(df_validation),
            'test': Dataset.from_pandas(df_test),
        })
        self.datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[],
            num_proc=None,  # default None
            load_from_cache_file=True, # default False
            desc="Running tokenizer on dataset line_by_line",
        )

        print(f'[INFO] {self.task_name} dataset loaded.')

    def tokenize_function(self, examples):
        # len(texts) = 1000
        inputs = examples['input']
        outputs = examples['output']
        padding = False

        batch_encoding = self.tokenizer(
            inputs,
            padding=padding,  # @@@ or 'max_length'
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        batch_encoding_output = self.text_tokenizer(
            outputs,
            padding=padding,   # @@@ or 'max_length'
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )
        if padding == "max_length":
            batch_encoding_output["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in batch_encoding_output["input_ids"]
            ]

        batch_encoding['labels'] = batch_encoding_output['input_ids']

        # batch_encoding['decoder_input_ids'] = batch_encoding_output.pop('input_ids')
        # batch_encoding['decoder_token_type_ids'] = batch_encoding_output.pop('token_type_ids')
        # batch_encoding['decoder_attention_mask'] = batch_encoding_output.pop('attention_mask')

        return batch_encoding

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
    parser.add_argument('--use_early_stopping', action='store_true')

    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()

    main(hparams)