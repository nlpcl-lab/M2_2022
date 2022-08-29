import os
import re
import sys
import math
import numpy as np
import torch
import wandb
import logging
import spacy
import transformers
import datasets
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser
from einops import rearrange
from torch import einsum
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from utils import bleu


class AugmentModel(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, config):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
        )

        print(f'[INFO] {model_name_or_path} model loaded.')

    def forward(self, batch):
        outputs = self.model(**batch)
        loss = outputs[0]  # tensor(0.7937, device='cuda:0')
        logits = outputs[1]  # tensor([[ 0.1360, -0.0559]], device='cuda:0')

        return outputs, loss, logits

    def generate(self, source_id, source_mask, max_length):
        generated_ids = self.model.generate(
            input_ids=torch.tensor(source_id),
            attention_mask=torch.tensor(source_mask),
            max_length=max_length,
            num_beams=3,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        return generated_ids


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
        num_display,
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
        self.num_display = num_display
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir

        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AugmentModel(model_name_or_path, cache_dir, config)

        self.input_keys = ['input_ids', 'attention_mask', 'labels']
                          # 'decoder_input_ids', 'decoder_attention_mask', 'decoder_token_type_ids']
        self.metric = load_metric("bleu")
        # self.spacy_nlp = spacy.load('en_core_web_sm')

    def setup(self, stage):
        total_docs = pd.read_json(os.path.join(self.data_dir, './total_docs.json'))

        idx1, idx2, idx3 = 80000, 90000, 100000
        df_train = pd.DataFrame({'input': total_docs.loc[0, :idx1], 'output': total_docs.loc[1, :idx1]})
        df_validation = pd.DataFrame({'input': total_docs.loc[0, idx1:idx2], 'output': total_docs.loc[1, idx1:idx2]})
        df_test = pd.DataFrame({'input': total_docs.loc[0, idx2:idx3], 'output': total_docs.loc[1, idx2:idx3]})

        # 빈 셀 포함한 행 제거
        for df in [df_train, df_validation, df_test]:
            df['input'].replace('', np.nan, inplace=True)
            df['output'].replace('', np.nan, inplace=True)
            df.dropna(subset=['input', 'output'], inplace=True)
            print(f"after dropping empty rows: {len(df)}")

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
            load_from_cache_file=False, # default False
            desc="Running tokenizer on dataset line_by_line",
        )

        print(f'[INFO] {self.task_name} dataset loaded.')

    def tokenize_function(self, examples):
        # len(texts) = 1000
        inputs = examples['input']
        outputs = [f'{inp} {outp}' for inp, outp in zip(examples['input'], examples['output'])]
        padding = 'max_length'

        batch_encoding = self.tokenizer(
            inputs,
            padding=padding,  # @@@ or 'max_length'
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='np',
        )

        batch_encoding_output = self.tokenizer(
            outputs,
            padding=padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='np',
        )

        batch_encoding['labels'] = batch_encoding_output['input_ids']

        return batch_encoding

    def tokenize_text(self, context, text):
        padding = 'max_length'

        batch_encoding = self.tokenizer(
            context,
            padding=padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='np',
        )

        batch_encoding_output = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='np',
        )

        batch_encoding['labels'] = batch_encoding_output['input_ids']

        return batch_encoding

    def _common_step(self, batch, stage):
        inputs = {k: v for (k, v) in batch.items() if k in self.input_keys}
        outputs, loss, logits = self.model(inputs)

        self.log(f'{stage}/loss', loss, batch_size=self.batch_size)

        if stage == 'tr':
            return loss
        else:
            pred_ids = logits.argmax(dim=2)

            return {
                'loss': loss,
                "gold_ids": batch["labels"],
                "pred_ids": pred_ids,
            }

    def _common_epoch_end(self, outputs, stage):
        output = outputs[0]
        loss, gold_ids, pred_ids = output["loss"], output["gold_ids"], output["pred_ids"]
        pred_tokens = self.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        gold_tokens = self.tokenizer.batch_decode(
            gold_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # bleu
        scores = bleu(predictions=pred_tokens, references=gold_tokens)

        # logging
        for k in scores.keys():
            self.log(f'{stage}/{k}', scores[k])

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
            num_training_steps=self.get_estimated_stepping_batches(),
        )

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'step', # or 'epoch'
                    'frequency': 1
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
        df = pd.DataFrame(batch)
        input_dict = df.to_dict(orient='list')
        for key in self.input_keys:
            input_dict[key] = torch.tensor(input_dict[key])

        return input_dict

    def get_callback_fn(self, monitor='val/loss', patience=50):
        early_stopping_callback = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='min',
            verbose=True
        )
        ckpt_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            filename='epoch={epoch}-val_loss={val/loss:.2f}',
            monitor=monitor,
            save_last=True,
            save_top_k=1,
            mode='min',
            verbose=True,
            auto_insert_metric_name=False
        )
        lr_monitor = LearningRateMonitor(logging_interval=None)

        return early_stopping_callback, ckpt_callback, lr_monitor

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
    parser.add_argument('--output_dir', default='output/bart-base')
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--use_logger", default="wandb", type=str, choices=["tensorboard", "wandb"])
    # parser.add_argument("--use_scheduler", default="linear", type=str, choices=["linear", "onecycle"])
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)

    # training arguments
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--overfit_batches", default=0, type=float, help="Not used, implemented in utils.py")
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
