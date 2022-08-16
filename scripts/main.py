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

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # initialization
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--accelerator', default='gpu')

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