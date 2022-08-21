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