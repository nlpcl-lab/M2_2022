import os
import pickle
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

from argparse import ArgumentParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

from augment_testor import AugmentorTester


class AugmentorEvaluator(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.test_docs = pd.read_pickle(hparams.test_fname)
        self.testors = self._load_testors(hparams.user2keywords)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            hparams.model_name_or_path,
            from_tf=bool(".ckpt" in hparams.model_name_or_path),
            cache_dir=hparams.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', cache_dir=hparams.cache_dir)

    def _load_testors(self, user2keywords):
        self.clusters = []
        testors = []

        for output_file in user2keywords:
            cluster = self._load_clusters(output_file)
            self.clusters.append(cluster)
            testors.append(AugmentorTester(cluster))

        return testors

    def _load_clusters(self, output_file):
        f = open(output_file, 'rb')
        user2keyword = pickle.load(f)
        return user2keyword

    def evaluate_sents(self, original_text, augmented_text):
        mu1_o, sig1_o, mu1_a, sig1_a = self.testors[0].evaluate(original_text, augmented_text)
        mu2_o, sig2_o, mu2_a, sig2_a = self.testors[1].evaluate(original_text, augmented_text)
        correct = (abs(mu1_o - mu2_o) > abs(mu1_a - mu2_a)) or (abs(sig1_o - sig2_o) > abs(sig1_a - sig2_a))

        return np.array([correct])

    def augment_sents(self, ctext):
        batch_encoding = self.tokenizer(
            ctext,
            padding='max_length',
            truncation=True,
            max_length=self.hparams.max_seq_length,
            return_tensors='np',
        )
        source_id, source_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=source_id,
                attention_mask=source_mask,
                max_length=self.hparams.max_length,
                truncation=True,
                num_beams=3,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            prediction = [self.tokenizer.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                          generated_ids]

            return prediction

    def evaluate(self):
        original_texts = self.test_docs.loc[0].values.tolist()
        augmented_texts = self.augment_sents(original_texts)
        total = np.array([])
        for original_text, augmented_text in zip(original_texts, augmented_texts):
            total = np.append(total, self.evaluate_sents(original_text, augmented_text))

        percentage = total.sum() / total.shape[0]
        print('증강률')

def main(hparams):
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
    parser.add_argument('--model_name_or_path', default='output/last-v1.ckpt')
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)

    hparams = parser.parse_args()
    hparams.test_fname = './data/test.pickle'
    hparams.user2keywords = ['./data/user2keyword1.pickle', './data/user2keyword2.pickle']

    main(hparams)
