import os
import pickle
import re
import sys
import math
import torch
import pandas as pd
import numpy as np

from argparse import ArgumentParser

from augment_testor import AugmentorTester
from main import CredibilityAugmentor


class AugmentorEvaluator(object):
    def __init__(self, hparams, module=None):
        self.hparams = hparams
        self.test_docs = pd.read_pickle(hparams.test_fname)
        self.testors = self._load_testors(hparams.user2keywords)
        if module:
            self.module = module
        else:
            self.module = CredibilityAugmentor.load_from_checkpoint(hparams.model_name_or_path)

    def _load_testors(self, user2keywords):
        self.clusters = []
        testors = []

        for idx, output_file in enumerate(user2keywords):
            cluster = self._load_clusters(output_file)
            self.clusters.append(cluster)
            testors.append(AugmentorTester(cluster, clusteridx=idx, version=self.hparams.trial))

        return testors

    def _load_clusters(self, output_file):
        f = open(output_file, 'rb')
        user2keyword = pickle.load(f)
        return user2keyword

    def evaluate_sents(self, original_text, augmented_text):
        mu1_o, sig1_o, mu1_a, sig1_a = self.testors[0].evaluate(original_text, augmented_text)
        mu2_o, sig2_o, mu2_a, sig2_a = self.testors[1].evaluate(original_text, augmented_text)
        correct = (abs(mu1_o - mu2_o) > abs(mu1_a - mu2_a)) or (abs(sig1_o - sig2_o) > abs(sig1_a - sig2_a))
        if abs(mu1_a - mu2_a) == 0.0 and abs(sig1_a - sig2_a) == 0.0:
            value2 = 0
        elif abs(mu1_a - mu2_a) == 0.0:
            value2 = ((abs(sig1_o - sig2_o) - abs(sig1_a - sig2_a)) / abs(sig1_a - sig2_a))
        elif abs(sig1_a - sig2_a) == 0.0:
            value2 = ((abs(mu1_o - mu2_o) - abs(mu1_a - mu2_a)) / abs(mu1_a - mu2_a))
        else:
            value2 = (((abs(mu1_o - mu2_o) - abs(mu1_a - mu2_a)) / abs(mu1_a - mu2_a)) + ((abs(sig1_o - sig2_o) - abs(sig1_a - sig2_a)) / abs(sig1_a - sig2_a))) / 2

        return correct, value2

    def augment_sents(self, ctext):
        batch_encoding = self.module.tokenizer(
            ctext,
            padding='max_length',
            truncation=True,
            max_length=self.hparams.max_seq_length,
            return_tensors='np',
        )
        source_id, source_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
        self.module.model.eval()
        with torch.no_grad():
            generated_ids = self.module.model.generate(source_id, source_mask, max_length=self.hparams.max_seq_length)
            prediction = self.module.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return prediction

    def evaluate(self):
        original_texts = self.test_docs.loc[0].values.tolist()
        augmented_texts = self.augment_sents(original_texts)
        total_c, total_v = [], []
        for original_text, augmented_text in zip(original_texts, augmented_texts):
            correct, value2 = self.evaluate_sents(original_text, augmented_text)
            total_c.append(correct)
            total_v.append(value2)

        percentage = sum(total_c) / len(total_c)
        tvalue = sum(total_v) / len(total_v)
        print(f'신뢰도 증강 성공률: {percentage:.2f}')
        print(f'신뢰도 증강율: {tvalue:.2f}')

        return percentage, tvalue


def main(hparams):
    evaluator = AugmentorEvaluator(hparams)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # initialization
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument("--trial", default=None)

    # model arguments
    parser.add_argument('--model_name_or_path', default='output/last-v1.ckpt')
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)

    hparams = parser.parse_args()
    hparams.test_fname = './data/test.pickle'
    hparams.user2keywords = ['./data/user2keyword1.pickle', './data/user2keyword2.pickle']

    main(hparams)
