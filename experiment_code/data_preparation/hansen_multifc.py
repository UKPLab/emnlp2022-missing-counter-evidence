import os
from typing import List, Tuple, Iterable, Dict, Set
import pickle

from tqdm import tqdm

"""
Code and data based on Hansen et. al: https://github.com/casperhansen/fake-news-reasoning 

This code is used to convert the datasplits provided by Hansen et al. into jsonl files to run these experiments.
"""

import pandas as pd


def read_snippets(file_path) -> List[Dict]:
    """
    Adapted from Hansen et al.
    """

    snippets = []

    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line_i, line in enumerate(lines):

            if line_i > 9:
                # only happens once due to formatting in MultiFC.
                continue

            content = line.split("\t")
            title = content[1] if len(content[1]) > 2 else None
            text = content[2] if len(content[2]) > 2 else None
            snippets.append({
                'title': title, 'text': text, 'id': int(content[0])
            })

    return snippets


def get_split_ids(file_path: str) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
    content = pickle.load(open(file_path, "rb"))
    assert len(content) == 3
    return tuple(content)


def verify_labels(samples: List[Dict], labels: Set[str]):
    assert set([sample['label'] for sample in samples]) == labels, f'Mismatch in allowed labels'


class MultiFCData:
    def __init__(self, source: str):
        assert os.path.exists(source) and os.path.isdir(source)
        self.directory = source

    def export(self, dataset: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:

        print(f'Create dataset for {dataset} ...')
        directory: str = self._get_dataset_dir(dataset)

        # Load claims
        claim_path: str = os.path.join(directory, f'{dataset}.tsv')
        claim_df: pd.DataFrame = pd.read_csv(claim_path, sep='\t', header=None, usecols=[0, 1, 2, 3, 10], names=[
            'id', 'claim', 'label', 'source', 'date'
        ])

        # Load accepted labels; Just to verify
        labels: Set[str] = set(pickle.load(open(os.path.join(directory, f'{dataset}_labels.pkl'), "rb")))

        # Get Load Split IDs
        ids_train, ids_dev, ids_test = get_split_ids(os.path.join(directory, f'{dataset}_index_split.pkl'))

        print('Create split (train) ...')
        data_train = self._create_data(claim_df.iloc[ids_train, :])

        print('Create split (dev) ...')
        data_dev = self._create_data(claim_df.iloc[ids_dev, :])

        print('Create split (test) ...')
        data_test = self._create_data(claim_df.iloc[ids_test, :])

        # Verify Labels
        verify_labels(data_train, labels)
        verify_labels(data_dev, labels)
        verify_labels(data_test, labels)

        # Verify no overlap
        all_ids = set(
            [sample['id'] for data in [data_train, data_dev, data_test] for sample in data]
        )
        assert len(all_ids) == sum([len(data) for data in [data_train, data_dev, data_test]]), f'Overlap with samples!'

        return data_train, data_dev, data_test

    def _create_data(self, sample_df: pd.DataFrame) -> List[Dict]:
        samples = sample_df.to_dict('records')
        snippet_directory: str = os.path.join(self.directory, 'snippets')
        for sample in tqdm(samples):
            snippet_path: str = os.path.join(snippet_directory, sample['id'])
            snippets = read_snippets(snippet_path)
            sample['snippets'] = snippets

        return samples

    def _get_dataset_dir(self, dataset: str) -> str:
        dataset_dir: str = os.path.join(self.directory, dataset)
        assert os.path.exists(dataset_dir)
        return dataset_dir


