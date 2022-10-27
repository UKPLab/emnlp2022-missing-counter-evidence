import codecs
import collections
import os
import re
from typing import Iterable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


def get_evidence_dict(directory: str, prefixes: Optional[Iterable[str]] = None) -> Dict[str, List]:
    """
    Load a dictionary that maps each evidence file (name) to a list of parsed snippets.

    :param directory: Directory containing the evidence files with the snippets.
    :param prefixes: An optional list of prefixes. If selected, only snippets with these prefixes will be read.
    :return:
    """

    # To extract the dates
    regexp_date = re.compile(r'(^[A-Z][a-z]{2} \d\d?, \d{4})(.+)$')

    # Output
    ev_dict = {}

    # If selected, only consider snippets with the defined prefixes
    if prefixes is not None:
        files = [
            file for file in os.listdir(directory)
            if file.split('-')[0] in prefixes
        ]
    else:
        files = list(os.listdir(directory))

    # Load all selected files. Each file contains multiple snippets
    for file in tqdm(files):
        with codecs.open(os.path.join(directory, file), encoding='utf-8') as f_in:
            lines = [line.strip() for line in f_in.readlines()]

        ev_dict[file] = []
        keys = ['id', 'snippet_title', 'snippet_text', 'url']

        # Go over each snippet
        for line in lines:
            current_sample = {}
            parts = line.split('\t')
            assert len(parts) == len(keys)

            # Parse snippet
            for i, part in enumerate(parts):
                current_key = keys[i]

                if current_key == 'snippet_text':
                    m = re.match(regexp_date, part)
                    if m:
                        current_sample[current_key] = m.group(2)
                        current_sample['date'] = m.group(1)
                    else:
                        current_sample[current_key] = part
                        current_sample['date'] = None
                else:
                    current_sample[current_key] = part

            # Append snippet
            ev_dict[file].append(current_sample)
    return ev_dict


def load_multifc_claims(src: str) -> pd.DataFrame:
    """
    Load claims from MultiFC.
    :param src: Point to the .tsv file containing the claims.
    """
    data = collections.defaultdict(list)
    keys = ['claimID', 'claim', 'label', 'claimURL', 'reason', 'categories', 'speaker', 'checker', 'tags',
            'articleTitle', 'publishDate', 'claimDate', 'entities']

    # Load claim file
    with codecs.open(src, encoding='utf-8') as f_in:
        lines = f_in.read().split('\n')
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        for line in lines:
            sample = line.split('\t')
            assert len(sample) == len(keys)
            for i, key in enumerate(keys):
                data[key].append(sample[i].strip())

    json_data = []
    for i in range(len(data[keys[0]])):
        sample = {}
        for key in data.keys():
            sample[key] = data[key][i]
        json_data.append(sample)

    return pd.DataFrame(data)
