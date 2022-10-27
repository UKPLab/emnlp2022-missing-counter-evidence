"""
Separately create samples with claims that contain *only* leaked evidence snippets and *only* unleaked evidence snippets.
Usage:
    create-leaked-unleaked-splits.py <config> <subset> <file>
"""
import collections
import os
from os.path import join
from typing import List, Dict, Tuple, Set, Iterable

import pandas as pd
from tqdm import tqdm
from docopt import docopt

from experiment_code.data_preparation.leaking_checker import RegexpHolder, FCLeakedClassifier, LeakedUrlHolder
from experiment_code.data_preparation.multifc import get_evidence_dict
from experiment_code.util.file_util import read_json, read_jsonl_lines, write_jsonl_lines


def create_leaked_unleaked_df(subset: str, file: str, multifc_path: str) -> pd.DataFrame:
    """
    Create a DataFrame containing all relevant fields for each claim / evidence snippet, and  additionally contain
    information for each evidence snippet whether it contains leaked evidence or not.

    :param subset: Subset of MultiFC ("snes", "pomt")
    :param file: File to consider ("test.jsonl")
    :param multifc_path: Path to extracted MultiFC directory
    """

    # Map each evidence file name to a list of snippets
    evidence_dict: Dict[str, List] = get_evidence_dict(os.path.join(multifc_path, 'snippets'), [subset])

    # Results for the DataFrame
    df_data: Dict[str, List] = collections.defaultdict(list)

    # Instantiate classifier to detect leaked evidence snippets
    regular_expressions: List = RegexpHolder.get()
    url_dict: Dict[str, Dict] = LeakedUrlHolder.get()
    fc_url_classifier: FCLeakedClassifier = FCLeakedClassifier(url_dict, regular_expressions)

    # Iterate over all samples and add them to the dataframe
    for sample in read_jsonl_lines(join('./data', join(subset, file))):
        claim_id: str = sample['id']
        snippets = evidence_dict[sample['id']]

        # Iterate over all snippets
        for snippet in snippets:
            snippet_id: int = int(snippet['id'])
            snippet_src: str = snippet['url']

            # Check if snipet is leaked
            leaked_url: bool = fc_url_classifier.is_fc_url(snippet_src)
            leaked_words: bool = fc_url_classifier.is_fc_statement(snippet)
            leaked: bool = leaked_url or leaked_words

            df_data['subset'].append(subset)
            df_data['split'].append(file)
            df_data['claim_id'].append(claim_id)
            df_data['snippet_id'].append(snippet_id)
            df_data['snippet_src'].append(snippet_src)
            df_data['leaked_url'].append(leaked_url)
            df_data['leaked_words'].append(leaked_words)
            df_data['leaked'].append(leaked)
            df_data['label'].append(sample['label'])

    return pd.DataFrame(df_data)


def create_leaked_unleaked_splits(subset: str, file: str, multifc_path: str) -> None:
    """
    Create two dats plits: one with claims and leaked evidence snippets. one with claims and unleaked evidence snippets.
    :param subset: Which subset of MultiFC to use ("pomt", "snes")
    :param file: Which file to use ("test.jsonl")
    :param multifc_path: Path to the extracted full MultiFC dataset
    """

    # Create a dataframe containing all relevant fields for each snippet, and whether it is leaked or not
    leaked_unleaked_df: pd.DataFrame = create_leaked_unleaked_df(subset, file, multifc_path)

    # Map each tuple of (claimID, snippetID) to a boolean value (leaked or not)
    id_to_leaked: Dict[Tuple[str, int], bool] = {
        (entry['claim_id'], entry['snippet_id']): entry['leaked']
        for entry in leaked_unleaked_df.to_dict('records')
    }

    # Load all required samples
    samples: Iterable[Dict] = read_jsonl_lines(join('./data', join(subset, file)))

    # Separately store leaked and unleaked samples
    leaked_samples: List[Dict] = []
    unleaked_samples: List[Dict] = []

    # Relevant keys to keep in the final samples
    copy_keys: List[str] = ['id', 'claim', 'label', 'source', 'date']

    for sample in tqdm(samples):
        # Keep required keys for the leaked claim
        sample_leaked = {k: sample[k] for k in copy_keys}
        sample_leaked['snippets'] = []

        # Keep required keys for the unleaked claim
        sample_unleaked = {k: sample[k] for k in copy_keys}
        sample_unleaked['snippets'] = []

        # Iterate over snippets and decide whether to add them to the leaked or unleaked sample
        for snippet in sample['snippets']:
            snippet_entry_id = (sample['id'], snippet['id'])
            if id_to_leaked[snippet_entry_id]:
                sample_leaked['snippets'].append(snippet)
            else:
                sample_unleaked['snippets'].append(snippet)

        # Keep each sample regardless of how many snippets it contains
        leaked_samples.append(sample_leaked)
        unleaked_samples.append(sample_unleaked)

    # Store
    write_jsonl_lines(leaked_samples, join('./data', join(subset, f'leaked-{file}')))
    write_jsonl_lines(unleaked_samples, join('./data', join(subset, f'unleaked-{file}')))


def main(args) -> None:
    config: Dict = read_json(args['<config>'])
    subset: str = args['<subset>']
    file: str = args['<file>']
    create_leaked_unleaked_splits(subset, file, config['path_multifc'])


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
