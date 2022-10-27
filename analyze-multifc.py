"""
Automatically assess the degree of leaked evidence in the entire MultiFC dataset.

Usage:
    analyze-multifc.py identify <config>
    analyze-multifc.py stats <config> [--misinfo]
    analyze-multifc.py extract-leaked <config> --num=<num> [--skip=<skip>] [--seed=<seed>] [--misinfo]
    analyze-multifc.py extract-unleaked <config> --num=<num> [--skip=<skip>] [--seed=<seed>] [--misinfo]
"""
import collections
import os
from typing import Dict, List

import pandas as pd
from docopt import docopt
from tqdm import tqdm

from experiment_code.data_preparation.leaking_checker import RegexpHolder, FCLeakedClassifier, LeakedUrlHolder
from experiment_code.data_preparation.multifc import load_multifc_claims, get_evidence_dict
from experiment_code.util.file_util import read_json


# Labels that we consider as misinformation
MISINFORMATION_LABELS = [
    'false', 'mostly false', 'pants on fire!', 'fiction!', 'facebook scams', 'verdict: false', 'determination: false',
    'incorrect', 'misleading', 'fake news', 'mostly fiction!', 'fake', '4 pinnochios', 'determination: misleading',
    'miscaptioned', 'factscan score: false', 'misattributed', 'factscan score: misleading', 'determination: barely true',
    'incorrect attribution!', 'not the whole story', 'determination: huckster propaganda',
    'fiction! & satire!',
    'misleading!', 'a lot of baloney', 'distorts the facts', 'spins the facts', 'rating: false', 'needs context',
    'we rate this claim false', 'inaccurate attribution!', 'mostly_false', 'fiction', 'conclusion: false',
    'cherry picks', 'in-the-red',
    '0', '1', '2', '3'
]


def identify_leaked(config: Dict):
    """
    Identify which claims are associated with leaked evidence snippets. Results are stored in a .csv file.
    :param config: Json file pointing to the MultiFC dataset directory ("path_multifc"), and the destination directory
    for the analysis results ("analysis_dir").
    """

    # Just to keep track of claims without any evidence
    num_no_evidence_found: int = 0

    # Extract source directory.
    directory_multifc: str = config['path_multifc']

    # Not include test here (no labels exist anyhow)
    claims = pd.concat((
        load_multifc_claims(os.path.join(directory_multifc, 'train.tsv')),
        load_multifc_claims(os.path.join(directory_multifc, 'dev.tsv'))
    )).to_dict('records')

    # Mapping atext file name to the contained evidence snippets
    evidence_dict: Dict[str, List] = get_evidence_dict(os.path.join(directory_multifc, 'snippets'))

    # Instantiate classifier to detect leaked evidence based on URL or content phrases
    regular_expressions: List = RegexpHolder.get()
    url_dict: Dict[str, Dict] = LeakedUrlHolder.get()
    fc_url_classifier: FCLeakedClassifier = FCLeakedClassifier(url_dict, regular_expressions)

    # Resulting data for the DataFrame (and csv file)
    df_data: Dict[str, List] = collections.defaultdict(list)

    # Go over each claim in MultiFC
    for claim in tqdm(claims):

        claim_id: str = claim['claimID']

        # Ignore if no evidence for claim found
        if claim_id not in evidence_dict:
            num_no_evidence_found += 1
            continue

        # Get evidence and iterate over snippets
        evidence: List = evidence_dict[claim_id]
        for ev in evidence:
            snippet_id: int = int(ev['id'])
            snippet_src: str = ev['url']

            # Test if snippet is leaked
            leaked_url: bool = fc_url_classifier.is_fc_url(snippet_src)
            leaked_words: bool = fc_url_classifier.is_fc_statement(ev)
            leaked: bool = leaked_url or leaked_words

            # Add fields to CSV
            df_data['claim_id'].append(claim_id)
            df_data['snippet_id'].append(snippet_id)
            df_data['snippet_src'].append(snippet_src)
            df_data['leaked_url'].append(leaked_url)
            df_data['leaked_words'].append(leaked_words)
            df_data['leaked'].append(leaked)
            df_data['label'].append(claim['label'])
            df_data['misinformation'].append(claim['label'] in MISINFORMATION_LABELS)

    print('No evidence found for', num_no_evidence_found, 'claims.')

    # Write CSV
    analysis_dir: str = config['analysis_dir']
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    pd.DataFrame(df_data).to_csv(os.path.join(analysis_dir, 'leaking-all-multifc.csv'), sep=',', index=False)


def create_claim_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the leaked dataframe (describing which snippet is leaked) to a dataframe, describing whoich claim contains
    at least one leaked evidence snippet.
    :param df: The dataframe describing which snippet is leaked.
    """

    df_from = df.copy()

    # Create numeric values to enable simle aggregation
    df_from['leaked_url_num'] = 0
    df_from['leaked_words_num'] = 0
    df_from.loc[df_from['leaked_words'], 'leaked_words_num'] = 1
    df_from.loc[df_from['leaked_url'], 'leaked_url_num'] = 1

    # Group and sum up to find claims with at least one leaked evidence snippet
    df_claims = df_from.loc[:, ['claim_id', 'leaked_words_num', 'leaked_url_num']]\
        .groupby(by='claim_id')\
        .sum()\
        .reset_index()

    df_claims['leaked'] = df_claims['leaked_words_num'] + df_claims['leaked_url_num']
    return df_claims


def print_stats(config: Dict, misinfo: bool) -> None:
    df: pd.DataFrame = pd.read_csv(
        os.path.join(config['analysis_dir'], 'leaking-all-multifc.csv')
    )

    if misinfo:
        df = df[df['misinformation']]

    claim_df: pd.DataFrame = create_claim_df(df)
    num_claims: int = len(claim_df)

    for name, key in [
        ('# Leaked by URL', 'leaked_url_num'),
        ('# Leaked by Phrase', 'leaked_words_num'),
        ('# Leaked by Any', 'leaked'),
    ]:
        current_num = len(claim_df[claim_df[key] > 0])
        print(f"{name}: {current_num}, ({round(100 * current_num / num_claims, 1)}%)")


def extract_leaked(config: Dict, args: Dict) -> None:
    """
    Used to extract a subset of leaked samples
    :param config: Loaded config file pointing to the MultiFC directory and the analysis directory.
    :param args: arguments from the command line
    """

    # If true, only misinformation claims are considered
    misinfo: bool = args['--misinfo']

    # Get number of samples, seed, numer of samples skipped
    num_samples: int = int(args['--num'] or 100)
    seed: int = int(args['--seed'] or 123)
    skip: int = int(args['--skip'] or 0)

    # Base the selection on claims from train and dev data
    claims: pd.DataFrame = pd.concat((
        load_multifc_claims(os.path.join(config['path_multifc'], 'train.tsv')),
        load_multifc_claims(os.path.join(config['path_multifc'], 'dev.tsv'))
    ))

    # Mapping dictionaries
    id_to_claim: Dict[str, Dict] = {
        sample['claimID']: sample for sample in claims.to_dict('records')
    }
    ev_dict: Dict[str, List] = get_evidence_dict(os.path.join(config['path_multifc'], 'snippets'))

    # Load csv indicating which snippet is leaked
    leaked_df: pd.DataFrame = pd.read_csv(os.path.join(config['analysis_dir'], 'leaking-all-multifc.csv'))
    claim_leaked_df: pd.DataFrame = create_claim_df(leaked_df)

    # Now filter
    claim_leaked_df = claim_leaked_df.merge(leaked_df.loc[:, ['claim_id', 'misinformation']].drop_duplicates(), on='claim_id')
    claim_leaked_df = claim_leaked_df[(claim_leaked_df['leaked'] > 0)]
    if misinfo:
        claim_leaked_df = claim_leaked_df[(claim_leaked_df['misinformation'])]

    # Sample
    subset = claim_leaked_df.sample(frac=1.0, random_state=seed).head(skip + num_samples).tail(num_samples)
    subset = subset.merge(leaked_df[leaked_df['leaked']], on='claim_id')

    # Create resulting CSV
    df_data = collections.defaultdict(list)
    for sample in subset.to_dict('records'):
        claim_sample = id_to_claim[sample['claim_id']]
        evidence = {
            int(ev['id']): ev
            for ev in ev_dict[sample['claim_id']]
        }[sample['snippet_id']]

        df_data['claimID'].append(sample['claim_id'])
        df_data['claim'].append(claim_sample['claim'])
        df_data['label'].append(claim_sample['label'])
        df_data['claimURL'].append(claim_sample['claimURL'])
        df_data['claimDate'].append(claim_sample['claimDate'])
        df_data['leaked_url'].append(sample['leaked_url'])
        df_data['leaked_words'].append(sample['leaked_words'])
        df_data['snippet_title'].append(evidence['snippet_title'])
        df_data['snippet_text'].append(evidence['snippet_text'])
        df_data['snippet_date'].append(evidence['date'])
        df_data['snippet_url'].append(sample['snippet_src'])
        df_data['Applicable Claim'].append('')
        df_data['Leaking'].append('')
        df_data['Comment'].append('')

    # Write file
    file_name = f'leaked-samples_{num_samples}-num_{skip}-skip_{seed}-seed_{misinfo}-misinfo.csv'
    pd.DataFrame(df_data).to_csv(os.path.join(config['analysis_dir'], file_name), sep=',', index=False)


def extract_unleaked(config: Dict, args: Dict):
    """
    Used to extract a subset of unleaked samples
    :param config: Loaded config file pointing to the MultiFC directory and the analysis directory.
    :param args: arguments from the command line
    """

    # If true, only misinformation claims are considered
    misinfo: bool = args['--misinfo']

    # Get number of samples, seed, numer of samples skipped
    num_samples: int = int(args['--num'] or 100)
    seed: int = int(args['--seed'] or 123)
    skip: int = int(args['--skip'] or 0)

    # Base the selection on claims from train and dev data
    claims: pd.DataFrame = pd.concat((
        load_multifc_claims(os.path.join(config['path_multifc'], 'train.tsv')),
        load_multifc_claims(os.path.join(config['path_multifc'], 'dev.tsv'))
    ))

    # Mapping dictionaries
    id_to_claim: Dict[str, Dict] = {
        sample['claimID']: sample for sample in claims.to_dict('records')
    }
    ev_dict: Dict[str, List] = get_evidence_dict(os.path.join(config['path_multifc'], 'snippets'))

    # Load csv indicating which snippet is leaked
    leaked_df: pd.DataFrame = pd.read_csv(os.path.join(config['analysis_dir'], 'leaking-all-multifc.csv'))
    claim_unleaked_df: pd.DataFrame = create_claim_df(leaked_df)

    # Now filter
    claim_unleaked_df = claim_unleaked_df.merge(leaked_df.loc[:, ['claim_id', 'misinformation']].drop_duplicates(), on='claim_id')
    claim_unleaked_df = claim_unleaked_df[(claim_unleaked_df['leaked'] == 0)]
    if misinfo:
        claim_unleaked_df = claim_unleaked_df[(claim_unleaked_df['misinformation'])]

    # Sample
    subset = claim_unleaked_df.sample(frac=1.0, random_state=seed).head(skip + num_samples).tail(num_samples)
    subset = subset.merge(leaked_df[~leaked_df['leaked']], on='claim_id')

    # Create resulting CSV
    df_data = collections.defaultdict(list)
    for sample in subset.to_dict('records'):
        claim_sample = id_to_claim[sample['claim_id']]
        evidence = {
            int(ev['id']): ev
            for ev in ev_dict[sample['claim_id']]
        }[sample['snippet_id']]

        df_data['claimID'].append(sample['claim_id'])
        df_data['claim'].append(claim_sample['claim'])
        df_data['label'].append(claim_sample['label'])
        df_data['claimURL'].append(claim_sample['claimURL'])
        df_data['claimDate'].append(claim_sample['claimDate'])
        df_data['leaked_url'].append(sample['leaked_url'])
        df_data['leaked_words'].append(sample['leaked_words'])
        df_data['snippet_title'].append(evidence['snippet_title'])
        df_data['snippet_text'].append(evidence['snippet_text'])
        df_data['snippet_date'].append(evidence['date'])
        df_data['snippet_url'].append(sample['snippet_src'])
        df_data['Stance'].append('')
        df_data['Leaking'].append('')
        df_data['Comment'].append('')

    # Write file
    file_name = f'unleaked-samples_{num_samples}-num_{skip}-skip_{seed}-seed_{misinfo}-misinfo.csv'
    pd.DataFrame(df_data).to_csv(os.path.join(config['analysis_dir'], file_name), sep=',', index=False)


def main(args) -> None:
    config_path: str = args['<config>']
    config_file: Dict = read_json(config_path)

    if args['identify']:
        identify_leaked(config_file)
    elif args['stats']:
        print_stats(config_file, args['--misinfo'])
    elif args['extract-leaked']:
        extract_leaked(config_file, args)
    elif args['extract-unleaked']:
        extract_unleaked(config_file, args)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
