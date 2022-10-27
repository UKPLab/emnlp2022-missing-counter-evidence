import collections
from os.path import join
from typing import Dict, List, Set

import pandas as pd

from experiment_code.util.file_util import read_jsonl_lines


def create_df_leaked_unleaked(leaked_predictions: List[Dict], unleaked_predictions: List[Dict]) -> pd.DataFrame:
    """
    Create a DataFrame to compare gold labels and prtedictions for claims with leaked and unleaked snippets.
    :param leaked_predictions: All predictions based on (only) leaked evidence snippets.
    :param unleaked_predictions: All predictions based on (only) unleaked evidence snippets.
    """

    # Identify all IDs that have leaked AND unleaked evidence snippets.
    leaked_ids: Set[str] = {pred['id'] for pred in leaked_predictions}
    unleaked_ids: Set[str] = {pred['id'] for pred in unleaked_predictions}
    leaked_and_unleaked_ids: Set[str] = leaked_ids & unleaked_ids

    # Resulting data for DataFrame
    df_data: Dict[str, List] = collections.defaultdict(list)

    # Build dataframe
    for split, predictions in [('Leaked', leaked_predictions), ('Unleaked', unleaked_predictions)]:
        for pred in predictions:

            # Category whether the claim has leaked / unleaked / both evidence snippets
            if pred['id'] in leaked_and_unleaked_ids:
                category = 'both'
            elif pred['id'] in leaked_ids:
                category = 'only-leaked'
            elif pred['id'] in unleaked_ids:
                category = 'only-unleaked'
            else:
                raise ValueError('Must not happen!')

            # Copy relevant fields
            df_data['id'].append(pred['id'])
            df_data['Claim'].append(pred['claim'])
            df_data['gold_label'].append(pred['gold_label'])
            df_data['predicted_label'].append(pred['predicted_label'])
            df_data['category'].append(category)
            df_data['Split'].append(split)

    return pd.DataFrame(df_data)


class LeakedUnleakedPredictions:
    """
    This class holds data and predictions for samples that contain *only* leaked evidence snippets and *only* unleaked
    evidence snippets.
    """

    def __init__(
            self,
            data_dir: str,
            prediction_dir: str,
            experiment: Dict,
            key_leaked: str,
            key_unleaked: str,
            dataset: str) -> None:
        """
        :param data_dir: Directory pointing to the extracted MultiFC data.
        :param prediction_dir: Directory of the prediction files.
        :param experiment: Loaded json file describing the experiment.
        :param key_leaked: Evaluation key pointing to the leaked subset (in the experiment config)
        :param key_unleaked: Evaluation key pointing to the unleaked subset (in the experiment config)
        :param dataset: Dataset name (e.g. "pomt", "snes")
        """

        # Load predictions
        self.predictions_leaked: List[Dict] = list(
            read_jsonl_lines(join(prediction_dir, experiment['evaluation'][key_leaked]['name'] + '.jsonl'))
        )
        self.predictions_unleaked: List[Dict] = list(
            read_jsonl_lines(join(prediction_dir, experiment['evaluation'][key_unleaked]['name'] + '.jsonl'))
        )

        # Load raw data
        self.data_leaked: List[Dict] = list(
            read_jsonl_lines(join(data_dir, join(dataset, experiment['evaluation'][key_leaked]['file'])))
        )

        self.data_unleaked: List[Dict] = list(
            read_jsonl_lines(join(data_dir, join(dataset, experiment['evaluation'][key_unleaked]['file'])))
        )

    def get_predictions_with(self, subset: str, min_snippets: int = 0, max_snippets: int = 100) -> List[Dict]:
        """
        Extract a subset of the predictions based on provided filter criteria.

        :param subset: Defines if predictions from the "leaked" or "unleaked" data are extracted.
        :param min_snippets: Only samples are considered if they contain at least that many leaked/unleaked
        (as specified) snippets.
        :param max_snippets: Only samples are considered if they contain at most that many leaked/unleaked
        (as specified) snippets.
        """

        # Verify correct subset
        assert subset in {'leaked', 'unleaked'}

        # Identify predictions and samples given the subset
        predictions: List[Dict] = self.predictions_leaked if subset == 'leaked' else self.predictions_unleaked
        samples: List[Dict] = self.data_leaked if subset == 'leaked' else self.data_unleaked

        # Filter based on snippet criteria.
        return [
            predictions[i] for i, sample in enumerate(samples)
            if min_snippets <= len(sample['snippets']) <= max_snippets
        ]

    def get_predictions_for_claims_with_leaked_and_unleaked_evidence(self):
        """
        Get a subset of all claims for which at least one unleaked and at least one leaked snippet exist.
        """

        # Select predictions with at least one snippet (leaked & unleaked)
        leaked_predictions: List[Dict] = self.get_predictions_with(subset='leaked', min_snippets=1)
        unleaked_predictions: List[Dict] = self.get_predictions_with(subset='unleaked', min_snippets=1)

        # Create dataframe containing the respecitve predictions and gold labels
        df_leaked_unleaked: pd.DataFrame = create_df_leaked_unleaked(
            leaked_predictions=leaked_predictions, unleaked_predictions=unleaked_predictions
        )

        # Only keep predictions that occur in both splits (have leaked and unleaked evidence splits)
        df_both = df_leaked_unleaked[df_leaked_unleaked['category'] == 'both']
        df_both_leaked = df_both[df_both['Split'] == 'Leaked'].copy()
        df_both_unleaked = df_both[df_both['Split'] == 'Unleaked'].copy()
        assert len(df_both_leaked) == len(df_both_unleaked)

        df_both_merged = df_both_leaked.merge(
            df_both_unleaked.loc[:, ['id', 'predicted_label']], on='id', suffixes=('_leaked', '_unleaked')
        )
        return df_both_merged.loc[:, ['id', 'Claim', 'gold_label', 'predicted_label_leaked', 'predicted_label_unleaked']].copy()


