import os.path
from os.path import join
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import classification_report

from experiment_code.util.file_util import read_jsonl_lines, read_json


def get_eval(predictions, output_dict=True):
    gold = [pred['gold_label'] for pred in predictions]
    pred = [pred['predicted_label'] for pred in predictions]
    return classification_report(gold, pred, digits=3, zero_division=0, output_dict=output_dict)


class ModelEvaluator:
    """
    Evaluate a prediction file. The evaluation is produced on three subsets of the predictions:
    - The entire prediction file
    - Only samples that contain leaked evidence snippets
    - Only samples that contain unleaked evidence snippets
    """

    def __init__(self, leaked_file_path: str, leaked_criteria: str = 'leaked'):
        """
        :param leaked_file_path: Path to the csv file that defines which claim contains leaked evidence snippets.
        :param leaked_criteria: Describes the column in the csv file that is used to identify whether a claim contains
        leaked evidence snippets or not.
        """

        # Load the analysis results to know for each claim (ID) whether it contains leaked evidence or not.
        self.df_leaked_by_claim_id: pd.DataFrame = pd.read_csv(leaked_file_path).groupby(by='claim_id').agg({
            'leaked': lambda x: len([val for val in x if val]) > 0,
            'leaked_words': lambda x: len([val for val in x if val]) > 0,
            'leaked_url': lambda x: len([val for val in x if val]) > 0,
            'label': lambda x: list(x)[0]
        }).reset_index()
        self.leaked_criteria: str = leaked_criteria

    def get_evaluation(self, experiment_path: str, predictions_directory: str, key: str) -> Dict:
        """
        Produce a dictionary with all evaluations for a given experiment.
        :param experiment_path: Path to the json file describing the experiment.
        :param predictions_directory: Path to the directory containing all prediction files.
        :param key: Evaluation key used in the experiment file to determine which predictions should be evaluated.
        """

        # Identify leaked IDs to differentiate between the leaked and unleaked subset
        leaked_ids: List[str] = list(
            self.df_leaked_by_claim_id[self.df_leaked_by_claim_id[self.leaked_criteria]]['claim_id']
        )

        # Load predictions
        assert os.path.exists(experiment_path), f'Predictions do not exist: {experiment_path}!'
        experiment_file: Dict = read_json(experiment_path)
        predictions = list(
            read_jsonl_lines(join(predictions_directory, experiment_file['evaluation'][key]['name'] + '.jsonl'))
        )

        return {
            'all': get_eval(predictions, output_dict=True),
            'leaked': get_eval([pred for pred in predictions if pred['id'] in leaked_ids], output_dict=True),
            'unleaked': get_eval([pred for pred in predictions if pred['id'] not in leaked_ids], output_dict=True)
        }


