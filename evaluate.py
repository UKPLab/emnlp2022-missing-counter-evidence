"""
Contains evaluation methods to evaluate trained models and compare the performance on leaked and unleaked evidence
snippets.

Usage:
    evaluate.py evaluate <config> <key> [--exp=<exp>...] [--pred=<prediction_dir>]
    evaluate.py compare <config> <key> <experiments_evidence_only> <experiments_complete> [--pred=<prediction_dir>]
    evaluate.py leaked-unleaked <key_leaked> <key_unleaked> <dataset> [--exp=<exp>...] [--pred=<prediction_dir>]
"""
import collections
from os.path import join
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.metrics import classification_report

from experiment_code.modeling.base.labels import LABEL_DICTS
from experiment_code.modeling.leaked_unleaked_predictions import LeakedUnleakedPredictions
from experiment_code.modeling.model_evaluator import ModelEvaluator
from experiment_code.util.file_util import read_json


def get_metrics_from_base_name(
        basis_name: str,
        prediction_dir: str,
        key: str,
        leaked_unleaked_evaluator: ModelEvaluator) -> List[Dict]:
    """
    tldr; A shortcut to automatically get all experiment files with different seeds using naming convention.

    Get the evaluation metrics on all three subsets (all, leaked, unleaked) for all three models of the same experiment
    but different seeds. This assumes that two experiment files exist that share the same name, but are appended with
    a "-2" and "-3".
    Example:

    Given the experiment file "my-experiment.json" the following experiment files must also exist:
    - "my-experiment-2.json"
    - "my-experiment-3.json"

    :param basis_name: The experiment file without a seed appendix ("my-experiment.json")
    :param prediction_dir: Path to directory containing all predictions
    :param key: Key in the evaluation section of the experiment files that is subject to evaluation.
    :param leaked_unleaked_evaluator: Instantiation of the ModelEvaluator to differentiate between leaked and unleaked
    samples.
    """

    # Automatically infer the names of the remaining experiment files.
    experiment_files = [
        basis_name,
        basis_name.replace('.json', '-2.json'),
        basis_name.replace('.json', '-3.json')
    ]

    # Compute metrics
    metrics = [
        leaked_unleaked_evaluator.get_evaluation(name, prediction_dir, key)
        for name in experiment_files
    ]

    return metrics


def get_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Compute and round mean and standard deviation from a list of values interpreted as percentages.
    Values are multiplied by 100 and rounded after one decimal.
    :param values: Numeric values
    """
    avg_value: float = round(100 * np.mean(values), 1)
    std_value: float = round(100 * np.std(values), 1)
    return avg_value, std_value


def evaluate_compare(args: Dict) -> None:
    """
    Compute the comparison between evidence-only experiments and experiments using the entire sample.
    :param args: Arguments from command line.
    """

    # Extract parameters
    prediction_directory: str = args['--pred'] or './predictions'
    key: str = args['<key>']
    config: Dict = read_json(args['<config>'])
    leaked_csv_path: str = join(config['analysis_dir'], 'leaking-all-multifc.csv')

    # Only need base names. The remaining experiment files with different seeds will be collected based on
    # naming convention.
    base_experiment_evidence_only: str = args['<experiments_evidence_only>']
    base_experiment_complete: str = args['<experiments_complete>']
    metrics_key: str = 'f1-score'

    # Instantiate an evaluator with awareness of leaked samples.
    leaked_unleaked_evaluator: ModelEvaluator = ModelEvaluator(leaked_csv_path)

    # Compute all metrics based on all experiments (different seeds) of both experiment setups.
    metrics_evidence_only: List[Dict] = get_metrics_from_base_name(
        base_experiment_evidence_only, prediction_directory, key, leaked_unleaked_evaluator
    )
    metrics_complete: List[Dict] = get_metrics_from_base_name(
        base_experiment_complete, prediction_directory, key, leaked_unleaked_evaluator
    )

    # Only these two are supported.
    assert key in {'snes-all', 'pomt-all'}
    dataset: str = key.split('-')[0]        # assuming this is pomt-all / snes-all
    labels: List[str] = sorted(list(LABEL_DICTS[dataset].keys()), key=lambda x: -LABEL_DICTS[dataset][x])

    for label in labels:

        result_dict: Dict = {}
        for subset in ['leaked', 'unleaked']:

            # Model with only evidence
            scores_evidence_only: List[float] = [
                m[subset][label][metrics_key] for m in metrics_evidence_only
            ]
            score_evidence_only, std_evidence_only = get_mean_std(scores_evidence_only)

            # Model with claim and evidence
            scores_complete: List[float] = [
                m[subset][label][metrics_key] for m in metrics_complete
            ]
            score_complete, std_complete = get_mean_std(scores_complete)

            # Support should be identical
            support_evidence_only = metrics_evidence_only[0][subset][label]['support']
            support_complete = metrics_complete[0][subset][label]['support']
            assert support_evidence_only == support_complete

            result_dict[subset] = {
                "score_evidence_only": score_evidence_only,
                "std_evidence_only": std_evidence_only,
                "score_complete": score_complete,
                "std_complete": std_complete,
                "support": support_evidence_only,

                "all_scores_evidence_only": scores_evidence_only,
                "all_scores_complete": scores_complete
            }

        print(f'Label: {label}:')
        for subset in ['leaked', 'unleaked']:
            out_line = f'[{subset.upper()}]: '
            out_line += f'evidence-only: {result_dict[subset]["score_evidence_only"]} ({result_dict[subset]["std_evidence_only"]} std); '
            out_line += f'complete: {result_dict[subset]["score_complete"]} ({result_dict[subset]["std_complete"]} std); '
            out_line += f'support: {result_dict[subset]["support"]}'
            print(out_line)

        # Compute Diffs
        def get_diff(key: str) -> List[float]:
            """
            Compute the differences across different seeds
            """
            return [
                result_dict['leaked'][key][i] - result_dict['unleaked'][key][i]
                for i in range(len(result_dict['unleaked'][key]))
            ]

        def add_sign(val: float) -> str:
            """
            Add the "+" sign in case of a positive number
            """
            if val > 0:
                return f'+{val}'
            else:
                return str(val)

        # For evidence only
        all_diff_scores_evidence_only: List[float] = get_diff('all_scores_evidence_only')
        mean_diff_evidence_only, std_diff_evidence_only = get_mean_std(all_diff_scores_evidence_only)

        # For complete
        all_diff_scores_complete: List[float] = get_diff('all_scores_complete')
        mean_diff_complete, std_diff_complete = get_mean_std(all_diff_scores_complete)

        out_line = '[DIFFERENCE]: '
        out_line += f'evidence-only: {add_sign(mean_diff_evidence_only)} ({std_diff_evidence_only} std); '
        out_line += f'complete: {add_sign(mean_diff_complete)} ({std_diff_complete} std)'
        print(out_line)
        print('--------------\n')


def evaluate_model(args: Dict) -> None:
    """
    Evaluate numerous models of the same experiment on three subsets (all, leaked, unleaked).
    :param args: Arguments via command line
    """

    # Get arguments
    prediction_directory: str = args['--pred'] or './predictions'
    key: str = args['<key>']
    config: Dict = read_json(args['<config>'])
    leaked_csv_path: str = join(config['analysis_dir'], 'leaking-all-multifc.csv')

    # Multiple experiment files can be provided. Make sure we have at least one.
    experiment_files: List[str] = args['--exp']
    assert len(experiment_files) > 0, f'No experiment files provided!'

    # Instantiate evaluator with information about which claim has leaked or unleaked evidence.
    leaked_unleaked_evaluator: ModelEvaluator = ModelEvaluator(leaked_csv_path)

    # Compute performances for all experiments
    performances: List[Dict] = [
        leaked_unleaked_evaluator.get_evaluation(
            exp, prediction_directory, key
        ) for exp in experiment_files
    ]

    # Print performance for all three subsets
    for name in ['all', 'leaked', 'unleaked']:
        print(f'Evaluation for {name} samples')

        # Get performances for subset
        current_performances: List[Dict] = [p[name] for p in performances]

        # Store a list of metrics (per experiment) here.
        result = collections.defaultdict(dict)

        # Metrics to report
        all_metrics: List[str] = ['precision', 'recall', 'f1-score', 'support']

        # Extract each metric with each aggregation (micro/macro)
        for avg in ['macro avg', 'weighted avg']:
            for metric in all_metrics:
                result[metric][avg] = [p[avg][metric] for p in current_performances]

        # Print the results
        for metric in all_metrics:
            if metric == 'support':
                assert len(set(result[metric]["macro avg"])) == 1
                print(f'{metric}:: {result[metric]["macro avg"][0]}')
            else:
                micro: str = f'Micro: {round(100 * np.mean(result[metric]["weighted avg"]), 1)} ({round(100 * np.std(result[metric]["weighted avg"]), 1)} std)'
                macro: str = f'Macro: {round(100 * np.mean(result[metric]["macro avg"]), 1)} ({round(100 * np.std(result[metric]["macro avg"]), 1)} std)'
                print(f'{metric}:: {micro}; {macro}')

        print('---------------\n')


def evaluate_leaked_unleaked(args: Dict) -> None:
    """
    Evaluate the predictions of claims with *only* leaked evidence snippets and *only* unleaked evidence snippets.
    A sample is only considered if it has at least one leaked and at least one unleaked snippet.
    :param args: Command line arguments
    """

    # get command line arguments
    key_leaked: str = args['<key_leaked>']
    key_unleaked: str = args['<key_unleaked>']
    prediction_directory: str = args['--pred'] or './predictions'
    experiment_files: List[str] = args['--exp']
    dataset: str = args['<dataset>']
    assert len(experiment_files) > 0, f'No experiment files provided!'

    # Instantiate a "LeakedUnleakedPredictions" holding the prediction for both subsets for all different experiment
    # seeds. This will only return samples for which claims with leaked and unleaked evidence exist
    leaked_unleaked_pred: List[pd.DataFrame] = [
        LeakedUnleakedPredictions(
            './data', prediction_directory, read_json(exp), key_leaked, key_unleaked, dataset
        ).get_predictions_for_claims_with_leaked_and_unleaked_evidence()
        for exp in experiment_files
    ]

    # Compute performance using only leaked snippets
    leaked_performance: List[Dict] = [
        classification_report(df_current['gold_label'], df_current['predicted_label_leaked'], output_dict=True, zero_division=0)
        for df_current in leaked_unleaked_pred
    ]

    # Compute performance using only unleaked snippets
    unleaked_performance: List[Dict] = [
        classification_report(df_current['gold_label'], df_current['predicted_label_unleaked'], output_dict=True, zero_division=0)
        for df_current in leaked_unleaked_pred
    ]

    # Print Metrics (leaked, unleaked) per label
    labels: List[str] = sorted(list(LABEL_DICTS[dataset].keys()), key=lambda x: -LABEL_DICTS[dataset][x])
    for label in labels:
        print(f'{label.upper():}')
        for subset, subset_name in [(leaked_performance, 'leaked'), (unleaked_performance, 'unleaked')]:
            out_line: str = f'[{subset_name.upper()}]: '
            for metric in ['precision', 'recall', 'f1-score']:
                mean_score, std_score = get_mean_std([sub[label][metric] for sub in subset])
                out_line += f'{metric}: {mean_score} ({std_score} std); '
            out_line += f'support: {subset[0][label]["support"]}'
            print(out_line)

        print('------\n')

    # Print general metrics
    print('\n--- Overall Metrics ---')
    for subset, subset_name in [(leaked_performance, 'leaked'), (unleaked_performance, 'unleaked')]:
        out_line = f'[{subset_name.upper()}]: '
        mean_acc, std_acc = get_mean_std([sub['accuracy'] for sub in subset])
        mean_f1_micro, std_f1_micro = get_mean_std([sub['weighted avg']['f1-score'] for sub in subset])
        mean_f1_macro, std_f1_macro = get_mean_std([sub['macro avg']['f1-score'] for sub in subset])
        support: int = subset[0]['macro avg']['support']

        out_line += f'Accuracy: {mean_acc} ({std_acc} std); '
        out_line += f'F1 Micro: {mean_f1_micro} ({std_f1_micro} std); '
        out_line += f'F1 Macro: {mean_f1_macro} ({std_f1_macro} std); '
        out_line += f'Support: {support}'

        print(out_line)


def main(args: Dict) -> None:
    if args['evaluate']:
        evaluate_model(args)
    elif args['compare']:
        evaluate_compare(args)
    elif args['leaked-unleaked']:
        evaluate_leaked_unleaked(args)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
