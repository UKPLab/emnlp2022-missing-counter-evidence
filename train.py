"""
Train a model based on the concatenation of claim and evidence snippets.
Usage:
    train.py train <experiment_file>
"""
import os
from typing import Dict

from docopt import docopt

from experiment_code.util.file_util import read_json
from experiment_code.modeling.train import train


def train_basic(experiment_file: str) -> None:
    """
    Train the model specified in the experiment configuration file.
    :param experiment_file: Path to the experiment configuration file.
    """

    config: Dict = read_json(os.path.join('experiments', experiment_file))
    train(config)


def main(args) -> None:

    if args['train']:
        train_basic(args['<experiment_file>'])


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
