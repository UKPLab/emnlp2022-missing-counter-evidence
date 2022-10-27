"""
Eval a basic model -  without the tweaks by Hansen et al. Just concatenate Claim and evidence, and truncate.
Usage:
    predict.py predict <config> <eval-key>
    predict.py predict-all <config>
"""
import os
from typing import Dict


from docopt import docopt

from experiment_code.util.file_util import read_json
from experiment_code.modeling.predict import predict


def eval_basic(config_name: str, eval_key: str) -> None:
    config: Dict = read_json(os.path.join('experiments', config_name))
    eval_item = config['evaluation'][eval_key]
    predict(config, eval_item)


def eval_all(config_name: str) -> None:
    config: Dict = read_json(os.path.join('experiments', config_name))
    for eval_key in config['evaluation']:
        predict(config, config['evaluation'][eval_key])


def main(args) -> None:

    if args['predict']:
        eval_basic(args['<config>'], args['<eval-key>'])
    elif args['predict-all']:
        eval_all(args['<config>'])


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
