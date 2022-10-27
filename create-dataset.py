"""
Recreate the dataset using the splits from Hansen et al. Must provide path to the downloaded data.
Usage:
    create-dataset.py <source> <dest>
"""
import os
from typing import List, Dict

from docopt import docopt

from experiment_code.data_preparation.hansen_multifc import MultiFCData
from experiment_code.util.file_util import write_jsonl_lines, unzip


def store_data(directory: str, name: str, samples: List[Dict]):
    """
    Store samples in the given place.
    :param directory: Destination directory
    :param name: Destination file name
    :param samples: Samples to store.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    write_jsonl_lines(samples, os.path.join(directory, name))


def recreate_dataset(source_zip: str, dest_dir: str) -> None:
    """
    Re-create the dataset for pomt and snes using the data splits from Hansen et al.
    :param source_zip: Path to downloaded zip file
    :param dest_dir Destination directory for prepared data splits
    """

    # Verify source files exist
    assert os.path.exists(source_zip), f'Make sure the multi_fc_publicdata.zip file is in the specified location: {source_zip}!'
    source_directory: str = os.path.dirname(source_zip)
    print('source_directory', source_directory)
    assert source_zip.endswith('.zip'), 'No zip file provided.'

    # Unzip downloaded file
    unzipped_directory_path: str = source_zip[:-len('.zip')]
    print(f'Unzipping ({source_zip}) ...')
    unzip(source_zip, source_directory)
    print(f'Done: {unzipped_directory_path}')

    # Load unzipped data
    multifc_data = MultiFCData(unzipped_directory_path)

    # Parse and export Snopes data
    snes_train, snes_dev, snes_test = multifc_data.export('snes')
    print('Loaded Snopes:', len(snes_train), len(snes_dev), len(snes_test))
    snes_dir = os.path.join(dest_dir, 'snes')
    store_data(snes_dir, 'train.jsonl', snes_train)
    store_data(snes_dir, 'dev.jsonl', snes_dev)
    store_data(snes_dir, 'test.jsonl', snes_test)

    # Parse and export Politifact data
    pomt_train, pomt_dev, pomt_test = multifc_data.export('pomt')
    print('Loaded PolitiFact:', len(pomt_train), len(pomt_dev), len(pomt_test))
    pomt_dir = os.path.join(dest_dir, 'pomt')
    store_data(pomt_dir, 'train.jsonl', pomt_train)
    store_data(pomt_dir, 'dev.jsonl', pomt_dev)
    store_data(pomt_dir, 'test.jsonl', pomt_test)

    print('Done.')


def main(args) -> None:
    recreate_dataset(args['<source>'], args['<dest>'])


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
