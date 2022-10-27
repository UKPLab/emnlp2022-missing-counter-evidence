import os
from typing import Dict, Tuple

import torch
from sklearn.metrics import classification_report
from torch import nn, LongTensor
from transformers import PreTrainedTokenizer

from experiment_code.util.file_util import write_jsonl_lines
from experiment_code.modeling.base.labels import LABEL_DICTS
from experiment_code.modeling.train import get_tokenizer_and_model, get_dataset


def load_for_prediction(config: Dict) -> Tuple[PreTrainedTokenizer, nn.Module]:
    """
    Load the trained model and tokenizer from an experiment file.
    :param config: Loaded experiment json file.
    """

    # Load model from results directory.
    checkpoint_dir: str = os.path.join("results", config['name'])

    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    print('Using Checkpoint', checkpoint)

    task_name: str = config['task-name']
    labels = LABEL_DICTS[task_name]

    tokenizer, model = get_tokenizer_and_model(config, labels, checkpoint)
    return tokenizer, model


def predict(config: Dict, eval_key: Dict):
    """
    Create a prediction file for the experiment.
    :param config: Loaded json file describing the experiment.
    :param eval_key: Evaluation key within the experiment file.
    """

    # Load trained model and tokenizer
    tokenizer, model = load_for_prediction(config)

    # Get label information to map numeric labels to strings
    task_name = config['task-name']
    labels = LABEL_DICTS[task_name]
    num_to_label = {
        labels[k]: k for k  in labels
    }

    # Select dataset for inference
    data = get_dataset(config, tokenizer, eval_key['data'], labels, eval_key['file'])

    # Run interence
    results = []
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sample in data:
            input_data = sample['text']
            for key in input_data:
                input_data[key] = LongTensor(input_data[key]).to(device).reshape((1, -1))
            pred = model(
                input_ids=input_data['input_ids'],
                token_type_ids=input_data['token_type_ids'],
                attention_mask=input_data['attention_mask']
            )

            logits = pred.logits.cpu().numpy()
            assert len(logits) == 1
            logits = logits[0]
            predicted_label = logits.argmax()

            current = {
                'id': sample['id'], 'claim': sample['claim'],
                'predicted_label': num_to_label[predicted_label], 'logits': list([float(val) for val in logits]),
                'gold_label': num_to_label[sample['label']]
            }

            if 'snippet' in sample:
                current['snippet'] = sample['snippet']
            results.append(current)

        gold_labels = [s['gold_label'] for s in results]
        pred_labels = [s['predicted_label'] for s in results]

        # Print evaluation
        print(classification_report(gold_labels, pred_labels, digits=3))

        # Store predictions
        dest_file = os.path.join('predictions', eval_key['name'] + '.jsonl')
        write_jsonl_lines(results, dest_file)

