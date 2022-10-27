import os
from typing import Dict, Tuple

from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments, IntervalStrategy, PreTrainedTokenizer, AutoTokenizer, \
    AutoModelForSequenceClassification, Trainer

from experiment_code.modeling.base.multifc_collator import DataCollatorMultiFC
from experiment_code.modeling.base.metrics import compute_metrics
from experiment_code.modeling.base.multifc_reader import MultiFcDataset
from experiment_code.modeling.base.multifc_snippet_reader import MultiFcDatasetBySnippet
from experiment_code.modeling.base.labels import LABEL_DICTS


def get_training_args(config: Dict) -> TrainingArguments:
    """
    Create training arguments.
    :param config: Loaded json configuration
    """

    # Select metric for model selection
    if 'metric_for_best_model' in config:
        metric_for_best_model = config['metric_for_best_model']
    else:
        metric_for_best_model = 'eval_f1'

    print('Using', metric_for_best_model)

    # Create training arguments
    return TrainingArguments(
        output_dir=os.path.join("results", config['name']),
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=config['lr'],
        per_device_train_batch_size=config['batch-size-train'],
        per_device_eval_batch_size=config['batch-size-eval'],
        num_train_epochs=config['epochs'],
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        seed=config['seed']
    )


def get_tokenizer_and_model(
        config: Dict,
        labels: Dict[str, int],
        model_name: str) -> Tuple[PreTrainedTokenizer, nn.Module]:
    """
    Get the tokenizer and model
    :param config: Loaded experiment configuration.
    :param labels: Dictionary mapping string labels to numeric values.
    :param model_name: Model name to load (e.g. "bert-base-uncased")
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    return tokenizer, model


def get_dataset(
        config: Dict,
        tokenizer: PreTrainedTokenizer,
        task_name: str,
        labels: Dict[str, int],
        file: str) -> Dataset:
    """
    Get an instantiation of the dataset based on the specified settings.
    :param config: Loaded experiment json file.
    :param tokenizer:  instantiated tokenizer
    :param task_name: Name of the task (e.g. "snes", "pomt")
    :param labels: Dictionary to get numeric values for each label
    :param file: jsonl file containing the samples.
    """

    # Extract criteria from the experiment configuration
    task_dir: str = os.path.join(config['data-dir'], task_name)
    lower: bool = config['lowercase']
    task_variant: str = config['task-variant']
    evidence_variant: str = config['evidence-variant']

    # Separate between different concatenation strategies of the claim with snippets & create dataset.
    if config['method'] == 'concat':
        data = MultiFcDataset(
            os.path.join(task_dir, file), tokenizer, task_variant, evidence_variant, labels, lower=lower
        )
    elif config['method'] == 'split-by-snippet':
        data = MultiFcDatasetBySnippet(
            os.path.join(task_dir, file), tokenizer, task_variant, evidence_variant, labels, lower=lower
        )
    else:
        raise ValueError(f'Unknown method: {config["method"]}')

    return data


def train(config: Dict):
    """
    Train a model.
    :param config: Loaded experiment file.
    """

    # Load defined experiment settings for training.
    training_args: TrainingArguments = get_training_args(config)

    model_name: str = config['model-name']
    task_name: str = config['task-name']
    labels = LABEL_DICTS[task_name]

    # Create model and tokenizer
    tokenizer, model = get_tokenizer_and_model(config, labels, model_name)

    # Load data
    data_train = get_dataset(config, tokenizer, task_name, labels, 'train.jsonl')
    data_dev = get_dataset(config, tokenizer, task_name, labels, 'dev.jsonl')

    # Instantiate trainer & train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_dev,
        tokenizer=tokenizer,
        data_collator=DataCollatorMultiFC(tokenizer=tokenizer, max_length=config['max-input-len']),
        compute_metrics=compute_metrics
    )

    trainer.train()
