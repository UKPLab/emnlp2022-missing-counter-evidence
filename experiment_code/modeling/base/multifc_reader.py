from typing import Dict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from experiment_code.util.file_util import read_jsonl_lines


class MultiFcDataset(Dataset):
    """
    Dataset of MultiFC. Each claim is concatenated with all associated snippets. Together they form one sample.
    This dataset class is used for the experiments in the paper.
    """

    def __init__(self,
                 src: str,
                 tokenizer: PreTrainedTokenizer,
                 task_variant: str,
                 evidence_variant: str,
                 label_dict: Dict,
                 max_num_snippets=99,
                 lower: bool = False):
        """
        :param tokenizer
          The tokenizer used by the Pretrained model
        :param task_variant
          Specifies whether the model only sees the claim, only sees the evidence, or sees both.
        :param evidence_variant
          Specifies whether evidence is represented by the title, the text, or both.
        :param label_dict
          To convert string labels into numerical values
        :param max_num_snippets
          Number of evidence snippets to be used (if not excluded based on task variant)
        """

        # Verify parameters
        assert task_variant in {'only-claim', 'only-evidence', 'complete'}
        assert evidence_variant in {'title', 'snippet', 'title-snippet', 'n/a'}
        if evidence_variant == 'n/a':
            assert task_variant == 'only-claim'

        self.max_num_snippets = max_num_snippets
        self.label_dict = label_dict
        self.evidence_variant = evidence_variant
        self.task_variant = task_variant
        self.tokenizer = tokenizer

        # Load data
        samples = list(read_jsonl_lines(src))
        print('Loaded', len(samples), 'samples')

        # Replace empty values
        for sample in samples:
            if evidence_variant != 'n/a':
                for snippet in sample['snippets']:
                    if snippet['title'] is None:
                        snippet['title'] = ''
                    if snippet['text'] is None:
                        snippet['text'] = ''

        # Lowercase if specified
        if lower:
            for sample in samples:
                sample['claim'] = sample['claim'].lower()

                if evidence_variant != 'n/a':
                    for snippet in sample['snippets']:
                        snippet['title'] = snippet['title'].lower()
                        snippet['text'] = snippet['text'].lower()

        self.samples = [
            self._prepare_sample(sample) for sample in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def _prepare_sample(self, sample: Dict):
        """
        Concatenate claim with evidence based on the selected options.
        """
        sample['label'] = self.label_dict[sample['label']]

        # Only claim - ignore evidence
        if self.task_variant == 'only-claim':
            sample['text'] = self.tokenizer(sample['claim'], truncation=True)
        else:
            # In this case evidence is relevant - check which parts
            if self.evidence_variant == 'title':
                evidence = [snippet['title'] for snippet in sample['snippets']]
            elif self.evidence_variant == 'snippet':
                evidence = [snippet['text'] for snippet in sample['snippets']]
            else:
                evidence = [snippet['title'] + ' ' + snippet['text'] for snippet in sample['snippets']]

            # merge evidence pieces together
            evidence = '; '.join(evidence[:self.max_num_snippets])

            # Create final text
            if self.task_variant == 'only-evidence':
                sample['text'] = self.tokenizer(evidence, truncation=True)
            else:
                assert self.task_variant == 'complete'
                sample['text'] = self.tokenizer(sample['claim'], evidence, truncation=True)

        return sample
