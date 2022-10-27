from typing import Dict, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from experiment_code.util.file_util import read_jsonl_lines


class MultiFcDatasetBySnippet(Dataset):
    """
    Dataset of MultiFC. Each claim is concatenated with each associated snippet individually. Each combination of
    claim with each single snippet becomes one sample.
    """

    def __init__(self, src: str, tokenizer: PreTrainedTokenizer, task_variant: str, evidence_variant: str,
                 label_dict: Dict, max_num_snippets=99, lower: bool = False):
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

        assert task_variant in {'only-evidence', 'complete'}
        assert evidence_variant in {'title', 'snippet', 'title-snippet'}

        self.max_num_snippets = max_num_snippets
        self.label_dict = label_dict
        self.evidence_variant = evidence_variant
        self.task_variant = task_variant
        self.tokenizer = tokenizer

        # Read data
        samples = list(read_jsonl_lines(src))
        print('Loaded', len(samples), 'samples')

        # Replace empty values
        for sample in samples:
            for snippet in sample['snippets']:
                if snippet['title'] is None:
                    snippet['title'] = ''
                if snippet['text'] is None:
                    snippet['text'] = ''

        # Lowercase if specified
        if lower:
            for sample in samples:
                sample['claim'] = sample['claim'].lower()
                for snippet in sample['snippets']:
                    snippet['title'] = snippet['title'].lower()
                    snippet['text'] = snippet['text'].lower()

        self.samples = [
            claim_snippet_pair
            for sample in samples
            for claim_snippet_pair in self._prepare_sample(sample)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def _prepare_sample(self, sample) -> List[Dict]:
        """
        Concatenate claim with evidence based on the selected options.
        """

        sample['label'] = self.label_dict[sample['label']]

        if self.evidence_variant == 'title':
            evidence = [(snippet['title'], snippet) for snippet in sample['snippets']]
        elif self.evidence_variant == 'snippet':
            evidence = [(snippet['text'], snippet) for snippet in sample['snippets']]
        else:
            evidence = [(snippet['title'] + ' ' + snippet['text'], snippet) for snippet in sample['snippets']]

        # Ignore missing evidence
        evidence = [(e, snippet) for e, snippet in  evidence if len(e.strip()) > 0]

        claim_evidence_samples: List[Dict] = []
        claim: str = sample['claim']
        claim_id = sample['id']
        for ev, snippet in evidence:
            if self.task_variant == 'only-evidence':
                text = self.tokenizer(ev, truncation=True)
            else:
                assert self.task_variant == 'complete'
                text = self.tokenizer(sample['claim'], ev, truncation=True)

            claim_evidence_samples.append({
                'claim': claim,
                'snippet': snippet,
                'id': claim_id,
                'text': text,
                'label': sample['label'],
                'snippet_id': snippet['id']
             })

        return claim_evidence_samples
