
# Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation  
  
Source code and data of our paper [Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation](https://arxiv.org/abs/2210.13865) (to appear at EMNLP 2022).  
  
> **Abstract** Misinformation emerges in times of uncertainty when credible information is limited. This is challenging for NLP-based fact-checking as it relies on counter-evidence, which may not yet be available. Despite increasing interest in automatic fact-checking, it is still unclear if automated approaches can realistically refute harmful real-world misinformation. Here, we contrast and compare NLP fact-checking with how professional fact-checkers combat misinformation in the absence of counter-evidence. In our analysis, we show that, by design, existing NLP task definitions for fact-checking cannot refute misinformation as professional fact-checkers do for the majority of claims. We then define two requirements that the evidence in datasets must fulfill for realistic fact-checking: It must be (1) sufficient to refute the claim and (2) not leaked from existing fact-checking articles. We survey existing fact-checking datasets and find that all of them fail to satisfy both criteria. Finally, we perform experiments to demonstrate that models trained on a large-scale fact-checking dataset rely on leaked evidence, which makes them unsuitable in real-world scenarios. Taken together, we show that current NLP fact-checking cannot realistically combat real-world misinformation because it depends on unrealistic assumptions about counter-evidence in the data.  
  
## Installation  
  
  
1. Clone this repository and change the working directory:  
```shell  
git clone git@github.com:UKPLab/emnlp2022-missing-counter-evidence.git && cd emnlp2022-missing-counter-evidence 
```  
  
2. Create a new virtual environment:  
```shell  
python3.6 -m venv missing-evidence-env
source missing-evidence-env/bin/activate
```  
  
3. Install all dependencies within the new environment:  
```shell  
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```  


## Getting the Data
This section describes how to get the data used for the experiments and analyses. We base our experiments and analysis on the MultiFC dataset by [Augenstein et al. (2019)](https://aclanthology.org/D19-1475/).

1. Download the file `multi_fc_publicdata.zip` as provided by [Hansen et al., (2021)](https://aclanthology.org/2021.acl-short.12/) ([Link to the repository](https://github.com/casperhansen/fake-news-reasoning)) and place it at `./data/download/multi_fc_publicdata.zip`. The file includes the MultiFC dataset and separate splits for the experiments on Snopes and PolitiFact.  

2. To prepare the data for the experiments, run the following command:  
```shell 
python create-dataset.py ./data/download/multi_fc_publicdata.zip ./data
```  
It will extract the zip file and create directories for claims from Snopes (*snes*) and PolitiFact (*pomt*) within the `./data` directory.  It will create train/dev/test splits in the required format for both directories.

3. Verify that the command produced two new directories:
   - `./data/snes` with data splits using Snopes data.
   - `./data/pomt` with data splits using PolitiFact data.

## Pattern-based detection of leaked evidence
This section describes how we identified leaked evidence within the MultiFC dataset using patterns for content and URLs.

### Setup the config file
Verify that the paths in `configs/multifc-analyze-config.json` point to the correct directories (**default values should work fine**):
```json
{
  "path_multifc": "./data/download/multi_fc_publicdata",
  "analysis_dir": "./data/analysis-results"
}
```
Description of the fields:

| Field | Description |
|-------|-------------|
|`path_multifc`| Points to the extracted zip directory containing the MultiFC dataset.|
|`analysis_dir` | Points to an existing directory. Outputs will be stored in this directory.|

### Run pattern-based detection
Run the script:
```shell
python analyze-multifc.py identify configs/multifc-analyze-config.json
```
This script will run over all snippets in MultiFC to determine whether they contain leaked evidence based on the used patterns. It produces a file `leaking-all-multifc.csv`. The file contains the following columns:

| Column | Description                                                                                                                 |
|--------|-----------------------------------------------------------------------------------------------------------------------------|
| `claim_id` | Unique MultiFC ID for the claim for which the evidence snippet was retrieved.                                               |
| `snippet_id` | Numeric ID (1-10) of the rank of the evidence snippet for the claim.                                                        |
| `snippet_src` | URL from which the snippet was retrieved.                                                                                   |
| `leaked_url` | Boolean value indicating if the pattern-based approach considers the URL leaked (True) or not (False).                      |
| `leaked_words` | Boolean value indicating if the pattern-based approach considers the content leaked (True) or not (False).                  |
| `leaked` | Boolean value indicating if any pattern-based approach (URL or content) consider the snippet leaked (True) or not (False).  |
| `label` | Verdict of the respective claim from MultiFC.                                                                               |
| `misinformation` | Boolean value if the respective claim is considered misinformation based on our predefined list of misinformation verdicts. |


### Compute leaked evidence for misinformation
To compute the ratio of leaked evidence (Table 5 in the paper) run:
```shell
python analyze-multifc.py stats configs/multifc-analyze-config.json --misinfo
```
This script computes the ratio of leaked evidence based on the previously created file `leaking-all-multifc.csv`.

  
## Experiments on Snopes and PolitiFact
This section describes how to reproduce the experiments on leaked and unleaked evidence snippets using claims from Snopes and PolitiFact (via MultiFC).

The settings for each trained model in our experiments are stored within the `.json` files in the `experiments` directory.
Each file contains the following fields:

| Field              | Values                                                 | Description                                                                                                                                                                                                                |
|--------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `method`           | `"concat"`, `"split-by-snippet"` (default: `"concat"`) | Defines if all evidence snippets are concatenated with the claim and form one sample (`"concat"`), or if the claim is concatenated with each snippet individually and forms one sample per snippet (`"split-by-snippet"`). |
| `name`             | Text                                                   | Name of the stored model.                                                                                                                                                                                                  |
| `batch-size-train` | Number (default: `16`)                                 | Batch size during training.                                                                                                                                                                                                |
| `batch-size-eval`  | Number (default: `16`)                                 | Batch size during evaluation.                                                                                                                                                                                              |
| `lr`               | Float (default: `2e-5`)                                | Learning rate.                                                                                                                                                                                                             |
| `epochs`           | Number (default: `5`)                                  | Number of epochs for training.                                                                                                                                                                                             |
| `seed`             | Number (defaults: `1`, `2` and `3`)                    | Random seed.                                                                                                                                                                                                               |
| `model-name`       | Text (default: `bert-base-uncased`)                    | Name of the used language model to fine-tune.                                                                                                                                                                              |
| `lowercase`        | Bool (default: `True`)                                 | Determines if the text will be lowercased.                                                                                                                                                                                 |
| `max-input-len`    | Number (default: `512`)                                | Number of tokens after which the input gets truncated.                                                                                                                                                                     |
| `task-variant`     | `"complete"`, `"only-evidence"`,`"only-claim"`         | Defines which parts of the samples are used.                                                                                                                                                                               |
| `evidence-variant` | `"title-snippet"`, `"title"`, `"snippet"`              | Defines which parts of each evidence snippet are used. (Ignored when `{"task-variant": "only-claim"}`).                                                                                                                    |
| `evaluation`       | Dictionary                                             | Settings for evaluating the model on different data splits.                                                                                                                                                                |

### Training
To train a model, use an existing experiment file in the `experiments` directory (or create a new one).

For example, the file `snes-evidence_only-snippet-bert-2.json` defines the experiment to train a BERT base model to predict the correct verdict only based on the evidence snippet texts, using the random seed 2. To run this experiment, run:
```shell
python train.py train snes-evidence_only-snippet-bert-2.json
```
The fine-tuned model will be stored in the `results` directory.

### Prediction
Each experiment `.json` file defines in the field `evaluation` on which data the trained model gets evaluated. The format looks as follows:
```json
{
  "...": "...",
   "evaluation": {
      "<key-1>": {
         "name": "<name-of-prediction-file.jsonl>",
         "data": "<directory-of-data> // e.g. 'snes'",  
         "file": "<file-in-directory> // e.g. 'test.jsonl'"
      },
      "<key-2>": {
         "name": "...",
         "...": "..."
      }
   }
}
```

To create a `.jsonl` file containing the predictions run the following command:
```shell
python predict.py predict <experiment-file> <key>
```
By default, the keys `snes-all` and `pomt-all` are used to create predictions on the test set of Snopes and PolitiFact respectively. For example:
```shell
python predict.py predict snes-evidence_only-snippet-bert-2.json snes-all
```
The prediction files are stored within the `./predictions` directory.

### Evaluation
To evaluate the predictions based on all / only leaked / only unleaked samples (Table 7) run:
```shell
python evaluate.py evaluate ./configs/multifc-analyze-config.json <key> --exp <experiment-file-1> --exp <experiment-file-2> --exp <experiment-file-3>
```
The script evaluates and averages the performance across all provided experiments. To separate leaked from unleaked samples this script requires the result file from the pattern-based detection of leaked evidence.
For example:
```shell
python evaluate.py evaluate ./configs/multifc-analyze-config.json snes-all --exp experiments/snes-evidence_only-snippet-bert.json --exp experiments/snes-evidence_only-snippet-bert-2.json --exp experiments/snes-evidence_only-snippet-bert-3.json 
```

### Evaluate comparison between evidence-only and complete model (Appendix)
To evaluate the direct comparison per veracity label between the model trained on the entire input (claim and evidence), and the model trained only on the evidence as input run the following commands (Tables 11 & 12 in the Appendix):

For Snopes:
```shell
python evaluate.py compare ./configs/multifc-analyze-config.json snes-all ./experiments/snes-evidence_only-bert.json ./experiments/snes-complete-bert.json
```

For PolitiFact:
```shell
python evaluate.py compare ./configs/multifc-analyze-config.json pomt-all ./experiments/pomt-evidence_only-bert.json ./experiments/pomt-complete-bert.json
```

The script will automatically evaluate the provided experiments with all three seeds (`1`, `2`, `3`) and requires the predictions from all these trained models.


### Evaluate on *only* leaked/unleaked snippets (Appendix)
The previous experiments consider a sample leaked if it contains at least one leaked evidence snippet and unleaked otherwise. The model's prediction is based on all evidence snippets. We additionally evaluate our model on samples that *only* contain leaked evidence snippets or *only* contain unleaked evidence snippets.

To reproduce this evaluation, first create leaked and unleaked samples using the following command for *snes* and *pomt*:
```shell
python create-leaked-unleaked-splits.py ./configs/multifc-analyze-config.json snes test.jsonl
python create-leaked-unleaked-splits.py ./configs/multifc-analyze-config.json pomt test.jsonl
```
It will separately combine each claim with *only* leaked and *only* unleaked evidence snippets (as detected by the pattern-based approach). The resulting files are stored in the respective directories of Snopes (*snes*) and PolitiFact (*pomt*).

Create the predictions by using the same command as described above and the keys `snes-leaked` / `snes-unleaked` (Snopes) or `pomt-leaked` / `pomt-unleaked` (PolitiFact). For example:
```shell
python predict.py predict snes-evidence_only-snippet-bert-2.json snes-leaked
python predict.py predict snes-evidence_only-snippet-bert-2.json snes-unleaked
```

To reproduce the results in Tables 13 & 14 (Appendix) run (train & predict) the following experiments: They describe evidence-only models using three different random seeds.

| Snopes                           | PolitiFact                       |
|----------------------------------|----------------------------------|
| `snes-evidence_only-bert.json`   | `pomt-evidence_only-bert.json`   |
| `snes-evidence_only-bert-2.json` | `pomt-evidence_only-bert-2.json` |
 | `snes-evidence_only-bert-3.json` | `pomt-evidence_only-bert-3.json` |

To evaluate on **Snopes** finally run:
```shell
python evaluate.py leaked-unleaked snes-leaked snes-unleaked snes --exp ./experiments/snes-evidence_only-bert.json --exp ./experiments/snes-evidence_only-bert-2.json --exp ./experiments/snes-evidence_only-bert-3.json

```

To evaluate on **PolitiFact** finally run:
```shell
python evaluate.py leaked-unleaked pomt-leaked pomt-unleaked pomt --exp ./experiments/pomt-evidence_only-bert.json --exp ./experiments/pomt-evidence_only-bert-2.json --exp ./experiments/pomt-evidence_only-bert-3.json
```

## Manual Analysis
The results of our manual analysis can be found in [data/manual-analysis](data/manual-analysis).

## Contact
For questions please contact Max Glockner: [lastname]@ukp.informatik.tu-darmstadt.de.

## Citing the paper  
Please this bibtex when citing our work:  
```  
@inproceedings{glockner-etal-2022-missing,  
	 title = "{M}issing {C}ounter-{E}vidence {R}enders {NLP} {F}act-{C}hecking {U}nrealistic for {M}isinformation", 
	 author = "Glockner, Max and Hou, Yufang and Gurevych, Iryna", 
	 booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing", 
	 month = dec, 
	 year = "2022", 
	 address = "Abu Dhabi", 
	 publisher = "Association for Computational Linguistics", 
	 pages = "(to appear)"
 }  
```
